"""Standalone utilities for deduplicating and visualising DEL selection data.

Extracted from the DELicious package (https://github.com/VonBoss/DELicious).

Dependencies: numpy, pandas, polars, tqdm, matplotlib

Functions
---------
Deduplication
    deduplicate_files          -- process one or more parquet files end-to-end
    deduplicate_dataframe      -- deduplicate a single pandas DataFrame
    clean_selection_dataframe  -- drop invalid/empty rows from a DataFrame
    print_dedup_summary        -- pretty-print the dict returned by deduplicate_files

Utilities
    split_compound_id          -- split a compound ID string into [lib_id, bb1, bb2, bb3]

Plotting
    plot_selection_counts      -- histogram of count column values
    plot_selection_enrichment  -- histogram of Z-score / enrichment values
    plot_cubic_library_space   -- 3-D building-block space scatter plot for a single target
"""

from __future__ import annotations

import glob
import multiprocessing as mp
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INVALID_VALUES = ["", "[Dy]", "nan", "NULL", "null", "None", None]
DEFAULT_COMPOUND_PATTERN = r"^[a-zA-Z]+\d+(?:_\d+)?(?:-\d+){3,}(?:-\d+)?$"

_POLARS_AGG: dict[str, Any] = {
    "sum":      lambda col: pl.col(col).sum(),
    "mean":     lambda col: pl.col(col).mean(),
    "median":   lambda col: pl.col(col).median(),
    "min":      lambda col: pl.col(col).min(),
    "max":      lambda col: pl.col(col).max(),
    # Unweighted Stouffer: sum(z) / sqrt(n)
    "stouffer": lambda col: (
        pl.col(col).sum() / pl.col(col).count().cast(pl.Float64).sqrt()
    ).alias(col),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DedupResult:
    file: str
    status: str
    original_rows: int = 0
    clean_rows: int = 0
    deduplicated_rows: int = 0
    columns_aggregated: list[str] = field(default_factory=list)
    output_path: str | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Deduplication helpers
# ---------------------------------------------------------------------------

def stouffer_unweighted(z_scores: np.ndarray) -> float:
    """Unweighted Stouffer method: Z = sum(z) / sqrt(n)."""
    z_scores = np.asarray(z_scores, dtype=float)
    if z_scores.size == 0:
        return float("nan")
    if z_scores.size == 1:
        return float(z_scores[0])
    return float(z_scores.sum() / np.sqrt(z_scores.size))


def infer_aggregation_scheme(
    columns: list[str],
    aggregation: dict[str, str] | None = None,
) -> dict[str, str]:
    """Infer a column → aggregation-method mapping from column names.

    Parameters
    ----------
    columns:
        Column names present in the DataFrame.
    aggregation:
        Optional explicit mapping of ``column → method``. If ``None``, defaults
        are applied: ``"sum"`` for ``*_count`` columns, ``"stouffer"`` for
        ``*_zscore`` / ``*_score`` columns.

    Returns
    -------
    dict[str, str]
        Mapping of column name → aggregation method for columns that exist.
        Columns specified in *aggregation* that are absent from *columns* are
        dropped with a :class:`UserWarning`.
    """
    if aggregation is not None:
        missing = [col for col in aggregation if col not in columns]
        if missing:
            warnings.warn(
                f"Aggregation columns not found and will be skipped: {missing}",
                UserWarning,
                stacklevel=2,
            )
        return {col: method for col, method in aggregation.items() if col in columns}

    return {
        col: ("sum" if col.endswith("_count") else "stouffer")
        for col in columns
        if col.endswith("_count") or col.endswith("_zscore") or col.endswith("_score")
    }


def _clean_pl(
    df: pl.DataFrame,
    compound_col: str | None,
    smiles_col: str | None,
    required_cols: list[str],
    compound_pattern: str,
) -> pl.DataFrame:
    """Polars-native row cleaning — replaces invalid strings, drops nulls, validates fields."""
    invalid_strings = [v for v in INVALID_VALUES if v is not None]
    str_cols = [c for c in [compound_col, smiles_col] if c and c in df.columns]

    if str_cols:
        df = df.with_columns([
            pl.when(pl.col(col).is_in(invalid_strings))
            .then(None)
            .otherwise(pl.col(col))
            .alias(col)
            for col in str_cols
        ])

    null_check_cols = [c for c in str_cols + required_cols if c in df.columns]
    if null_check_cols:
        df = df.drop_nulls(subset=null_check_cols)

    if compound_col and compound_col in df.columns:
        df = df.filter(pl.col(compound_col).str.contains(compound_pattern))

    if smiles_col and smiles_col in df.columns:
        df = df.filter(
            pl.col(smiles_col).str.contains("[Cc]")
            & (pl.col(smiles_col).str.len_chars() > 10)
        )

    return df


def _deduplicate_pl(
    df: pl.DataFrame,
    dedup_col: str,
    agg_scheme: dict[str, str],
) -> pl.DataFrame:
    """Deduplicate a Polars DataFrame, aggregating named columns and keeping first for the rest."""
    exprs: list[pl.Expr] = [_POLARS_AGG[method](col) for col, method in agg_scheme.items()]

    for col in df.columns:
        if col != dedup_col and col not in agg_scheme:
            exprs.append(pl.col(col).first())

    result = df.group_by(dedup_col).agg(exprs)

    # Restore input column order
    col_order = [c for c in df.columns if c in result.columns]
    return result.select(col_order)


# ---------------------------------------------------------------------------
# Public API — DataFrames
# ---------------------------------------------------------------------------

def clean_selection_dataframe(
    df: pd.DataFrame,
    compound_col: str | None = "compound",
    smiles_col: str | None = "SMILES",
    required_cols: list[str] | None = None,
    compound_pattern: str = DEFAULT_COMPOUND_PATTERN,
) -> pd.DataFrame:
    """Drop empty rows and apply light DEL-specific field validation.

    Parameters
    ----------
    df:
        Input DataFrame.
    compound_col:
        Column containing compound IDs. Rows whose IDs do not match
        *compound_pattern* are removed. Pass ``None`` to skip this check.
    smiles_col:
        Column containing SMILES strings. Rows without a carbon atom or with
        SMILES shorter than 10 characters are removed. Pass ``None`` to skip.
    required_cols:
        Additional columns that must be non-null; rows with nulls in any of
        these are dropped.
    compound_pattern:
        Regex used to validate compound IDs (default matches BCM OpenDEL format).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with index reset.
    """
    return _clean_pl(
        pl.from_pandas(df),
        compound_col=compound_col,
        smiles_col=smiles_col,
        required_cols=required_cols or [],
        compound_pattern=compound_pattern,
    ).to_pandas()


def deduplicate_dataframe(
    df: pd.DataFrame,
    dedup_col: str = "SMILES",
    aggregation: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Deduplicate rows by *dedup_col* and aggregate numeric fields.

    All columns from the input DataFrame are preserved. Columns included in
    *aggregation* (or inferred automatically) are aggregated using the specified
    method; all remaining columns retain the first observed value within each
    group. Input column order is preserved in the output.

    Aggregation uses Polars internally for performance.

    Parameters
    ----------
    df:
        Input DataFrame.
    dedup_col:
        Column to group by when deduplicating. Defaults to ``"SMILES"``.
    aggregation:
        Mapping of ``column → method`` specifying how to aggregate each column.
        Valid methods:

        - ``"sum"`` — sum of values
        - ``"min"`` / ``"max"`` — minimum / maximum value
        - ``"mean"`` / ``"median"`` — arithmetic mean / median
        - ``"stouffer"`` — unweighted Stouffer's method: ``sum(z) / sqrt(n)``

        If ``None``, the scheme is inferred automatically: columns ending in
        ``_count`` → ``"sum"``; ``_zscore`` or ``_score`` → ``"stouffer"``.

    Returns
    -------
    pd.DataFrame
        Deduplicated DataFrame with input column order and index reset.

    Raises
    ------
    KeyError
        If *dedup_col* is not present in *df*.
    ValueError
        If an unknown aggregation method is specified.

    Examples
    --------
    >>> out = deduplicate_dataframe(df, dedup_col="SMILES")

    >>> out = deduplicate_dataframe(
    ...     df,
    ...     dedup_col="SMILES",
    ...     aggregation={
    ...         "target_count": "sum",
    ...         "target_zscore": "stouffer",
    ...         "ntc_count": "sum",
    ...         "ntc_zscore": "stouffer",
    ...     },
    ... )
    """
    if dedup_col not in df.columns:
        raise KeyError(f"Missing dedup column: {dedup_col!r}")

    agg_scheme = infer_aggregation_scheme(df.columns.tolist(), aggregation)

    for col, method in agg_scheme.items():
        if method not in _POLARS_AGG:
            raise ValueError(f"Unknown aggregation method {method!r} for column {col!r}")

    return _deduplicate_pl(pl.from_pandas(df), dedup_col, agg_scheme).to_pandas()


# ---------------------------------------------------------------------------
# Public API — Files
# ---------------------------------------------------------------------------

def _process_file(
    file_path: str,
    output_dir: str,
    suffix: str,
    dedup_col: str,
    compound_col: str | None,
    smiles_col: str,
    aggregation: dict[str, str] | None,
) -> DedupResult:
    """Process one parquet file entirely in Polars — no pandas conversions."""
    file_name = Path(file_path).stem
    try:
        df = pl.read_parquet(file_path)
        original_rows = len(df)

        if dedup_col not in df.columns:
            return DedupResult(
                file=file_name, status="failed", error=f"Missing dedup_col: {dedup_col!r}"
            )

        agg_scheme = infer_aggregation_scheme(df.columns, aggregation)
        effective_smiles_col = smiles_col if dedup_col == smiles_col else None

        clean_df = _clean_pl(
            df,
            compound_col=compound_col,
            smiles_col=effective_smiles_col,
            required_cols=list(agg_scheme.keys()),
            compound_pattern=DEFAULT_COMPOUND_PATTERN,
        )

        dedup_df = _deduplicate_pl(clean_df, dedup_col=dedup_col, agg_scheme=agg_scheme)

        out_path = Path(output_dir) / f"{file_name}{suffix}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dedup_df.write_parquet(out_path, compression="zstd")

        return DedupResult(
            file=file_name,
            status="success",
            original_rows=int(original_rows),
            clean_rows=int(len(clean_df)),
            deduplicated_rows=int(len(dedup_df)),
            columns_aggregated=list(agg_scheme.keys()),
            output_path=str(out_path),
        )

    except Exception as exc:
        return DedupResult(file=file_name, status="failed", error=f"{type(exc).__name__}: {exc}")


def deduplicate_files(
    glob_pattern: str,
    output_dir: str,
    suffix: str = "_deduplicated",
    dedup_col: str = "SMILES",
    compound_col: str | None = "compound",
    smiles_col: str = "SMILES",
    aggregation: dict[str, str] | None = None,
    max_workers: int = 1,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Deduplicate and aggregate DEL parquet files matching a glob pattern.

    Loads each parquet file, cleans rows with invalid compound IDs or SMILES,
    deduplicates by *dedup_col*, and writes the result to *output_dir*. All
    input columns are preserved; non-aggregated columns retain their first
    observed value per group.

    Parameters
    ----------
    glob_pattern:
        Glob pattern matching the parquet files to process, e.g.
        ``"selections_raw/E3_ligase_v3/*.parquet"``.
    output_dir:
        Directory where deduplicated files will be written (created if needed).
    suffix:
        String appended to each output filename before ``.parquet``.
        Defaults to ``"_deduplicated"``.
    dedup_col:
        Column used for grouping during deduplication. Defaults to ``"SMILES"``.
    compound_col:
        Column containing compound IDs, used for BCM-pattern row validation
        during cleaning. Pass ``None`` to skip compound validation.
    smiles_col:
        Column containing SMILES strings, used for structural validation during
        cleaning (only applied when *dedup_col* == *smiles_col*).
    aggregation:
        Explicit ``column → method`` mapping. Valid methods: ``"sum"``,
        ``"min"``, ``"max"``, ``"mean"``, ``"median"``, ``"stouffer"``.
        If ``None``, inferred from column name suffixes (see
        :func:`infer_aggregation_scheme`).
    max_workers:
        Number of parallel workers. ``1`` runs sequentially (default);
        ``0`` or ``None`` auto-detects from CPU count and file count.
        When ``> 1``, each worker runs in a fresh process (spawn) to avoid
        deadlocks with Polars's internal thread pool.
    show_progress:
        Display a tqdm progress bar while processing. Default ``True``.

    Returns
    -------
    dict
        Summary with keys ``"files_processed"``, ``"files_failed"``, and
        ``"details"`` (list of :class:`DedupResult` dicts).

    Examples
    --------
    >>> summary = deduplicate_files(
    ...     "selections_raw/E3_ligase_v3/*.parquet",
    ...     output_dir="deduplicated/",
    ... )

    >>> summary = deduplicate_files(
    ...     "selections_raw/E3_ligase_v3/*.parquet",
    ...     output_dir="deduplicated/",
    ...     aggregation={"target_count": "sum", "target_zscore": "stouffer"},
    ...     max_workers=4,
    ... )
    """
    parquet_files = sorted(glob.glob(glob_pattern))
    if not parquet_files:
        return {"files_processed": 0, "files_failed": 0, "details": []}

    if max_workers in (None, 0):
        max_workers = min(os.cpu_count() or 1, len(parquet_files))

    results: list[DedupResult] = []

    if max_workers == 1:
        for fp in tqdm(parquet_files, desc="Processing files", disable=not show_progress):
            results.append(
                _process_file(
                    file_path=fp,
                    output_dir=output_dir,
                    suffix=suffix,
                    dedup_col=dedup_col,
                    compound_col=compound_col,
                    smiles_col=smiles_col,
                    aggregation=aggregation,
                )
            )
    else:
        # spawn avoids deadlocks from forking Polars's internal thread pool on Linux
        mp_context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as ex:
            fut_to_file = {
                ex.submit(
                    _process_file, fp, output_dir, suffix,
                    dedup_col, compound_col, smiles_col, aggregation,
                ): fp
                for fp in parquet_files
            }
            with tqdm(
                total=len(parquet_files), desc="Processing files", disable=not show_progress
            ) as pbar:
                for fut in as_completed(fut_to_file):
                    results.append(fut.result())
                    pbar.update(1)

    successful = [r for r in results if r.status == "success"]
    failed = [r for r in results if r.status == "failed"]

    return {
        "files_processed": len(successful),
        "files_failed": len(failed),
        "details": [r.__dict__ for r in results],
    }


def print_dedup_summary(result: dict[str, Any]) -> None:
    """Print a human-readable summary of results returned by :func:`deduplicate_files`.

    Parameters
    ----------
    result:
        The dict returned by :func:`deduplicate_files`.
    """
    details = result.get("details", [])
    n_processed = result.get("files_processed", 0)
    n_failed = result.get("files_failed", 0)

    def _pct(part: int, whole: int) -> str:
        if whole == 0:
            return "  n/a"
        return f"{(part - whole) / whole * 100:+.1f}%"

    print("\n── Deduplication Summary ────────────────────────────────")
    print(f"  Files processed : {n_processed:,}")
    print(f"  Files failed    : {n_failed:,}")

    successful = [d for d in details if d["status"] == "success"]
    failed = [d for d in details if d["status"] == "failed"]

    if successful:
        total_orig  = sum(d["original_rows"]     for d in successful)
        total_clean = sum(d["clean_rows"]         for d in successful)
        total_dedup = sum(d["deduplicated_rows"]  for d in successful)
        if n_processed > 1:
            print(f"\n  Totals across {n_processed} files:")
            print(f"    Original rows  : {total_orig:>12,}")
            print(f"    After cleaning : {total_clean:>12,}  ({_pct(total_clean, total_orig)})")
            print(f"    After dedup    : {total_dedup:>12,}  ({_pct(total_dedup, total_orig)})")

        print()
        for d in successful:
            print(f"  {d['file']}")
            print(f"    Original rows  : {d['original_rows']:>12,}")
            print(f"    After cleaning : {d['clean_rows']:>12,}  ({_pct(d['clean_rows'], d['original_rows'])})")
            print(f"    After dedup    : {d['deduplicated_rows']:>12,}  ({_pct(d['deduplicated_rows'], d['original_rows'])})")
            if d["columns_aggregated"]:
                cols = ", ".join(d["columns_aggregated"])
                print(f"    Aggregated     : {cols}")
            if d["output_path"]:
                print(f"    Output         : {d['output_path']}")

    if failed:
        print()
        print("  Failures:")
        for d in failed:
            print(f"    {d['file']}: {d['error']}")

    print("─" * 55)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def split_compound_id(
    compound_id: str,
    delimiter: str = "-",
    num_of_cycles: int = 3,
) -> list[str]:
    """Split a compound ID into its constituent parts.

    Parameters
    ----------
    compound_id:
        The compound ID string, e.g. ``"qDOS1-0001-0042-0017"``.
    delimiter:
        Delimiter to split on. Defaults to ``"-"``.
    num_of_cycles:
        Number of splits (from the right). Defaults to ``3``, yielding
        ``[lib_id, bb1_id, bb2_id, bb3_id]``.

    Returns
    -------
    list[str]
        Constituent parts, e.g. ``["qDOS1", "0001", "0042", "0017"]``.
    """
    return compound_id.rsplit(delimiter, num_of_cycles)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _parse_bb_columns(
    df: pd.DataFrame,
    compound_col: str,
    compound_delimiter: str,
) -> tuple[pd.DataFrame, str, str, str]:
    """Return (df, bb1_name, bb2_name, bb3_name) by parsing compound IDs."""
    if compound_col not in df.columns:
        raise ValueError(f"compound_col {compound_col!r} not found in DataFrame.")
    parts = df[compound_col].str.rsplit(compound_delimiter, n=3, expand=True)
    if parts.shape[1] < 4:
        raise ValueError(
            f"Column {compound_col!r} could not be split into 4 parts using "
            f"delimiter {compound_delimiter!r}. Expected format: lib-bb1-bb2-bb3."
        )
    df = df.copy()
    df["_parsed_bb1"] = pd.to_numeric(parts[1], errors="coerce")
    df["_parsed_bb2"] = pd.to_numeric(parts[2], errors="coerce")
    df["_parsed_bb3"] = pd.to_numeric(parts[3], errors="coerce")
    return df, "_parsed_bb1", "_parsed_bb2", "_parsed_bb3"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_selection_counts(
    target_name: str,
    target_data: pd.DataFrame,
    total_compounds: int | None = 890_000_000,
    count_col: str = "target_count",
    max_count: int = 20,
    min_count: int = 1,
    yscale: str = "log",
    ylim: tuple[float, float] | None = None,
):
    """Plot observed count distribution, optionally including unobserved count=0 bucket."""
    bins = range(min_count, max_count + 1)
    plt.figure(figsize=(8, 6))
    plt.hist(target_data[count_col], bins=bins, color="skyblue", edgecolor="black", label="Observed")

    if total_compounds is not None:
        remaining = total_compounds - len(target_data)
        plt.bar(0, max(0, remaining), color="orange", edgecolor="black", label="Unobserved (count=0)")

    plt.title(f"Distribution of {count_col} for {target_name}")
    plt.xlabel(count_col)
    plt.ylabel("Frequency")
    plt.yscale(yscale)
    if ylim is not None:
        plt.ylim(ylim)
    plt.legend()
    return plt.gcf()


def plot_selection_enrichment(
    target_name: str,
    target_data: pd.DataFrame,
    enrichment_col: str = "target_zscore",
    min_value: float = 0.0,
    max_value: float = 3.0,
    yscale: str = "linear",
    ylim: tuple[float, float] | None = None,
):
    """Plot enrichment distribution for one target dataset."""
    bins = int((max_value - min_value) * 30)
    plt.figure(figsize=(8, 6))
    plt.hist(
        target_data[enrichment_col],
        bins=bins,
        range=(min_value, max_value),
        color="skyblue",
        edgecolor="black",
        label="Observed",
    )

    plt.title(f"Distribution of {enrichment_col} for {target_name}")
    plt.xlabel(enrichment_col)
    plt.ylabel("Frequency")
    plt.yscale(yscale)
    if ylim is not None:
        plt.ylim(ylim)
    plt.legend()
    return plt.gcf()


def plot_cubic_library_space(
    df: pd.DataFrame,
    compound_col: str = "compound",
    compound_delimiter: str = "-",
    size_col: str = "zscore_PGK2",
    size_min: float = 1.0,
    color_col: str = "zscore_PGK2",
    color_min: float | None = None,
    color_max: float | None = None,
    max_marker_size: int = 250,
    min_marker_size: int = 5,
    cmap: str = "viridis",
    title: str | None = None,
) -> plt.Figure:
    """Plot DEL building-block space as a 3D scatter for a single target.

    Building-block coordinates are parsed from *compound_col* by splitting on
    *compound_delimiter* (right-to-left, three times), yielding
    ``[lib_id, bb1, bb2, bb3]``. The three BB parts become the X/Y/Z axes.

    Parameters
    ----------
    df:
        DataFrame for a single target. Must contain *compound_col*, *size_col*,
        and *color_col*.
    compound_col:
        Column containing compound IDs (e.g. ``"qDOS1-0001-0042-0017"``).
    compound_delimiter:
        Delimiter used to split compound IDs. Defaults to ``"-"``.
    size_col:
        Column whose values drive marker size. Rows at or below *size_min* are
        excluded from the plot entirely.
    size_min:
        Minimum value of *size_col* required to plot a point.
    color_col:
        Column whose values determine marker colour.
    color_min:
        If provided, rows where *color_col* is at or below this value are also
        excluded. ``None`` applies no additional filter.
    color_max:
        If provided, clamp the colour scale at this value. Points above the
        threshold are still plotted but rendered in the maximum colour of the
        colormap. ``None`` applies no upper clamp.
    max_marker_size, min_marker_size:
        Marker area range mapped linearly to normalised *size_col* values.
    cmap:
        Matplotlib colormap name for marker colour. Defaults to ``"viridis"``.
    title:
        Plot title. ``None`` produces no title.

    Returns
    -------
    plt.Figure
        The generated figure.
    """
    # Filter before parsing — compound ID splitting is expensive on large frames
    mask = df[size_col].astype(float) > size_min
    if color_min is not None:
        mask &= df[color_col].astype(float) > color_min

    if mask.sum() == 0:
        raise ValueError(
            f"No rows remain after filtering {size_col!r} > {size_min}"
            + (f" and {color_col!r} > {color_min}" if color_min is not None else "")
            + "."
        )

    df, bb1, bb2, bb3 = _parse_bb_columns(df[mask].reset_index(drop=True), compound_col, compound_delimiter)

    size_vals = df[size_col].astype(float)
    size_norm = (size_vals - size_vals.min()) / (size_vals.max() - size_vals.min() + 1e-9)
    areas = size_norm * (max_marker_size - min_marker_size) + min_marker_size

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(  # type: ignore[arg-type]
        df[bb1].to_numpy(dtype=float),
        df[bb2].to_numpy(dtype=float),
        df[bb3].to_numpy(dtype=float),
        s=areas.to_numpy(dtype=float),
        c=df[color_col].to_numpy(dtype=float),
        cmap=cmap,
        vmax=color_max,
        alpha=0.4,
    )

    ax.set_xlabel("Building Block 1")
    ax.set_ylabel("Building Block 2")
    ax.set_zlabel("Building Block 3")
    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    if title:
        plt.title(title)
    cbar = plt.colorbar(sc, pad=0.1)
    cbar.set_label(color_col)

    return fig
