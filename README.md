# DEL Selection Deduplication — PGK2 DREAM Challenge

Deduplication and visualization tools for DNA-Encoded Library (DEL) selection data,
released for the [DREAM × CACHE Target 2035 Drug Discovery Challenge](https://www.synapse.org/Synapse:syn75349604/wiki/641044).

We demonstrate one approach to handling duplicate chemical structures in raw selection
data and provide flexible tools for contestants to try alternative strategies.

## Background

BCM OpenDEL libraries can assign multiple compound IDs to the same chemical structure due to:
- **Protecting group redundancy** — building blocks with identical deprotected structures assigned distinct IDs
- **Codon redundancy** — identical building blocks encoded by different DNA sequences

The raw PGK2 selection file contains **7,703,070 rows** but only **7,487,567 unique structures**
(~2.8% duplicates). Without deduplication, the same structure appears as multiple independent
data points, inflating enrichment signals and causing data leakage in train/test splits.

## Contents

| File                               | Description                                          |
| ---------------------------------- | ---------------------------------------------------- |
| `del_deduplicate.py`               | Standalone deduplication and visualization functions |
| `deduplicate_aggregate_demo.ipynb` | Worked example on the PGK2 selection data            |
| `environment.yaml`                 | Conda environment for one-command setup              |

## Setup

```bash
conda env create -f environment.yaml
conda activate del-dedup
```

Requires Python 3.11. Key dependencies: `polars`, `pandas`, `matplotlib`, `pyarrow`.

## Quick start

See [`deduplicate_aggregate_demo.ipynb`](deduplicate_aggregate_demo.ipynb) for a full walkthrough, including:
- Deduplicating and aggregating raw parquet files
- Visualizing count and Z-score distributions
- Exploring enrichment in 3D building-block space

```python
from del_deduplicate import deduplicate_files, print_dedup_summary

result = deduplicate_files(
    glob_pattern="PGK2_selection_raw.parquet",
    output_dir="selection_deduplicated/",
    dedup_col="SMILES",
    aggregation={
        "count_PGK2":                "sum",
        "count_PGK2_with_inhibitor": "sum",
        "count_NTC":                 "sum",
        "zscore_PGK2":               "stouffer",
        "zscore_PGK2_with_inhibitor": "stouffer",
        "zscore_NTC":                "stouffer",
        "historic_hits":             "max",
    },
)
print_dedup_summary(result)
```

## Data

Raw selection data is available on Synapse as part of the [DREAM challenge](https://www.synapse.org/Synapse:syn75349604/wiki/641044).
Deduplicated data is not provided — contestants are free to handle duplicates however they see fit.
