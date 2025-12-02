# Bid Optimization Refactoring - Changes Summary

## Overview
Simplified the bid optimization workflow by:
1. Auto-generating unique keyword embeddings during data preparation
2. Using embedding method as a single parameter instead of multiple file arguments
3. Auto-computing missing embeddings on-demand if needed

## Changes

### 1. `tidy_get_data.py` → Enhanced Output
**New Behavior:**
- Saves `unique_keyword_embeddings_{embedding_method}.csv` alongside training/test splits
- Example outputs:
  - `clean_data/unique_keyword_embeddings_tfidf.csv`
  - `clean_data/unique_keyword_embeddings_bert.csv`

**Command:**
```bash
python scripts/tidy_get_data.py --embedding-method bert
```

### 2. `utils/data_pipeline.py` → New Utility Function
**New Function: `load_or_update_embeddings()`**
- Loads embeddings from file
- Computes missing keywords if not in file
- Auto-updates embeddings file with newly computed ones
- Usage:
```python
from utils import load_or_update_embeddings

embedding_df = load_or_update_embeddings(
    keywords=['keyword1', 'keyword2'],
    embeddings_file='clean_data/unique_keyword_embeddings_bert.csv',
    embedding_method='bert',
    n_components=50
)
```

### 3. `bid_optimization.py` → Simplified Interface
**Old Interface:**
```bash
python scripts/bid_optimization.py \
    --embedding-file raw_data/unique_keyword_embeddings.csv \
    --conv-model models/lr_conversion.json \
    --clicks-model models/lr_clicks.json \
    --output opt_results/optimized_bids.csv \
    --budget 68096.51
```

**New Interface:**
```bash
# BERT embeddings (default)
python scripts/bid_optimization.py --budget 68096.51

# TF-IDF embeddings
python scripts/bid_optimization.py --embedding-method tfidf --budget 68096.51

# Custom budget
python scripts/bid_optimization.py --embedding-method bert --budget 50000 --max-bid 150
```

**Behavior:**
- Automatically derives model paths: `models/lr_{embedding_method}_conversion.json`
- Automatically derives embeddings file: `clean_data/unique_keyword_embeddings_{embedding_method}.csv`
- Output saves to: `opt_results/optimized_bids_{embedding_method}.csv`
- If embeddings file doesn't have all keywords, computes and updates it automatically

### 4. New CLI Arguments for `bid_optimization.py`
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--embedding-method` | str | `bert` | Choice: `tfidf` or `bert` |
| `--budget` | float | `68096.51` | Total budget |
| `--max-bid` | float | `100.0` | Max individual bid |
| `--max-active` | int | `14229` | Max active keywords |
| `--data-dir` | str | `clean_data` | Dir with embeddings |
| `--models-dir` | str | `models` | Dir with models |

**Removed Arguments:**
- `--embedding-file` (auto-derived)
- `--conv-model` (auto-derived)
- `--clicks-model` (auto-derived)
- `--output` (auto-derived)

## File Structure
```
project/
├── clean_data/
│   ├── train_tfidf.csv
│   ├── test_tfidf.csv
│   ├── ad_opt_data_tfidf.csv
│   ├── unique_keyword_embeddings_tfidf.csv  ← NEW
│   ├── train_bert.csv
│   ├── test_bert.csv
│   ├── ad_opt_data_bert.csv
│   └── unique_keyword_embeddings_bert.csv   ← NEW
├── models/
│   ├── lr_tfidf_conversion.json
│   ├── lr_tfidf_clicks.json
│   ├── lr_bert_conversion.json
│   └── lr_bert_clicks.json
└── opt_results/
    ├── optimized_bids_tfidf.csv             ← NEW
    └── optimized_bids_bert.csv              ← NEW
```

## Benefits
1. **Simpler CLI:** Single `--embedding-method` parameter instead of 4 file paths
2. **Auto-discovery:** Models and embeddings auto-found based on embedding method
3. **Flexibility:** Missing embeddings computed on-demand if needed
4. **Consistency:** All embedding types saved during data prep
5. **Reproducibility:** Model names encode embedding method

## Workflow Example
```bash
# 1. Prepare data with BERT embeddings
python scripts/tidy_get_data.py --embedding-method bert

# 2. Train models (generates models/lr_bert_*.json)
python scripts/prediction_modeling.py --embedding-method bert --models rf,xgb

# 3. Optimize bids (auto-discovers models and embeddings)
python scripts/bid_optimization.py --embedding-method bert --budget 68096.51

# Results saved to opt_results/optimized_bids_bert.csv
```
