# Tabula v1 Training Run — Results Summary
# Generated: 2026-03-16

## Pretraining Results

| Metric | Value |
|---|---|
| Best checkpoint | Step 45,000 |
| Best val loss | **0.2295** |
| Rows seen at best | 23,040,000 |
| Final step | 61,825 / 200,000 |
| Total rows trained | 31,654,400 |
| Training time | ~3 hours |
| Hardware | RTX A4500 (20 GB) |
| Throughput | ~3,000 rows/s |

Model pushed to: https://huggingface.co/avewright/tabula-v1

## Corpus State at Session End

### HuggingFace repo: avewright/tabula-pretraining-corpus-v2
- Total shards uploaded: **541** (train-00395.parquet through train-00540.parquet this session)
- Estimated total rows: ~1.6 billion
- Estimated size: ~160 GB

### Real Dataset Registry (artifacts/loop_discovery_registry.json)

| Source | OK | Schema Fail | Download Fail | Total Attempted |
|---|---|---|---|---|
| PMLB | 422 | 0 | 1 | 423 |
| OpenML | 2,949 | 1,900 | 37 | 4,886 |
| HuggingFace | 0 | 1 | 66 | 67 |
| **Total** | **3,371** | **1,901** | **104** | **5,376** |

## Dataset Exhaustion Notes

### PMLB — FULLY EXHAUSTED
All 422 of 423 known PMLB datasets have been processed and included in the corpus.
- 1 failure: `chess` (download error — likely missing from pmlb package)
- No further PMLB datasets can be added without an upstream pmlb library update.

### OpenML — LARGELY EXHAUSTED (~60% yield)
4,886 unique OpenML datasets attempted. 2,949 passed. Not resumable for new data
without major pipeline changes:
- **1,900 schema_fail** — Almost entirely datasets rejected as "too small":
  single-output (1-column) datasets with very few rows or features.
  Top failure reason: `too small: (53, 1)` (421 instances), meaning 53 rows and
  only 1 feature column after target removal — unusable for tabular pretraining.
  These are structurally unrecoverable without lowering quality filters.
- **37 download_fail** — Transient timeouts or corrupted OpenML files.
- Unknown tail: OpenML has 40,000+ total datasets but the vast majority are
  image/text or single-column tasks. Pagination of tabular-tagged datasets
  effectively exhausted ~5,000 viable candidates.

### HuggingFace Tabular — NOT VIABLE WITH CURRENT CATALOG
67 datasets attempted from `catalogs/hf_tabular.json`. All failed:
- 66 download failures (auth timeouts, missing splits, format mismatches)
- 1 schema fail
- The HF catalog needs manual curation and expansion to be useful.

## Next Steps for Corpus Expansion

1. **Kaggle integration** — `catalogs/kaggle_tabular.json` exists but connector
   is unimplemented. Kaggle has 10,000+ tabular competition datasets.
2. **UCI ML Repository** — Not yet integrated. ~600 tabular datasets.
3. **HF catalog rebuild** — Manually vet 50+ high-quality tabular HF datasets.
4. **Lower OpenML quality threshold** — Accept datasets with n_features >= 2
   (currently likely filtering out 2-3 feature datasets). Could recover ~200 more.
5. **Synthetic scaling** — Already running; can increase synthetic volume
   independently of real data exhaustion.

## Training Next Steps

1. Resume pretraining from step 61,825 (`artifacts/pretrain_corpus_v1/latest.pt`)
   targeting 200,000 steps total.
2. Current best val loss: 0.2295 (achieved at step 45,000).
3. Consider fine-tuning on specific benchmarks (Adult, Titanic, etc.) to evaluate
   downstream transfer quality.
