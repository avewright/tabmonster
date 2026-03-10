# Roadmap

This scaffold is meant to support a multi-stage program toward a TabPFN-competitive model.

## Phase 1: Infrastructure

- Stable config-driven training entrypoints
- Synthetic task generation
- Real-dataset finetuning
- Reproducible checkpointing and metrics

## Phase 2: Better priors

- Expand synthetic generators beyond linear-plus-threshold tasks
- Add causal graphs, missingness, heavy-tailed marginals, class imbalance, and heteroscedastic regression
- Sample task families with curriculum and temperature control

## Phase 3: PFN-style objectives

- Train on train/test splits inside each episode
- Predict posterior labels for query rows conditioned on context rows
- Support permutation-invariant row conditioning and masked targets

## Phase 4: Benchmark rigor

- Add OpenML / TabZilla style benchmark runners
- Track mean rank, normalized regret, calibration, and latency
- Compare against TabPFN, CatBoost, LightGBM, XGBoost, and FT-Transformer variants

## Phase 5: Scaling

- Multi-GPU training
- Mixed precision
- Experiment tracking
- Artifact versioning for synthetic generators and pretrained checkpoints
