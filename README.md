# LSST Time Series Classification

PyTorch pipeline for multivariate time series classification on the LSST dataset. Five architectures are supported — two pre-trained foundation model adapters and three from-scratch models — plus a soft-voting ensemble.

## Models

| `--model.name`   | Type               | Params   | Pre-trained |
|------------------|--------------------|----------|-------------|
| `inception_time` | InceptionTime CNN  | ~1.9M    | No          |
| `patch_tst`      | PatchTST Transformer | ~1.0M  | No          |
| `units`          | UniTS Transformer  | ~870K    | No          |
| `chronos`        | Chronos-2 encoder  | ~46M     | Yes (forecasting) |
| `moment`         | MOMENT-1-large     | ~341M    | Yes (embedding)   |


## Setup

```bash
uv sync
```

Requires Python 3.12+. GPU strongly recommended for foundation models.

## Training

Each model has a self-contained config in `cfg/`. Run from the project root:

```bash
# From-scratch models (single stage)
uv run python -m dlts.train --cfg cfg/inception_time.yaml
uv run python -m dlts.train --cfg cfg/patch_tst.yaml
uv run python -m dlts.train --cfg cfg/units.yaml

# Foundation model adapters (2-stage: linear probing -> partial fine-tuning)
uv run python -m dlts.train --cfg cfg/chronos.yaml
uv run python -m dlts.train --cfg cfg/moment.yaml
```

Override any config field from the CLI:

```bash
uv run python -m dlts.train --cfg cfg/patch_tst.yaml --model.d_model 64 --stage1.lr 5e-4
```

Checkpoints and sidecar JSON files are written to `checkpoints/`. Training progress is logged to Weights & Biases (set `--wandb.mode online` to enable).

## Evaluation

```bash
uv run python -m dlts.eval --checkpoint_dir checkpoints/ --model patch_tst
```

Prints accuracy, macro-F1, balanced accuracy, and log-loss on the held-out test set, and saves a confusion matrix to `checkpoints/`.

## Ensemble

Soft-voting ensemble weighted by each model's validation macro-F1:

```bash
uv run python -m dlts.ensemble \
    --checkpoint_dir checkpoints/ \
    --min_val_f1 0.3 \
    --temperature 10.0
```

Models below `--min_val_f1` are excluded. Saves per-model F1 comparison
chart and ensemble confusion matrix to `checkpoints/`.

## Training protocol

**From-scratch models** (InceptionTime, PatchTST, UniTS):
- Single training stage, 200 epochs, early stopping on val macro-F1 (patience 25).
- AdamW, cosine annealing with 1-epoch linear warm-up, gradient clipping at 1.0.

**Foundation model adapters** (Chronos, MOMENT):
1. **Stage 1 — linear probing**: backbone frozen, head only. 100 epochs, patience 15.
2. **Stage 2 — partial fine-tuning**: last `model.unfreeze_last_n` encoder blocks
   unfrozen with a lower LR. 50 epochs, patience 15.

**Loss**: `CrossEntropyLoss` with inverse-frequency class weights and label smoothing (eps = 0.1).

## Data

LSST dataset (UCR/UEA archive via `tslearn`): 2,459 train / 2,466 test, shape `(T=36, C=6)`, 14 classes.

Preprocessing:
1. NaN imputation: forward-fill -> backward-fill -> zero-fill per channel.
2. Global per-channel z-normalisation.
3. Stratified 80/20 train/val split.

Training-time augmentation: jitter (sigma=0.05), amplitude scaling (xU[0.9, 1.1]), channel dropout (p=0.15).

## Metrics

- Accuracy
- Macro F1
- Balanced Accuracy
- Log-Loss

## Project structure

```
src/dlts/
  data/lsst_ts.py     — data loading, preprocessing, Dataset class
  models/             — InceptionTime, PatchTST, UniTS, Chronos adapter, MOMENT adapter
  train.py            — training loop (single + 2-stage)
  eval.py             — per-checkpoint test evaluation
  ensemble.py         — F1-weighted soft-voting ensemble
  losses.py           — inverse-frequency class weights
  metrics.py          — accuracy, macro-F1, balanced accuracy, log-loss
cfg/                  — per-model YAML configs
colab.ipynb           — end-to-end training notebook (Colab / T4 GPU)
```
