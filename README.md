# Neural Random Forest (NRF)

A Neural Random Forest for tabular classification: a bagged ensemble of
differentiable **soft decision trees**. Each tree routes inputs softly with
sigmoid gates, predicts with a small per-leaf network, and the ensemble combines
trees by out-of-bag–weighted voting. It targets supervised binary and multiclass
problems on any fixed-length numeric feature vectors.

This repository accompanies a bachelor thesis. The implementation draws on several lines of work — adaptive neural trees (ANT), deep neural decision forests (Deep NDF), deep neural decision trees (DNDT) and the original Neural Random Forest (NNRF); the per-component lineage is annotated in `src/model.py`.

## Repository structure

```
src/
  model.py        NRF ensemble: soft neural trees, fit/predict/save/load
  datasets.py     dataset loaders + auto-download (Kaggle/UCI/torchvision/sklearn)
  train.py        Optuna search, 5-fold CV runner, baseline eval, ablation
  baselines.py    GridSearchCV tuning for LR/SVM/RF/MLP/XGB
experiments/
  run_optuna.py            hyperparameter search -> best_cfg_<variant>/<dataset>.json
  run_experiment.py        5-fold CV: NRF vs baselines
  run_baselines_eval.py    per-fold baseline-only evaluation
  tune_baselines.py        GridSearchCV baseline tuning -> cfg/
  retune_baselines_full.py baseline tuning on full (not subsampled) data
  run_ablation.py          architecture ablation study
  run_sensitivity.py       n_trees x depth sweep
requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended but not required (training falls back to CPU).
For GPU runs, install a CUDA build of PyTorch that matches your driver — see
[pytorch.org](https://pytorch.org/get-started/locally/) — then install the rest
with the command above. Verify the GPU is visible:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Usage

The model has a scikit-learn–style API (`fit` / `predict` / `predict_proba`):

```python
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.model import NRF

X, y = load_wine(return_X_y=True)
X = StandardScaler().fit_transform(X).astype(np.float32)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

clf = NRF(n_trees=25, depth=3)
clf.fit(Xtr, ytr)
print('accuracy:', (clf.predict(Xte) == yte).mean())

clf.save('nrf_wine.pt')
reloaded = NRF.load('nrf_wine.pt')
```

`X` is a `float32` array of shape `(n_samples, n_features)` and `y` is an
integer label array of shape `(n_samples,)`; standardizing the features (as
above) is recommended. Useful constructor arguments include `n_trees`, `depth`,
`h_dim`, and the soft-routing temperature schedule (`temp_start` / `temp_end`) —
see the `NRF` constructor in `src/model.py` for the full list.

## Running experiments

Run all scripts from the project root. `--dataset` takes any slug from
`DATASET_DISPLAY` in `src/datasets.py`: `magic`, `eeg`, `adult`, `letter`,
`shuttle`, `higgs`, `covertype`, `otto`, `cifar10_resnet18`, `mnist_resnet18`,
`wine`, `wdbc`, `isolet`, `pokerhand`.

The first run auto-downloads the requested dataset. `wine` and `wdbc` are
sklearn built-ins (no download); `shuttle` comes from the UCI ML repository via
`ucimlrepo`; `cifar10_resnet18` / `mnist_resnet18` are downloaded by torchvision
and cached as ResNet-18 features. The remaining sets are pulled from Kaggle via
`kagglehub`, which needs Kaggle API credentials (`~/.kaggle/kaggle.json`, or the
`KAGGLE_USERNAME` / `KAGGLE_KEY` environment variables).

Training pipeline:

```bash
# 1. Hyperparameter search (required — writes best_cfg_full/<dataset>.json)
python experiments/run_optuna.py --dataset wine --n_trials 60

# 2. (Optional) Tune baselines for a fair comparison. Skip this and the
#    baselines fall back to sklearn defaults; or skip baselines entirely
#    with --no_baselines in step 3.
python experiments/tune_baselines.py --dataset wine

# 3. 5-fold CV — NRF vs baselines (needs the config from step 1)
python experiments/run_experiment.py --dataset wine
#    ...or NRF only, no baselines:
python experiments/run_experiment.py --dataset wine --no_baselines
```

Step 1 is required — `run_experiment.py` exits if `best_cfg_<variant>/<dataset>.json`
is missing. Step 2 is optional: without tuned configs the baselines use sklearn
defaults, and `--no_baselines` drops them from the run altogether.

Additional studies (all optional): `run_baselines_eval.py` scores baselines
on their own over the same folds, `retune_baselines_full.py` re-tunes baselines
on the full (non-subsampled) data, `run_ablation.py` toggles architectural
components, and `run_sensitivity.py` sweeps the `n_trees` × `depth` grid.

## Reproducibility

- The global seed is `SEED = 42` in `src/model.py`.
- Per-tree randomness (bootstrap, feature subsets, input noise) is seeded
deterministically so single-GPU runs reproduce against the seed.

## Acknowledgments

Developed as a bachelor thesis at the **Technical University of Košice**, under
the supervision of **Ing. Martina Szabóová, PhD**.

## Citation

If you use this work, please cite the accompanying thesis. A ready-to-use BibTeX
entry is in `[CITATION.bib](CITATION.bib)`, and `[CITATION.cff](CITATION.cff)`
powers GitHub's "Cite this repository" button.

```bibtex
@mastersthesis{kostianets2026NRF,
  author  = {Oleksandr Kostianets},
  title   = {Neural Random Forest for Classification Tasks},
  school  = {Technical University of Košice},
  address = {Košice, Slovakia},
  year    = {2026},
  type    = {Bachelor's thesis},
  note    = {Supervisor: Ing. Martina Szabóová, PhD}
}
```

## License

Released under the [MIT License](LICENSE).