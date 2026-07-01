"""Re-tune the five baselines (LR/SVM/RF/MLP/XGB) on the FULL dataset.

Unlike `src.baselines.tune_baselines` — which grid-searches on a 10k stratified
subsample for datasets >10k rows — this tunes on the full data for LR/RF/XGB/MLP,
and caps only RBF-SVM at `--svm_cap` rows (matching the eval-time LinearSVC>50k
swap in `src.train`). The grids and scoring policy are identical to
`src.baselines.tune_baselines`.

Writes `cfg_full/<dataset>_<method>.json` (same format as `cfg/`). Point the
experiment pipeline at them by copying `cfg_full/*` into `cfg/` once you're ready.

Run from the project root:

    python experiments/retune_baselines_full.py                       # 10 large datasets
    python experiments/retune_baselines_full.py --dataset higgs       # one dataset
    python experiments/retune_baselines_full.py --dataset higgs --dataset pokerhand
    python experiments/retune_baselines_full.py --out_dir cfg         # overwrite originals
    python experiments/retune_baselines_full.py --max_rows 200000     # cap heavy methods too
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src import datasets as D
from src.datasets import DATASET_DISPLAY
from src.baselines import search_scoring          # identical scoring policy
from src.model import SEED                         # = 42

# The 10 large (>10k-row) datasets that were originally tuned on a 10k subsample
# (or, for magic/otto, never tuned at all).
DEFAULT_DATASETS = ['magic', 'eeg', 'letter', 'adult', 'shuttle', 'otto',
                    'covertype', 'higgs', 'pokerhand', 'isolet']


def _xgb_device():
    """CUDA for XGB if a GPU is visible, else CPU — mirrors src.baselines."""
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        return 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'


def _load_one(slug):
    """Single-dataset loader mirroring load_datasets() dispatch, so we never
    trigger the heavy CIFAR/MNIST ResNet feature extraction for tabular runs."""
    if slug == 'shuttle':
        return D._load_shuttle()
    if slug == 'isolet':
        return D._load_isolet()
    if slug == 'pokerhand':
        return D._load_pokerhand()
    if slug in ('wine', 'wdbc'):
        return getattr(D, f'_load_{slug}')()
    if slug == 'cifar10_resnet18':
        return D._load_cifar10_resnet18_features()
    if slug == 'mnist_resnet18':
        return D._load_mnist_resnet18_features()
    return D._load_from_kaggle(D.KAGGLE_DATASET_SOURCES[slug])


def _subsample(X, y, n):
    if n is None or X.shape[0] <= n:
        return X, y
    idx, _ = next(StratifiedShuffleSplit(n_splits=1, train_size=n,
                                         random_state=SEED).split(X, y))
    return X[idx], y[idx]


def tune_baselines_full(dataset_name, X, y, out_dir='cfg_full', cv=3,
                        svm_cap=50_000, max_rows=None):
    """GridSearchCV for LR/SVM/RF/MLP/XGB on full data (RBF-SVM capped at svm_cap).
    Grids are a verbatim copy of src.baselines.tune_baselines — keep in sync."""
    os.makedirs(out_dir, exist_ok=True)
    X_s = StandardScaler().fit(X).transform(X)
    is_binary = len(set(y)) == 2
    lr_solvers = ['lbfgs', 'liblinear'] if is_binary else ['lbfgs']
    use_gpu = _xgb_device() == 'cuda'

    grids = {
        'LR': (
            LogisticRegression(max_iter=2000, random_state=SEED),
            {'C': [0.01, 0.1, 1, 10, 100], 'solver': lr_solvers},
        ),
        'SVM': (
            SVC(kernel='rbf', probability=True, random_state=SEED),
            {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.01, 0.1]},
        ),
        'RF': (
            RandomForestClassifier(random_state=SEED, n_jobs=-1),
            {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20],
             'min_samples_split': [2, 5], 'max_features': ['sqrt', 'log2']},
        ),
        'NN': (
            MLPClassifier(max_iter=1000, random_state=SEED),
            {'hidden_layer_sizes': [(64,), (128,), (100, 50), (128, 64)],
             'alpha': [0.0001, 0.001, 0.01], 'learning_rate_init': [0.001, 0.01]},
        ),
        'XGB': (
            XGBClassifier(eval_metric='logloss', random_state=SEED,
                          tree_method='hist', device=_xgb_device()),
            {'n_estimators': [200, 500], 'max_depth': [3, 6, 10],
             'learning_rate': [0.05, 0.1], 'subsample': [0.7, 1.0],
             'colsample_bytree': [0.7, 1.0]},
        ),
    }
    scoring = search_scoring(dataset_name, y)
    print(f'\n{"="*64}')
    print(f'{dataset_name}  ({X.shape[0]}x{X.shape[1]}, {len(np.unique(y))} classes)'
          f'  scoring={scoring}')
    print(f'{"="*64}')

    best = {}
    for name, (clf, grid) in grids.items():
        Xt, yt = _subsample(X_s, y, svm_cap if name == 'SVM' else max_rows)
        # On a single GPU, keep XGB's GridSearch serial to avoid GPU contention.
        n_jobs = 1 if (name == 'XGB' and use_gpu) else -1
        t0 = time.time()
        gs = GridSearchCV(clf, grid, cv=cv, scoring=scoring, n_jobs=n_jobs, refit=False)
        gs.fit(Xt, yt)
        dt = time.time() - t0
        best[name] = gs.best_params_
        params_json = {k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in gs.best_params_.items()}
        fname = os.path.join(out_dir, f'{dataset_name.lower()}_{name.lower()}.json')
        with open(fname, 'w') as f:
            json.dump(params_json, f, indent=2)
        print(f'  {name:>4}  n={len(Xt):>9}  {scoring}={gs.best_score_*100:6.2f}%'
              f'  {dt/60:6.1f} min  -> {fname}  {gs.best_params_}')
    return best


def main():
    p = argparse.ArgumentParser(description='Re-tune baselines on full datasets.')
    p.add_argument('--dataset', action='append', dest='datasets',
                   choices=list(DATASET_DISPLAY.keys()),
                   help='Dataset slug; repeatable. Default: the 10 large datasets.')
    p.add_argument('--out_dir', default='cfg_full',
                   help="Output dir for configs (default cfg_full; use 'cfg' to overwrite).")
    p.add_argument('--cv', type=int, default=3, help='GridSearchCV folds (default 3).')
    p.add_argument('--svm_cap', type=int, default=50_000,
                   help='Max rows for RBF-SVM tuning (default 50000).')
    p.add_argument('--max_rows', type=int, default=None,
                   help='Cap rows for LR/RF/XGB/MLP tuning too (default: full data).')
    args = p.parse_args()

    slugs = args.datasets or DEFAULT_DATASETS
    print(f'Re-tuning on full data -> {args.out_dir}/  '
          f'(cv={args.cv}, svm_cap={args.svm_cap}, max_rows={args.max_rows})')
    print(f'Datasets: {slugs}')

    summary = []
    for slug in slugs:
        disp = DATASET_DISPLAY[slug]
        try:
            X, y = _load_one(slug)
        except Exception as e:
            print(f'SKIP {slug}: {e}')
            continue
        t0 = time.time()
        tune_baselines_full(disp, X, y, out_dir=args.out_dir, cv=args.cv,
                            svm_cap=args.svm_cap, max_rows=args.max_rows)
        summary.append((slug, time.time() - t0))

    print(f'\n{"="*64}\nWall-clock per dataset:')
    for s, t in summary:
        print(f'  {s:<12} {t/60:7.1f} min')
    print(f'Configs written to: {args.out_dir}/')


if __name__ == '__main__':
    main()
