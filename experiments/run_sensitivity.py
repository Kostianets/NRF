import argparse
import os
import sys
import json
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from src.datasets import load_datasets, DATASET_DISPLAY
from src.model import NRF, SEED


def main():
    parser = argparse.ArgumentParser(description='Sensitivity sweep: n_trees x depth grid')
    parser.add_argument('--dataset', default='magic', choices=list(DATASET_DISPLAY.keys()),
                        help='Dataset to sweep on (lowercase)')
    parser.add_argument('--n_trees_grid', nargs='+', type=int, default=[5, 10, 15, 20, 25, 30])
    parser.add_argument('--depth_grid', nargs='+', type=int, default=[2, 3, 4, 5, 6])
    parser.add_argument('--n_folds', type=int, default=3)
    parser.add_argument('--cfg_dataset', default=None,
                        help='Which Optuna config to use as base (defaults to --dataset)')
    parser.add_argument('--cfg_variant', default='full',
                        help='Optuna variant directory slug (best_cfg_{variant})')
    parser.add_argument('--results_dir', default='results')
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    cfg_dataset = args.cfg_dataset or args.dataset

    # Load base config
    cfg_path = f'best_cfg_{args.cfg_variant}/{cfg_dataset}.json'
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            base_cfg = json.load(f)
        # remove fields that will be swept
        base_cfg.pop('depth', None)
        base_cfg.pop('n_trees', None)
        print(f'Using Optuna config from {cfg_path} as base')
    else:
        base_cfg = dict(h_dim=32, router_hidden=16, solver_hidden=16, lr=0.003,
                        weight_decay=1e-4, dropout=0.1, router_dropout=0.05,
                        temp_start=1.0, temp_end=0.3, entropy_reg=0.01,
                        max_noise_std=0.05, batch_size=32, epochs=150,
                        patience=20, min_epochs=40, feature_subset=False)
        print(f'No Optuna config found at {cfg_path} — using hardcoded defaults')

    print(f'Loading datasets...')
    all_datasets = load_datasets()
    name = DATASET_DISPLAY[args.dataset]

    if name not in all_datasets:
        print(f'ERROR: {name} could not be loaded.')
        sys.exit(1)

    X, y = all_datasets[name]
    print(f'Dataset: {name}  {X.shape}')

    n_depths  = len(args.depth_grid)
    n_trees_n = len(args.n_trees_grid)
    results_sens = np.zeros((n_depths, n_trees_n))
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=SEED)

    total = n_depths * n_trees_n
    done  = 0
    t0    = time.time()

    print(f'\nSweep: {total} configs × {args.n_folds}-fold CV on {name}')
    print(f'{"depth":>7}  {"n_trees":>8}  {"acc":>8}  {"elapsed":>10}')
    print('-' * 40)

    for i, depth in enumerate(args.depth_grid):
        for j, n_trees in enumerate(args.n_trees_grid):
            cfg = {**base_cfg, 'depth': depth, 'n_trees': n_trees}
            accs = []
            for tr, te in skf.split(X, y):
                scaler = StandardScaler().fit(X[tr])
                Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
                clf = NRF(**cfg)
                clf.fit(Xtr, y[tr], verbose=False)
                accs.append(accuracy_score(y[te], clf.predict(Xte)))
            results_sens[i, j] = np.mean(accs) * 100
            done += 1
            elapsed = time.time() - t0
            print(f'{depth:>7}  {n_trees:>8}  {np.mean(accs)*100:>7.2f}%  {elapsed:>9.1f}s'
                  f'  [{done}/{total}]')

    output = {
        'dataset':      args.dataset,
        'n_trees_grid': args.n_trees_grid,
        'depth_grid':   args.depth_grid,
        'results':      results_sens.tolist(),
    }
    out_path = os.path.join(args.results_dir, f'sensitivity_{args.dataset}.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f'\nSensitivity results saved to: {out_path}')
    print(f'Total time: {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
