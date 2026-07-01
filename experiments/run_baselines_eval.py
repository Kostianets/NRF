import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import load_datasets, DATASET_DISPLAY
from src.train import run_baselines_eval


def main():
    parser = argparse.ArgumentParser(
        description='Per-fold baseline evaluation (LR/SVM/XGB/RF/NN). '
                    'Variant-agnostic — runs once per dataset.')
    parser.add_argument('--dataset', required=True, choices=list(DATASET_DISPLAY.keys()),
                        help='Dataset name (lowercase)')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--results_dir', default='results',
                        help='Directory to save baselines results JSON')
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    print('Loading datasets...')
    datasets = load_datasets()
    name = DATASET_DISPLAY[args.dataset]

    if name not in datasets:
        print(f'ERROR: {name} could not be loaded.')
        sys.exit(1)

    X, y = datasets[name]
    df = run_baselines_eval(name, X, y, n_folds=args.n_folds)

    out_path = os.path.join(args.results_dir, f'baselines_{args.dataset}.json')
    df.to_json(out_path, orient='records', indent=2)
    print(f'\nBaselines results saved to: {out_path}')


if __name__ == '__main__':
    main()
