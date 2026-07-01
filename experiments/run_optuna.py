import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import load_datasets, DATASET_DISPLAY
from src.train import run_optuna

def main():
    parser = argparse.ArgumentParser(description='Optuna hyperparameter search for NRF')
    parser.add_argument('--dataset', required=True, choices=list(DATASET_DISPLAY.keys()),
                        help='Dataset name (lowercase)')
    parser.add_argument('--n_trials', type=int, default=60,
                        help='Number of Optuna trials')
    parser.add_argument('--force', action='store_true',
                        help='Re-run even if config file already exists')
    parser.add_argument('--variant',        default='full',
                        help='Variant slug — determines output directory')
    parser.add_argument('--feature_subset', action='store_true',
                        help='Use sqrt(m) random features per tree')
    parser.add_argument('--no_transformer', action='store_true',
                        help='Replace transformer with single linear layer')
    parser.add_argument('--linear_routers', action='store_true',
                        help='Use linear routers instead of MLP routers')
    parser.add_argument('--rf_routing', action='store_true',
                        help='Per-node raw-feature routing (true Neural RF)')
    parser.add_argument('--n_gpus', type=int, default=None,
                        help='Number of GPUs for parallel tree training. '
                             'Default: NRF_N_GPUS env var or torch.cuda.device_count().')
    args = parser.parse_args()

    fname = f'best_cfg_{args.variant}/{args.dataset}.json'

    if not args.force and os.path.exists(fname):
        print(f'Config already exists at {fname}')
        print('Use --force to re-run Optuna search.')
        sys.exit(0)

    print(f'Loading datasets...')
    datasets = load_datasets()
    name = DATASET_DISPLAY[args.dataset]

    if name not in datasets:
        print(f'ERROR: {name} could not be loaded.')
        sys.exit(1)

    X, y = datasets[name]
    run_optuna(
        args.dataset, X, y, n_trials=args.n_trials,
        feature_subset=args.feature_subset,
        use_transformer=not args.no_transformer,
        linear_routers=args.linear_routers,
        rf_routing=args.rf_routing,
        variant=args.variant,
        n_gpus=args.n_gpus,
    )

    print(f'\nOptuna search complete.')
    print(f'Best config saved to: {fname}')


# Guard against re-execution when torch.multiprocessing 'spawn' re-imports
# this script in worker processes. Without it, the child re-runs the body,
# tries to spawn more workers, and crashes with "An attempt has been made to
# start a new process before the current process has finished its
# bootstrapping phase."
if __name__ == '__main__':
    main()
