import argparse
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import load_datasets, DATASET_DISPLAY
from src.model import NRF
from src.train import run_experiment_single

def main():
    parser = argparse.ArgumentParser(description='Run 5-fold CV experiment: NRF vs baselines')
    parser.add_argument('--dataset', required=True, choices=list(DATASET_DISPLAY.keys()),
                        help='Dataset name (lowercase)')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--results_dir', default='results', help='Directory to save results JSON')
    parser.add_argument('--models_dir', default='models', help='Directory to save trained model checkpoints')
    parser.add_argument('--variant',        default='full',
                        help='Variant slug — determines config directory and results filename')
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
    parser.add_argument('--no_baselines', action='store_true',
                        help='Skip baseline training (NRF only).')
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)

    variant_flags = dict(
        feature_subset=args.feature_subset,
        use_transformer=not args.no_transformer,
        linear_routers=args.linear_routers,
        rf_routing=args.rf_routing,
    )

    # Load Optuna best config (required)
    cfg_path = f'best_cfg_{args.variant}/{args.dataset}.json'
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            nrf_cfg = json.load(f)
        print(f'Loaded Optuna config from {cfg_path}')
    else:
        print(f'ERROR: missing Optuna config at {cfg_path}.')
        print('Run Optuna first (or use jobs/submit_all.sh for dependency-safe pipeline).')
        sys.exit(1)

    print(f'Loading datasets...')
    datasets = load_datasets()
    name = DATASET_DISPLAY[args.dataset]

    if name not in datasets:
        print(f'ERROR: {name} could not be loaded.')
        sys.exit(1)

    X, y = datasets[name]
    df = run_experiment_single(name, X, y, nrf_cfg=nrf_cfg, n_folds=args.n_folds,
                               variant_flags=variant_flags, n_gpus=args.n_gpus,
                               skip_baselines=args.no_baselines)

    # Save with orient='records' — preserves list-typed columns (Acc_folds etc.)
    out_path = os.path.join(args.results_dir, f'{args.variant}_{args.dataset}_experiment.json')
    df.to_json(out_path, orient='records', indent=2)
    print(f'\nResults saved to: {out_path}')

    # Train and save final NRF model on full dataset for later reuse.
    final_cfg = nrf_cfg.copy()
    final_cfg.update(variant_flags)
    final_model = NRF(**final_cfg)
    final_model.fit(X, y, verbose=False, n_gpus=args.n_gpus)
    model_path = os.path.join(args.models_dir, f'{args.variant}_{args.dataset}.pt')
    final_model.save(model_path)
    print(f'Model saved to: {model_path}')


# Guard against re-execution when torch.multiprocessing 'spawn' re-imports
# this script in worker processes.
if __name__ == '__main__':
    main()
