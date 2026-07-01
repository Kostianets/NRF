import argparse
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import load_datasets, DATASET_DISPLAY
from src.train import run_ablation

parser = argparse.ArgumentParser(description='Run ablation study (transformer/router/RF variants)')
parser.add_argument('--datasets', nargs='+', default=['magic', 'eeg'],
                    help='Datasets to ablate (lowercase names)')
parser.add_argument('--n_folds', type=int, default=3, help='Number of CV folds')
parser.add_argument('--cfg_dataset', default='magic',
                    help='Which dataset Optuna config to use as base')
parser.add_argument('--cfg_variant', default='full',
                    help='Optuna variant directory slug (best_cfg_{variant})')
parser.add_argument('--results_dir', default='results', help='Directory to save results JSON')
args = parser.parse_args()

os.makedirs(args.results_dir, exist_ok=True)

# Load base config
cfg_path = f'best_cfg_{args.cfg_variant}/{args.cfg_dataset}.json'
nrf_cfg = None
if os.path.exists(cfg_path):
    with open(cfg_path) as f:
        nrf_cfg = json.load(f)
    print(f'Using Optuna config from {cfg_path} as ablation base')
else:
    print(f'No Optuna config at {cfg_path} — using hardcoded defaults')

print(f'Loading datasets...')
all_datasets = load_datasets()

ablation_datasets = {}
for ds in args.datasets:
    if ds not in DATASET_DISPLAY:
        print(f'WARNING: unknown dataset {ds}, skipping')
        continue
    name = DATASET_DISPLAY[ds]
    if name not in all_datasets:
        print(f'WARNING: {name} could not be loaded, skipping')
        continue
    ablation_datasets[name] = all_datasets[name]

if not ablation_datasets:
    print('ERROR: no datasets loaded for ablation.')
    sys.exit(1)

df = run_ablation(ablation_datasets, n_folds=args.n_folds, nrf_cfg=nrf_cfg)

out_path = os.path.join(args.results_dir, 'ablation.json')
df.to_json(out_path, orient='records', indent=2)
print(f'\nAblation results saved to: {out_path}')
