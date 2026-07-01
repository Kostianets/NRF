import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import load_datasets, DATASET_DISPLAY
from src.train import tune_baselines

parser = argparse.ArgumentParser(description='Tune baselines with GridSearchCV')
parser.add_argument('--dataset', required=True, choices=list(DATASET_DISPLAY.keys()),
                    help='Dataset name (lowercase)')
parser.add_argument('--cv', type=int, default=3, help='CV folds for GridSearchCV')
parser.add_argument('--force', action='store_true',
                    help='Re-tune even if cfg files already exist')
args = parser.parse_args()

methods = ['LR', 'SVM', 'RF', 'NN', 'XGB']
cfg_files = [f'cfg/{args.dataset}_{m.lower()}.json' for m in methods]

if not args.force and all(os.path.exists(f) for f in cfg_files):
    print(f'All baseline configs for {args.dataset} already exist.')
    print('Use --force to re-tune.')
    sys.exit(0)

print(f'Loading datasets...')
datasets = load_datasets()
name = DATASET_DISPLAY[args.dataset]

if name not in datasets:
    print(f'ERROR: {name} could not be loaded.')
    sys.exit(1)

X, y = datasets[name]
tune_baselines(args.dataset, X, y, cv=args.cv)
print(f'\nBaseline tuning complete for {args.dataset}.')
