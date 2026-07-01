import json
import os

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from src.model import SEED


def _gridsearch_jobs():
    slurm_cpus = os.getenv('SLURM_CPUS_PER_TASK')
    if slurm_cpus and slurm_cpus.isdigit():
        return max(1, int(slurm_cpus))
    return -1


def search_scoring(dataset_name, y):
    '''HP-search metric: roc_auc for imbalanced binary (adult),
    f1_macro for multi-class, accuracy for other binaries.'''
    if dataset_name.lower() == 'adult':
        return 'roc_auc'
    return 'f1_macro' if len(set(y)) > 2 else 'accuracy'


def tune_baselines(dataset_name, X, y, cv=3):
    '''
    Tune LR, SVM, RF, MLP with GridSearchCV.
    Saves best params to cfg/<dataset_name>_<method>.json.
    '''
    os.makedirs('cfg', exist_ok=True)
    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)
    if X_s.shape[0] > 10000:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, train_size=10000, random_state=SEED)
        sub_idx, _ = next(sss.split(X_s, y))
    else:
        sub_idx = None
    is_binary = len(set(y)) == 2
    lr_solvers = ['lbfgs', 'liblinear'] if is_binary else ['lbfgs']

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
            RandomForestClassifier(random_state=SEED),
            {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20],
             'min_samples_split': [2, 5], 'max_features': ['sqrt', 'log2']},
        ),
        'NN': (
            MLPClassifier(max_iter=1000, random_state=SEED),
            {'hidden_layer_sizes': [(64,), (128,), (100, 50), (128, 64)],
             'alpha': [0.0001, 0.001, 0.01], 'learning_rate_init': [0.001, 0.01]},
        ),
        'XGB': (
            XGBClassifier(
                eval_metric='logloss',
                random_state=SEED,
                tree_method='hist',
                device='cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu',
            ),
            {'n_estimators': [200, 500],
             'max_depth': [3, 6, 10],
             'learning_rate': [0.05, 0.1],
             'subsample': [0.7, 1.0],
             'colsample_bytree': [0.7, 1.0]},
        ),
    }

    best_cfgs = {}
    scoring = search_scoring(dataset_name, y)
    print(f'\nTuning baselines for {dataset_name}  ({cv}-fold CV)...')
    print(f'Baseline search metric: {scoring}')
    for name, (clf, param_grid) in grids.items():
        gs = GridSearchCV(
            clf,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=_gridsearch_jobs(),
            refit=False,
        )
        if sub_idx is not None:
            gs.fit(X_s[sub_idx], y[sub_idx])
        else:
            gs.fit(X_s, y)
        best_cfgs[name] = gs.best_params_
        params_json = {k: list(v) if isinstance(v, tuple) else v
                       for k, v in gs.best_params_.items()}
        fname = f'cfg/{dataset_name.lower()}_{name.lower()}.json'
        with open(fname, 'w') as f:
            json.dump(params_json, f, indent=2)
        print(f'  {name:>4}  cv_score={gs.best_score_*100:.2f}%  {gs.best_params_}')
    return best_cfgs


def load_baseline_cfgs(dataset_name):
    cfgs = {}
    for name in ['LR', 'SVM', 'XGB', 'RF', 'NN']:
        fname = f'cfg/{dataset_name.lower()}_{name.lower()}.json'
        if os.path.exists(fname):
            with open(fname) as f:
                params = json.load(f)
            if 'hidden_layer_sizes' in params and isinstance(params['hidden_layer_sizes'], list):
                params['hidden_layer_sizes'] = tuple(params['hidden_layer_sizes'])
            cfgs[name] = params
    return cfgs if cfgs else None
