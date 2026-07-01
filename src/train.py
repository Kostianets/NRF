import json
import os
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from src.model import NRF, SEED
from src.baselines import tune_baselines, load_baseline_cfgs


def _fold_auc_ovr(y_true, proba, present_classes, all_classes):
    '''
    Fold-safe multiclass AUC.
    Uses one-vs-rest only for classes represented in this test fold.
    '''
    aligned = np.zeros((len(y_true), len(all_classes)), dtype=float)
    class_to_col = {cls: idx for idx, cls in enumerate(present_classes)}
    for j, cls in enumerate(all_classes):
        col = class_to_col.get(cls)
        if col is not None:
            aligned[:, j] = proba[:, col]

    aucs = []
    for j, cls in enumerate(all_classes):
        y_bin = (y_true == cls).astype(int)
        # AUC is undefined when the test fold has only one label for this class.
        if y_bin.min() == y_bin.max():
            continue
        aucs.append(roc_auc_score(y_bin, aligned[:, j]))

    return float(np.mean(aucs)) if aucs else np.nan


# ── Optuna search ─────────────────────────────────────────────────────────────

def run_optuna(dataset_name, X_raw, y, n_trials=40,
               feature_subset=False, use_transformer=True,
               linear_routers=False, rf_routing=False, variant='full',
               n_gpus=None):
    '''
    Bayesian HP search. Saves to best_cfg_{variant}/<name>.json.
    Best hyperparameters are printed to stdout and saved to JSON.
    Delete the file to re-run search.
    '''
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print('Optuna not installed.'); return None

    os.makedirs(f'best_cfg_{variant}', exist_ok=True)
    print(f'Optuna | {dataset_name}  {X_raw.shape}  n_trials={n_trials}  variant={variant}')

    OPTUNA_OBJECTIVE = {
        'adult':      'roc_auc',
    }
    if dataset_name.lower() in OPTUNA_OBJECTIVE:
        metric_name = OPTUNA_OBJECTIVE[dataset_name.lower()]
    else:
        # macro-F1 for any multi-class task; accuracy is dominated by the majority class
        metric_name = 'f1_macro' if len(np.unique(y)) > 2 else 'accuracy'
    print(f'Optuna objective metric: {metric_name}')

    if dataset_name.lower() in ('covertype', 'higgs', 'pokerhand'):
        batch_size_choices = [16384]
    elif dataset_name.lower() in ('wine', 'wdbc'):
        batch_size_choices = [32, 64, 128]
    else:
        batch_size_choices = [512]

    def objective(trial):
        cfg = dict(
            n_trees        = 15,
            depth          = trial.suggest_int('depth', 2, 6),
            h_dim          = trial.suggest_int('h_dim', 16, 512, step=16),
            router_hidden  = trial.suggest_int('router_hidden', 8, 32, step=8),
            solver_hidden  = trial.suggest_int('solver_hidden', 8, 32, step=8),
            lr             = trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            weight_decay   = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
            dropout        = trial.suggest_float('dropout', 0.0, 0.3),
            router_dropout = trial.suggest_float('router_dropout', 0.0, 0.2),
            temp_end       = trial.suggest_float('temp_end', 0.1, 0.8),
            entropy_reg    = trial.suggest_float('entropy_reg', 0.0, 0.05),
            max_noise_std  = trial.suggest_float('max_noise_std', 0.0, 0.15),
            epochs=80, batch_size=trial.suggest_categorical('batch_size', batch_size_choices), patience=15, min_epochs=40,
            temp_start=1.0,
            feature_subset=feature_subset,
            use_transformer=use_transformer,
            linear_routers=linear_routers,
            rf_routing=rf_routing,
        )
        skf    = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
        scores = []
        for step, (tr, te) in enumerate(skf.split(X_raw, y)):
            scaler   = StandardScaler().fit(X_raw[tr])
            Xtr, Xte = scaler.transform(X_raw[tr]), scaler.transform(X_raw[te])
            clf = NRF(**cfg)
            clf.fit(Xtr, y[tr], verbose=False, n_gpus=n_gpus)
            if metric_name == 'roc_auc':
                proba = clf.predict_proba(Xte)
                # binary: use the positive-class column;
                # multiclass: OvR macro-averaged
                if proba.shape[1] == 2:
                    score = roc_auc_score(y[te], proba[:, 1])
                else:
                    score = roc_auc_score(y[te], proba, multi_class='ovr',
                                          average='macro')
            elif metric_name == 'f1_macro':
                score = f1_score(y[te], clf.predict(Xte), average='macro')
            else:
                score = accuracy_score(y[te], clf.predict(Xte))
            scores.append(score)
            trial.report(float(np.mean(scores)), step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # ── Print best hyperparameters clearly to stdout (visible in SLURM logs) ──
    print(f'\n{"="*60}')
    print(f'OPTUNA BEST RESULTS — {dataset_name}')
    print(f'{"="*60}')
    print(f'Best CV {metric_name}: {study.best_value*100:.2f}%')
    print(f'Best trial: #{study.best_trial.number}')
    print(f'\nBest hyperparameters:')
    for k, v in study.best_params.items():
        print(f'  {k:>20}: {v}')
    print(f'{"="*60}\n')

    best_cfg = dict(
        **study.best_params,
        n_trees=30, epochs=150, patience=20,
        temp_start=1.0,
        feature_subset=feature_subset,
        use_transformer=use_transformer,
        linear_routers=linear_routers,
        rf_routing=rf_routing,
    )
    fname = f'best_cfg_{variant}/{dataset_name.lower()}.json'
    with open(fname, 'w') as f:
        json.dump(best_cfg, f, indent=2)
    print(f'Saved -> {fname}')

    # Save importance plot if available
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import optuna.visualization.matplotlib as ovm
        os.makedirs('plots', exist_ok=True)
        fig = ovm.plot_param_importances(study)
        plt.tight_layout()
        plt.savefig(f'plots/optuna_importance_{dataset_name.lower()}.png',
                    bbox_inches='tight', dpi=150)
        plt.close()
    except Exception:
        pass

    return best_cfg


# ── Baseline construction helper ──────────────────────────────────────────────

def _make_baseline_clf(tag, baseline_cfgs):
    '''Construct a baseline classifier. Uses tuned cfg if available, else sklearn defaults.'''
    if baseline_cfgs and tag in baseline_cfgs:
        p = baseline_cfgs[tag]
        if tag == 'LR':  return LogisticRegression(**p, max_iter=2000, random_state=SEED)
        if tag == 'SVM': return SVC(**p, kernel='rbf', probability=True, random_state=SEED)
        if tag == 'RF':  return RandomForestClassifier(**p, random_state=SEED, n_jobs=-1)
        if tag == 'NN':  return MLPClassifier(**p, max_iter=1000, random_state=SEED)
        if tag == 'XGB':
            return XGBClassifier(**p, eval_metric='logloss', random_state=SEED,
                                 tree_method='hist',
                                 device='cuda' if torch.cuda.is_available() else 'cpu')
    return {'LR':  LogisticRegression(max_iter=1000, random_state=SEED),
            'SVM': SVC(kernel='rbf', probability=True, random_state=SEED),
            'RF':  RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
            'NN':  MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=SEED),
            'XGB': XGBClassifier(n_estimators=200, max_depth=6,
                                eval_metric='logloss', random_state=SEED,
                                tree_method='hist',
                                device='cuda' if torch.cuda.is_available() else 'cpu')}[tag]


# ── Per-fold baseline evaluation (variant-agnostic; runnable as standalone CPU job) ──

def run_baselines_eval(dataset_name, X, y, n_folds=5):
    '''
    Per-fold evaluation of LR/SVM/XGB/RF/NN. Uses the same StratifiedKFold split
    (shared SEED) as run_experiment_single, so the resulting rows can be merged
    1:1 with NRF rows from a separate --no_baselines run on the same (dataset,
    n_folds).
    '''
    print(f'\n{"="*62}')
    print(f'Baselines eval: {dataset_name}  ({X.shape[0]}x{X.shape[1]})')
    print(f'{"="*62}')

    all_classes = np.unique(y)
    C      = len(all_classes)
    avg_kw = 'macro' if C > 2 else 'binary'
    skf    = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    baseline_cfgs = load_baseline_cfgs(dataset_name)
    if baseline_cfgs:
        print(f'  Baselines: GridSearchCV tuned ({", ".join(baseline_cfgs.keys())})')
    else:
        print('  Baselines: sklearn defaults  (run tune_baselines() first for fair comparison)')

    methods    = ['LR', 'SVM', 'XGB', 'RF', 'NN']
    acc_scores = {m: [] for m in methods}
    f1_scores  = {m: [] for m in methods}
    auc_scores = {m: [] for m in methods}

    # rbf SVM (LibSVM) is O(n²) — unusable above ~50k rows.
    # Above this threshold we swap to LinearSVC (liblinear, O(n)) wrapped in
    # CalibratedClassifierCV to preserve predict_proba.
    SVM_N_MAX = 50_000

    for fold, (tr, te) in enumerate(skf.split(X, y)):
        scaler   = StandardScaler().fit(X[tr])
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
        ytr, yte = y[tr], y[te]

        for tag in methods:
            clf_obj = _make_baseline_clf(tag, baseline_cfgs)
            if tag == 'SVM' and len(Xtr) > SVM_N_MAX:
                clf_obj = CalibratedClassifierCV(
                    LinearSVC(max_iter=2000, random_state=SEED), cv=3)
            clf_obj.fit(Xtr, ytr)
            p   = clf_obj.predict(Xte)
            prb = clf_obj.predict_proba(Xte)
            acc_scores[tag].append(accuracy_score(yte, p))
            f1_scores[tag].append(f1_score(yte, p, average=avg_kw))
            auc = (
                roc_auc_score(yte, prb[:, 1]) if C == 2
                else _fold_auc_ovr(yte, prb, clf_obj.classes_, all_classes)
            )
            auc_scores[tag].append(auc)

        print(f'  Fold {fold+1}/{n_folds}')

    rows = []
    print(f'\n  {"Method":<10}  {"Acc mean±std":>16}   {"F1 mean±std":>16}   {"AUC mean±std":>16}')
    print(f'  {"-"*68}')
    for m in methods:
        a = np.array(acc_scores[m]) * 100
        f = np.array(f1_scores[m]) * 100
        u = np.array(auc_scores[m]) * 100
        rows.append({'Dataset': dataset_name, 'Method': m,
                     'Acc': a.mean(), 'Acc_std': a.std(),
                     'F1':  f.mean(), 'F1_std':  f.std(),
                     'AUC': np.nanmean(u), 'AUC_std': np.nanstd(u),
                     'Acc_folds': list(a), 'F1_folds': list(f), 'AUC_folds': list(u)})
        print(
            f'  {m:<10}  {a.mean():>6.2f} ± {a.std():>4.2f}   '
            f'{f.mean():>6.2f} ± {f.std():>4.2f}   '
            f'{np.nanmean(u):>6.2f} ± {np.nanstd(u):>4.2f}'
        )
    return pd.DataFrame(rows)


# ── Single-dataset experiment ─────────────────────────────────────────────────

def run_experiment_single(dataset_name, X, y, nrf_cfg=None, n_folds=5,
                          variant_flags=None, n_gpus=None, skip_baselines=False):
    '''5-fold CV — NRF vs tuned baselines. Auto-loads cfg/ if present.'''
    print(f'\n{"="*62}')
    print(f'Dataset: {dataset_name}  ({X.shape[0]}x{X.shape[1]})')
    print(f'{"="*62}')

    all_classes = np.unique(y)
    C      = len(all_classes)
    avg_kw = 'macro' if C > 2 else 'binary'
    skf    = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    min_class_count = int(np.bincount(np.asarray(y, dtype=int)).min())
    if min_class_count < n_folds:
        print(
            f'  Warning: smallest class has {min_class_count} samples, '
            f'less than n_folds={n_folds}; fold-level AUC may ignore absent classes.'
        )

    if nrf_cfg is not None:
        cfg = nrf_cfg.copy()
        print('  NRF: Optuna best params')
    else:
        cfg = dict(n_trees=30, depth=3, h_dim=32, router_hidden=16, solver_hidden=16,
                   epochs=150, batch_size=512, lr=0.003, patience=20, temp_start=1.0,
                   temp_end=0.3, entropy_reg=0.01, feature_subset=False, dropout=0.1,
                   weight_decay=1e-4, router_dropout=0.05, max_noise_std=0.05)

    if variant_flags:
        cfg.update(variant_flags)

    if skip_baselines:
        print('  Baselines: skipped (--no_baselines)')
        baseline_cfgs = {}
    else:
        baseline_cfgs = load_baseline_cfgs(dataset_name)
        if baseline_cfgs:
            print(f'  Baselines: GridSearchCV tuned ({", ".join(baseline_cfgs.keys())})')
        else:
            print('  Baselines: sklearn defaults  (run tune_baselines() first for fair comparison)')

    methods    = ['NRF'] if skip_baselines else ['LR', 'SVM', 'XGB', 'RF', 'NN', 'NRF']
    acc_scores = {m: [] for m in methods}
    f1_scores  = {m: [] for m in methods}
    auc_scores = {m: [] for m in methods}

    # rbf SVM (LibSVM) is O(n²) — unusable above ~50k rows.
    # Above this threshold we swap to LinearSVC (liblinear, O(n)) wrapped in
    # CalibratedClassifierCV to preserve predict_proba.
    SVM_N_MAX = 50_000

    for fold, (tr, te) in enumerate(skf.split(X, y)):
        scaler   = StandardScaler().fit(X[tr])
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
        ytr, yte = y[tr], y[te]

        # ── NRF first so GPU work is never blocked by slow CPU baselines ──
        clf = NRF(**cfg)
        clf.fit(Xtr, ytr, verbose=False, n_gpus=n_gpus)
        p   = clf.predict(Xte)
        prb = clf.predict_proba(Xte)
        acc_scores['NRF'].append(accuracy_score(yte, p))
        f1_scores['NRF'].append(f1_score(yte, p, average=avg_kw))
        auc = (
            roc_auc_score(yte, prb[:, 1]) if C == 2
            else _fold_auc_ovr(yte, prb, clf.classes_, all_classes)
        )
        auc_scores['NRF'].append(auc)

        for tag in ([] if skip_baselines else ['LR', 'SVM', 'XGB', 'RF', 'NN']):
            clf_obj = _make_baseline_clf(tag, baseline_cfgs)
            if tag == 'SVM' and len(Xtr) > SVM_N_MAX:
                # Replace kernel SVM with calibrated LinearSVC for large datasets
                clf_obj = CalibratedClassifierCV(
                    LinearSVC(max_iter=2000, random_state=SEED), cv=3)
            clf_obj.fit(Xtr, ytr)
            p   = clf_obj.predict(Xte)
            prb = clf_obj.predict_proba(Xte)
            acc_scores[tag].append(accuracy_score(yte, p))
            f1_scores[tag].append(f1_score(yte, p, average=avg_kw))
            auc = (
                roc_auc_score(yte, prb[:, 1]) if C == 2
                else _fold_auc_ovr(yte, prb, clf_obj.classes_, all_classes)
            )
            auc_scores[tag].append(auc)

        print(f'  Fold {fold+1}/{n_folds}')

    rows = []
    print(f'\n  {"Method":<10}  {"Acc mean±std":>16}   {"F1 mean±std":>16}   {"AUC mean±std":>16}')
    print(f'  {"-"*68}')
    best_acc = max(np.mean(acc_scores[m]) for m in methods)
    for m in methods:
        a = np.array(acc_scores[m]) * 100
        f = np.array(f1_scores[m]) * 100
        u = np.array(auc_scores[m]) * 100
        star = ' *' if np.mean(acc_scores[m]) == best_acc else ''
        rows.append({'Dataset': dataset_name, 'Method': m,
                     'Acc': a.mean(), 'Acc_std': a.std(),
                     'F1':  f.mean(), 'F1_std':  f.std(),
                     'AUC': np.nanmean(u), 'AUC_std': np.nanstd(u),
                     'Acc_folds': list(a), 'F1_folds': list(f), 'AUC_folds': list(u)})
        print(
            f'  {m:<10}  {a.mean():>6.2f} ± {a.std():>4.2f}   '
            f'{f.mean():>6.2f} ± {f.std():>4.2f}   '
            f'{np.nanmean(u):>6.2f} ± {np.nanstd(u):>4.2f}{star}'
        )
    return pd.DataFrame(rows)


# ── Ablation study ────────────────────────────────────────────────────────────

def run_ablation(datasets, n_folds=3, nrf_cfg=None):
    '''
    Ablation of core NRF architectural choices.
    Covers transformer/routers and RF-style randomization switches.
    '''
    variants = {
        'Full':             dict(use_transformer=True,  linear_routers=False, feature_subset=False, rf_routing=False),
        'No Transformer':   dict(use_transformer=False, linear_routers=False, feature_subset=False, rf_routing=False),
        'Linear Routers':   dict(use_transformer=True,  linear_routers=True,  feature_subset=False, rf_routing=False),
        'Feature Subset':   dict(use_transformer=True,  linear_routers=False, feature_subset=True,  rf_routing=False),
        'RF Routing':       dict(use_transformer=True,  linear_routers=False, feature_subset=False, rf_routing=True),
        'Subset + RF Route': dict(use_transformer=True, linear_routers=False, feature_subset=True,  rf_routing=True),
    }

    if nrf_cfg is not None:
        base = nrf_cfg.copy()
        print('Ablation using Optuna best config')
    else:
        base = dict(n_trees=30, depth=3, h_dim=32, router_hidden=16, solver_hidden=16,
                    epochs=150, batch_size=512, lr=0.003, patience=20, min_epochs=50,
                    temp_start=1.0, temp_end=0.3, entropy_reg=0.01,
                    dropout=0.1, weight_decay=1e-4,
                    router_dropout=0.05, max_noise_std=0.05)

    all_rows = []
    for name, (X, y) in datasets.items():
        C      = len(np.unique(y))
        avg_kw = 'macro' if C > 2 else 'binary'
        skf    = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

        print(f'\n{"="*52}\nAblation -- {name}')
        print(f'  {"Variant":<22}  {"Accuracy":>14}  {"F1-macro":>14}')
        print(f'  {"-"*56}')

        for var_name, flags in variants.items():
            accs, f1s = [], []
            for tr, te in skf.split(X, y):
                scaler   = StandardScaler().fit(X[tr])
                Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
                cfg = {k: v for k, v in base.items() if k not in flags}
                cfg.update(flags)
                clf = NRF(**cfg)
                clf.fit(Xtr, y[tr], verbose=False)
                p = clf.predict(Xte)
                accs.append(accuracy_score(y[te], p))
                f1s.append(f1_score(y[te], p, average=avg_kw))
            a = np.array(accs) * 100
            f = np.array(f1s)  * 100
            all_rows.append({'Dataset': name, 'Variant': var_name,
                             'Acc': a.mean(), 'Acc_std': a.std(),
                             'F1':  f.mean(), 'F1_std':  f.std()})
            print(f'  {var_name:<22}  {a.mean():>6.2f}+/-{a.std():>4.1f}  {f.mean():>6.2f}+/-{f.std():>4.1f}')

    return pd.DataFrame(all_rows)
