from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import torch

CIFAR_FEATURE_CACHE = Path('cache/cifar10_resnet18_features.npz')
MNIST_FEATURE_CACHE = Path('cache/mnist_resnet18_features.npz')

DATASET_DISPLAY = {
    'magic':   'Magic',
    'eeg':     'EEG',
    'adult':   'Adult',
    'letter':  'Letter',
    'shuttle': 'Shuttle',
    'higgs':   'HIGGS',
    'covertype': 'Covertype',
    'otto':    'Otto',
    'cifar10_resnet18': 'CIFAR10_ResNet18',
    'mnist_resnet18': 'MNIST_ResNet18',
    'wine':    'Wine',
    'wdbc':    'WDBC',
    'isolet':  'Isolet',
    'pokerhand': 'PokerHand',
}
DATASET_SLUGS = tuple(DATASET_DISPLAY.keys())

KAGGLE_DATASET_SOURCES = {
    'magic': {
        'handle': 'abhinand05/magic-gamma-telescope-dataset',
        'display': 'Magic',
        'target_candidates': ('class', 'target', 'label'),
        'prefer_files': ('telescope_data.csv',),
    },
    'eeg': {
        'handle': 'robikscube/eye-state-classification-eeg-dataset',
        'display': 'EEG',
        'target_candidates': ('eye_detection', 'eye_state', 'class', 'label', 'target'),
        'prefer_files': ('EEG_Eye_State_Classification.csv',),
    },
    'adult': {
        'handle': 'uciml/adult-census-income',
        'display': 'Adult',
        'target_candidates': ('income', 'class', 'label', 'target'),
        'prefer_files': ('adult.csv',),
    },
    'letter': {
        'handle': 'datajameson/letter-recognition-dataset',
        'display': 'Letter',
        'target_candidates': ('lettr', 'letter', 'class', 'label', 'target'),
        'prefer_files': ('letter-recognition.data',),
    },
    'higgs': {
        'handle': 'erikbiswas/higgs-uci-dataset',
        'display': 'HIGGS',
        'target_candidates': ('class', 'target', 'label'),
        'target_col': 0,  # UCI CSV has no header; binary label is always first column
        'prefer_files': ('higgs.csv', 'HIGGS.csv'),
    },
    'covertype': {
        'handle': 'uciml/forest-cover-type-dataset',
        'display': 'Covertype',
        'target_candidates': ('cover_type', 'covertype', 'class', 'target', 'label'),
        'prefer_files': ('covtype.csv', 'forest_cover_type.csv'),
    },
    'otto': {
        'handle': 'msandipan98/otto-group-product-classification',
        'display': 'Otto',
        'target_candidates': ('target', 'class', 'label'),
        'prefer_files': ('train.csv', 'otto.csv'),
    },
}

def _load_cifar10_resnet18_features():
    """CIFAR-10 represented by ImageNet-pretrained ResNet-18 features.

    Forward pass is cached to disk; subsequent calls load from cache.
    Returns (X, y) with X of shape (60000, 512) and y of shape (60000,).
    """
    if CIFAR_FEATURE_CACHE.exists():
        npz = np.load(CIFAR_FEATURE_CACHE)
        return npz['X'].astype(np.float32), npz['y'].astype(np.int64)

    import torchvision
    from torchvision import transforms
    from torchvision.models import resnet18, ResNet18_Weights
    from torch.utils.data import DataLoader, ConcatDataset

    weights = ResNet18_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()

    data_root = Path(os.environ.get('CIFAR_DATA_ROOT', 'cache/cifar10'))
    data_root.mkdir(parents=True, exist_ok=True)
    train = torchvision.datasets.CIFAR10(
        root=str(data_root), train=True, download=True, transform=preprocess)
    test = torchvision.datasets.CIFAR10(
        root=str(data_root), train=False, download=True, transform=preprocess)
    full = ConcatDataset([train, test])  # 60_000 samples total

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet18(weights=weights).to(device).eval()
    # Replace final FC with identity so model outputs the 512-dim
    # penultimate layer.
    model.fc = torch.nn.Identity()

    loader = DataLoader(full, batch_size=256, shuffle=False,
                        num_workers=4, pin_memory=True)

    feats, labels = [], []
    with torch.inference_mode():
        for xb, yb in loader:
            f = model(xb.to(device, non_blocking=True))
            feats.append(f.detach().cpu().numpy())
            labels.append(yb.numpy())
    X = np.concatenate(feats, axis=0).astype(np.float32)
    y = np.concatenate(labels, axis=0).astype(np.int64)

    CIFAR_FEATURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CIFAR_FEATURE_CACHE, X=X, y=y)
    return X, y


def _load_mnist_resnet18_features():
    """MNIST represented by ImageNet-pretrained ResNet-18 features.

    Forward pass is cached to disk; subsequent calls load from cache.
    Returns (X, y) with X of shape (70000, 512) and y of shape (70000,).
    """
    if MNIST_FEATURE_CACHE.exists():
        npz = np.load(MNIST_FEATURE_CACHE)
        return npz['X'].astype(np.float32), npz['y'].astype(np.int64)

    import torchvision
    from torchvision import transforms
    from torchvision.models import resnet18, ResNet18_Weights
    from torch.utils.data import DataLoader, ConcatDataset

    weights = ResNet18_Weights.IMAGENET1K_V1
    # MNIST is single-channel PIL; ImageNet preprocess expects RGB.
    preprocess = transforms.Compose([
        transforms.Lambda(lambda im: im.convert('RGB')),
        weights.transforms(),
    ])

    data_root = Path(os.environ.get('MNIST_DATA_ROOT', 'cache/mnist'))
    data_root.mkdir(parents=True, exist_ok=True)
    train = torchvision.datasets.MNIST(
        root=str(data_root), train=True, download=True, transform=preprocess)
    test = torchvision.datasets.MNIST(
        root=str(data_root), train=False, download=True, transform=preprocess)
    full = ConcatDataset([train, test])  # 70_000 samples total

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet18(weights=weights).to(device).eval()
    model.fc = torch.nn.Identity()

    loader = DataLoader(full, batch_size=256, shuffle=False,
                        num_workers=4, pin_memory=True)

    feats, labels = [], []
    with torch.inference_mode():
        for xb, yb in loader:
            f = model(xb.to(device, non_blocking=True))
            feats.append(f.detach().cpu().numpy())
            labels.append(yb.numpy())
    X = np.concatenate(feats, axis=0).astype(np.float32)
    y = np.concatenate(labels, axis=0).astype(np.int64)

    MNIST_FEATURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(MNIST_FEATURE_CACHE, X=X, y=y)
    return X, y

def _load_wine():
    """UCI Wine: 178 samples × 13 features, 3 classes (sklearn built-in)."""
    from sklearn.datasets import load_wine
    d = load_wine()
    return d.data.astype(np.float32), d.target.astype(np.int64)


def _load_wdbc():
    """Wisconsin Diagnostic Breast Cancer: 569 × 30, 2 classes (sklearn built-in)."""
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer()
    return d.data.astype(np.float32), d.target.astype(np.int64)


def _read_headerless_partitions(dataset_dir: Path):
    """Read and concat every *.data/*.csv table in a Kaggle dataset dir.

    Handles UCI partitions (train + test) that the generic loader would
    split. Each file is read headerless; if the first row turns out to be a
    text header (last column non-numeric), it is re-read with skiprows=1.
    """
    files = []
    for pattern in ('*.data', '*.csv'):
        files.extend(sorted(dataset_dir.rglob(pattern)))
    if not files:
        raise FileNotFoundError(f'No *.data/*.csv files found in {dataset_dir}')

    frames = []
    for path in files:
        df = pd.read_csv(path, header=None)
        # UCI files are headerless; drop a stray text header if one slipped in.
        first_cell = str(df.iloc[0, -1]).strip().rstrip('.')
        if pd.to_numeric(pd.Series([first_cell]), errors='coerce').isna().all():
            df = pd.read_csv(path, header=None, skiprows=1)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def _load_isolet():
    """ISOLET spoken-letter dataset: ~7797 × 617, 26 classes (Kaggle/UCI).

    617 continuous features in [-1, 1]; last column is the class label
    1-26 (sometimes written with a trailing period, e.g. "3.").
    """
    dataset_dir = Path(kagglehub.dataset_download('gorangsolanki/isolet-dataset'))
    df = _read_headerless_partitions(dataset_dir)

    y_raw = df.iloc[:, -1].astype(str).str.strip().str.rstrip('.')
    X_df = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')
    mask = X_df.notna().all(axis=1) & y_raw.notna()
    X = X_df[mask].values.astype(np.float32)
    y = LabelEncoder().fit_transform(y_raw[mask])
    return X, y


def _load_pokerhand():
    """UCI Poker Hand: 1,025,010 × 10, 10 classes (Kaggle/UCI).

    10 integer features (suit/rank of 5 cards); last column is the hand
    class 0-9. Train + test partitions are concatenated.
    """
    dataset_dir = Path(kagglehub.dataset_download('rasvob/uci-poker-hand-dataset'))
    df = _read_headerless_partitions(dataset_dir)

    y_raw = df.iloc[:, -1]
    X_df = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')
    mask = X_df.notna().all(axis=1) & y_raw.notna()
    X = X_df[mask].values.astype(np.float32)
    y = LabelEncoder().fit_transform(y_raw[mask].astype(str))
    return X, y

def _load_shuttle():
    from ucimlrepo import fetch_ucirepo
    d = fetch_ucirepo(id=148)                       # Statlog (Shuttle) = 58,000 rows
    X = d.data.features.apply(pd.to_numeric, errors='coerce').values.astype(np.float32)
    y = LabelEncoder().fit_transform(d.data.targets.iloc[:, 0].astype(str))
    assert X.shape[0] == 58000, f'expected 58000 rows, got {X.shape[0]}'
    return X, y

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().lower().replace(' ', '_') for col in df.columns]
    return df

_ID_COL_NAMES = {'id', 'index', 'idx', 'row', 'row_id', 'rownum', 'unnamed:_0', 'unnamed:0'}

def _drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove index/ID columns. When a source file is sorted by class, a row
    index (Magic's saved pandas index, Otto's 'id') becomes a perfect predictor
    of the label and leaks it into the features. Drops columns whose name is an
    id/index, or whose values are a 1:1 row enumeration (a run of unique ints)."""
    n = len(df)
    drop = []
    for col in df.columns:
        name = str(col).strip().lower()
        if name in _ID_COL_NAMES or name.startswith('unnamed'):
            drop.append(col)
            continue
        s = pd.to_numeric(df[col], errors='coerce')
        if s.notna().all() and s.nunique() == n:
            v = np.sort(s.to_numpy())
            if np.array_equal(v, np.arange(v[0], v[0] + n)):  # 0..n-1 or 1..n
                drop.append(col)
    return df.drop(columns=drop)

def _detect_target_column(df: pd.DataFrame, target_candidates, target_col=None):
    if target_col is not None:
        return df.columns[target_col]
    for candidate in target_candidates:
        if candidate in df.columns:
            return candidate
    return df.columns[-1]


def _read_dataset_file(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in ('.csv', '.data'):
        return pd.read_csv(path)
    if ext in ('.txt', '.trn', '.tst'):
        return pd.read_csv(path, sep=r'\s+', header=None)
    raise ValueError(f'Unsupported dataset file type: {path.name}')


def _select_dataset_file(dataset_dir: Path, prefer_files):
    for prefer_name in prefer_files:
        prefer_path = dataset_dir / prefer_name
        if prefer_path.exists():
            return prefer_path

    candidates = []
    for pattern in ('*.csv', '*.data', '*.txt', '*.trn', '*.tst'):
        candidates.extend(sorted(dataset_dir.rglob(pattern)))

    if not candidates:
        raise FileNotFoundError(f'No supported data files found in {dataset_dir}')

    return candidates[0]


def _load_from_kaggle(config):
    dataset_dir = Path(kagglehub.dataset_download(config['handle']))
    data_file = _select_dataset_file(dataset_dir, config['prefer_files'])
    df = _read_dataset_file(data_file)

    if df.empty:
        raise ValueError(f'Loaded dataset is empty: {data_file}')

    df = _normalize_columns(df)
    target_col = _detect_target_column(df, config['target_candidates'],
                                       config.get('target_col'))

    y_raw = df[target_col]
    X_df = df.drop(columns=[target_col])
    X_df = _drop_identifier_columns(X_df)

    cat_cols = X_df.select_dtypes(include=['category', 'object']).columns.tolist()
    if cat_cols:
        X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=True)

    X_df = X_df.apply(pd.to_numeric, errors='coerce')
    mask = X_df.notna().all(axis=1) & y_raw.notna()
    X = X_df[mask].values.astype(np.float32)
    y = LabelEncoder().fit_transform(y_raw[mask].astype(str))
    return X, y


def load_datasets():
    datasets = {}

    for slug in DATASET_SLUGS:
        display = DATASET_DISPLAY[slug]
        try:
            if slug == 'cifar10_resnet18':
                X, y = _load_cifar10_resnet18_features()
            elif slug == 'mnist_resnet18':
                X, y = _load_mnist_resnet18_features()
            elif slug == 'wine':
                X, y = _load_wine()
            elif slug == 'wdbc':
                X, y = _load_wdbc()
            elif slug == 'isolet':
                X, y = _load_isolet()
            elif slug == 'pokerhand':
                X, y = _load_pokerhand()
            elif slug == 'shuttle':
                X, y = _load_shuttle()
            else:
                config = KAGGLE_DATASET_SOURCES[slug]
                X, y = _load_from_kaggle(config)
            datasets[display] = (X, y)
            print(f'{display:<14}{X.shape[0]}x{X.shape[1]}, '
                  f'{len(np.unique(y))} classes')
        except Exception as e:
            print(f'Could not load {display}: {e}')

    return datasets
