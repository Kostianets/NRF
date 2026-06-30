import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import warnings
import numpy as np
import random
import time
import threading
from concurrent.futures import ThreadPoolExecutor

from torch.amp import autocast, GradScaler
from sklearn.preprocessing import LabelEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

class SoftNeuralTree(nn.Module):
    '''
    Soft Decision Tree — improved version.

    Components and origins:
      [ANT, Tanno 2019]        Shared MLP feature transformer (use_transformer=True)
      [Deep NDF, K. 2015]      Soft routing: sigmoid routers, mixture of leaf predictions
      [NNRF, Wang 2017]        feature_indices: per-tree random feature subspace
      [DNDT, Yang 2018 adapt]  temperature parameter: controls routing sharpness
      [Own]                    use_transformer / linear_routers flags for ablation
      [Own]  PERF              Batched routers/solvers via einsum — eliminates Python loops
      [Own]  PERF + NUM        Vectorised log-space leaf probs — faster & no float underflow
      [Own]  ARCH-1            LayerNorm in transformer — stable on tiny batches/datasets
      [Own]  ARCH-2            Dropout in router hidden layer — prevents router overfitting
    '''

    def __init__(self, input_dim, n_classes, depth=3, h_dim=32,
                 router_hidden=16, solver_hidden=16, dropout=0.1,
                 router_dropout=0.05,
                 feature_indices=None,
                 use_transformer=True,
                 linear_routers=False,
                 rf_routing=False):
        super().__init__()
        self.depth          = depth
        self.n_classes      = n_classes
        self.n_internal     = 2**depth - 1
        self.n_leaves       = 2**depth
        self.dropout        = dropout
        self.router_dropout = router_dropout
        self.linear_routers = linear_routers
        self.rf_routing     = rf_routing

        # per-node RF subsets (rf_routing=True) override per-tree subsets
        if rf_routing:
            self.feature_indices      = None
            r = max(1, int(np.sqrt(input_dim)))
            node_feats = np.stack([
                np.sort(np.random.choice(input_dim, r, replace=False))
                for _ in range(self.n_internal)
            ])  # (n_internal, r)
            self.register_buffer('node_feature_indices', torch.LongTensor(node_feats))
            self.subset_size = r
            effective_dim = input_dim   # transformer still sees full input
        else:
            self.node_feature_indices = None
            self.subset_size          = None
            if feature_indices is not None:
                self.register_buffer('feature_indices', torch.LongTensor(feature_indices))
            else:
                self.feature_indices = None
            effective_dim = len(feature_indices) if feature_indices is not None else input_dim

        # ── [ANT] Shared Feature Transformer ──────────────────────────────────
        if use_transformer:
            self.transformer = nn.Sequential(
                nn.Linear(effective_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),
                nn.Linear(h_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.LeakyReLU(0.1),
            )
        else:
            self.transformer = nn.Linear(effective_dim, h_dim)

        # ── [PERF-2] Batched Routers ──────────────────────────────────────────
        router_in = self.subset_size if rf_routing else h_dim
        if linear_routers:
            if rf_routing:
                # one weight vector per node, applied to its raw feature subset
                self.router_linear = nn.Parameter(torch.empty(self.n_internal, router_in))
                for i in range(self.n_internal):
                    nn.init.kaiming_uniform_(self.router_linear[i].unsqueeze(0), a=0.1)
            else:
                self.router = nn.Linear(h_dim, self.n_internal, bias=True)
        else:
            self.router_W1 = nn.Parameter(torch.empty(self.n_internal, router_in, router_hidden))
            self.router_b1 = nn.Parameter(torch.zeros(self.n_internal, router_hidden))
            self.router_W2 = nn.Parameter(torch.empty(self.n_internal, router_hidden, 1))
            self.router_b2 = nn.Parameter(torch.zeros(self.n_internal, 1))
            for i in range(self.n_internal):
                nn.init.kaiming_uniform_(self.router_W1[i], a=0.1)
                nn.init.kaiming_uniform_(self.router_W2[i], a=0.1)

        # ── [PERF-2] Batched Solvers ──────────────────────────────────────────
        self.solver_W1 = nn.Parameter(torch.empty(self.n_leaves, h_dim, solver_hidden))
        self.solver_b1 = nn.Parameter(torch.zeros(self.n_leaves, solver_hidden))
        self.solver_W2 = nn.Parameter(torch.empty(self.n_leaves, solver_hidden, n_classes))
        self.solver_b2 = nn.Parameter(torch.zeros(self.n_leaves, n_classes))
        for i in range(self.n_leaves):
            nn.init.kaiming_uniform_(self.solver_W1[i], a=0.1)
            nn.init.kaiming_uniform_(self.solver_W2[i], a=0.1)

        # ── [PERF-1] Precompute left/right path masks ─────────────────────────
        leaf_start = self.n_leaves   # local only — not stored on self
        path_info  = []
        for l in range(leaf_start, leaf_start + self.n_leaves):
            path = []
            node = l
            while node > 1:
                parent  = node // 2
                is_left = (node == 2 * parent)
                path.append((parent - 1, is_left))
                node = parent
            path_info.append(path)

        left_mask  = torch.zeros(self.n_leaves, self.n_internal)
        right_mask = torch.zeros(self.n_leaves, self.n_internal)
        for leaf_idx, path in enumerate(path_info):
            for router_idx, is_left in path:
                if is_left:
                    left_mask[leaf_idx, router_idx]  = 1.0
                else:
                    right_mask[leaf_idx, router_idx] = 1.0
        self.register_buffer('left_mask',  left_mask)
        self.register_buffer('right_mask', right_mask)

    def forward(self, x, temperature=1.0):
        # per-tree feature subset (only when rf_routing=False)
        if self.feature_indices is not None:
            x = x[:, self.feature_indices]

        h = self.transformer(x)   # solvers always use transformer output

        if self.rf_routing:
            # each node sees its own sqrt(m) raw features — true RF routing
            # x_nodes: (batch, n_internal, subset_size)
            x_nodes = x[:, self.node_feature_indices]
            if self.linear_routers:
                route_logits = (x_nodes * self.router_linear).sum(-1)  # (batch, n_internal)
            else:
                r1 = torch.einsum('bns,nsh->bnh', x_nodes, self.router_W1) + self.router_b1
                r1 = F.leaky_relu(r1, 0.1)
                r1 = F.dropout(r1, p=self.router_dropout, training=self.training)
                r2 = torch.einsum('bnh,nho->bno', r1, self.router_W2) + self.router_b2
                route_logits = r2.squeeze(-1)
        elif self.linear_routers:
            route_logits = self.router(h)
        else:
            r1 = torch.einsum('bh,nhr->bnr', h, self.router_W1) + self.router_b1
            r1 = F.leaky_relu(r1, 0.1)
            r1 = F.dropout(r1, p=self.router_dropout, training=self.training)
            r2 = torch.einsum('bnr,nro->bno', r1, self.router_W2) + self.router_b2
            route_logits = r2.squeeze(-1)

        route_probs = torch.sigmoid(route_logits / temperature)
        route_probs = route_probs.clamp(1e-6, 1 - 1e-6)

        log_left  = torch.log(route_probs)
        log_right = torch.log(1 - route_probs)
        leaf_log_probs = (log_left  @ self.left_mask.T
                        + log_right @ self.right_mask.T)
        leaf_probs = torch.exp(leaf_log_probs)

        s1 = torch.einsum('bh,lhs->bls', h, self.solver_W1) + self.solver_b1
        s1 = F.leaky_relu(s1, 0.1)
        s1 = F.dropout(s1, p=self.dropout, training=self.training)
        s2 = torch.einsum('bls,lsc->blc', s1, self.solver_W2) + self.solver_b2
        leaf_preds = F.softmax(s2, dim=-1)

        output = (leaf_probs.unsqueeze(2) * leaf_preds).sum(dim=1)
        return output, leaf_probs


def train_tree(model, X_train, y_train, epochs=150, batch_size=512, lr=0.003,
               weight_decay=1e-4, bootstrap=True, patience=20, min_epochs=60,
               temp_start=1.0, temp_end=0.3, entropy_reg=0.01,
               noise_std=0.0, verbose=False, temp_schedule='linear'):
    '''
    Train a single SoftNeuralTree.
    [BUG-3 fix]  sched_patience = max(patience, min_epochs//3)
    [BUG-5 fix]  Validation always at temp_end — stationary signal across all epochs
    [ARCH-3]     noise_std > 0: per-tree Gaussian input noise for ensemble diversity
    [ENS-1]      Returns oob_acc for weighted ensemble voting
    [DEF-Q3]     temp_schedule: 'linear' (canonical), 'exp' (geometric decay
                 between the same endpoints), or 'const_end' (T = temp_end from
                 epoch 0 — no annealing, sharp from the start). All schedules
                 end at temp_end, so validation/OOB/inference (always at
                 temp_end) stay consistent across schedules.
    '''
    n = X_train.shape[0]

    if bootstrap:
        idx      = np.random.choice(n, size=n, replace=True)
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[np.unique(idx)] = False
        X_boot, y_boot = X_train[idx], y_train[idx]
        if oob_mask.sum() >= 10:
            X_val_np, y_val_np = X_train[oob_mask], y_train[oob_mask]
        else:
            split = max(1, n // 6)
            X_val_np, y_val_np = X_boot[:split], y_boot[:split]
    else:
        split    = max(1, int(0.15 * n))
        X_val_np, y_val_np = X_train[:split], y_train[:split]
        X_boot, y_boot     = X_train[split:], y_train[split:]
        oob_mask = None

    if noise_std > 0.0:
        X_boot = X_boot + (np.random.randn(*X_boot.shape) * noise_std).astype(np.float32)

    X_t = torch.FloatTensor(X_boot).to(device)
    y_t = torch.LongTensor(y_boot).to(device)
    X_val_t = torch.FloatTensor(X_val_np).to(device)
    y_val_t = torch.LongTensor(y_val_np).to(device)

    #trainable = [p for p in model.parameters() if p.requires_grad]
    #optimizer = optim.Adam(trainable, lr=lr, weight_decay=weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched_patience = max(patience, min_epochs // 3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=sched_patience, min_lr=1e-5)
    scaler = GradScaler('cuda')

    best_val_loss    = float('inf')
    patience_counter = 0
    # Seed best_state with initial weights so that, if training goes NaN before
    # any improvement is recorded, we can fall back to a valid (untrained)
    # state instead of returning NaN weights that would poison the ensemble.
    init_state       = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_state       = init_state
    losses           = []
    nonfinite_epochs = 0

    for ep in range(epochs):
        progress    = ep / max(1, epochs - 1)
        if temp_schedule == 'exp':
            temperature = temp_start * (temp_end / temp_start) ** progress
        elif temp_schedule == 'const_end':
            temperature = temp_end
        else:  # 'linear' (canonical)
            temperature = temp_start + (temp_end - temp_start) * progress

        model.train()
        ep_loss = torch.zeros(1, device=device)
        n_batches = 0
        _perm = torch.randperm(len(X_t), device=device)
        for _start in range(0, len(X_t), batch_size):
            _idx = _perm[_start:_start + batch_size]
            X_batch, y_batch = X_t[_idx], y_t[_idx]
            optimizer.zero_grad()
            with autocast('cuda'):
                pred, leaf_probs = model(X_batch, temperature=temperature)
                y_oh = F.one_hot(y_batch, model.n_classes).float()
                nll  = -torch.mean(torch.sum(y_oh * torch.log(pred + 1e-8), dim=1))
                if entropy_reg > 0:
                    avg_leaf     = leaf_probs.mean(dim=0)
                    leaf_entropy = -(avg_leaf * torch.log(avg_leaf + 1e-8)).sum()
                    loss = nll - entropy_reg * leaf_entropy
                else:
                    loss = nll
            # Mixed-precision forward can overflow (esp. use_transformer=False
            # with large h_dim or deep trees). Skip backward/step on non-finite
            # loss so we don't pollute weights. Do NOT call scaler.update() —
            # it asserts that an inf-check was recorded by a prior scaler.step,
            # which didn't happen here. Leaving the scale unchanged is fine:
            # the next finite batch will run the normal path, and the natural
            # scaler.step on inf grads will adapt the scale if needed.
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            ep_loss  += nll.detach()
            n_batches += 1

        if n_batches == 0:
            # Every batch overflowed — abort if it persists; otherwise let
            # GradScaler keep adapting on subsequent epochs.
            nonfinite_epochs += 1
            if nonfinite_epochs >= 3:
                break
            continue
        nonfinite_epochs = 0
        avg_loss_t = ep_loss / n_batches

        model.eval()
        with torch.no_grad():
            val_loss_acc = torch.zeros(1, device=device)
            val_n = 0
            for v0 in range(0, len(X_val_t), batch_size):
                xv      = X_val_t[v0:v0 + batch_size]
                yv      = y_val_t[v0:v0 + batch_size]
                pred, _ = model(xv, temperature=temp_end)
                y_oh_v  = F.one_hot(yv, model.n_classes).float()
                val_loss_acc += -torch.sum(
                    torch.sum(y_oh_v * torch.log(pred + 1e-8), dim=1)
                )
                val_n += len(xv)
            val_loss = (val_loss_acc / val_n).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss    = val_loss
            best_state       = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            if ep >= min_epochs:
                patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f'    Early stop at epoch {ep+1}')
                break

        if verbose and (ep + 1) % 20 == 0:
            avg_loss = avg_loss_t.item()
            losses.append(avg_loss)
            print(f'    Ep {ep+1:3d} | train={avg_loss:.4f} | val={val_loss:.4f} | T={temperature:.3f}')

    if best_state is not None:
        model.load_state_dict(best_state)
    # Last-line defense: if the chosen state still has any non-finite params
    # (best_state never updated, or a corrupt checkpoint), reset to the
    # untrained init. An untrained tree will be near-chance on OOB and get
    # ensemble weight 0.5 — far better than a NaN tree contaminating predict.
    if any((~torch.isfinite(p)).any().item() for p in model.parameters()):
        model.load_state_dict(init_state)

    oob_acc = 0.5
    if oob_mask is not None and oob_mask.sum() >= 5:
        X_oob, y_oob = X_train[oob_mask], y_train[oob_mask]
        X_oob_t = torch.FloatTensor(X_oob).to(device)
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for v0 in range(0, len(X_oob_t), batch_size):
                xv      = X_oob_t[v0:v0 + batch_size]
                out, _  = model(xv, temperature=temp_end)
                correct += (out.argmax(1).cpu().numpy() == y_oob[v0:v0 + batch_size]).sum()
                total   += len(xv)
        oob_acc = float(correct / total)

    return losses, oob_mask, oob_acc


# ── Per-tree multi-GPU training ───────────────────────────────────────────────
#
# Spawn one worker process per GPU; round-robin trees across workers. Trees are
# independent, so this is embarrassingly parallel with near-linear speedup.
# Each worker pins itself to one GPU via CUDA_VISIBLE_DEVICES before importing
# torch, so the module-level `device = cuda:0` resolves to the correct device
# inside the worker.

def _fit_one_tree(args):
    '''Train a single SoftNeuralTree. Runs in a worker process pinned to one
    GPU. Returns (tree_idx, cpu_state_dict, oob_acc, tree_depth, feat_idx).'''
    (tree_idx, X, y_enc, tree_depth, feat_idx, noise_std,
     C, m, base_seed, nrf_cfg) = args

    # Per-tree deterministic seed so bootstrap/noise reproduce regardless of
    # which worker happens to pick up this tree.
    np.random.seed(base_seed + tree_idx)
    torch.manual_seed(base_seed + tree_idx)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_seed + tree_idx)

    # In a worker process, _worker_entry has already redirected the
    # module-level `device` global to this worker's assigned GPU. In the
    # single-GPU path it's just the original cuda:0 / cpu.
    local_device = device
    tree = SoftNeuralTree(
        input_dim=m, n_classes=C, depth=tree_depth,
        h_dim=nrf_cfg['h_dim'],
        router_hidden=nrf_cfg['router_hidden'],
        solver_hidden=nrf_cfg['solver_hidden'],
        dropout=nrf_cfg['dropout'],
        router_dropout=nrf_cfg['router_dropout'],
        feature_indices=(None if nrf_cfg['rf_routing'] else feat_idx),
        use_transformer=nrf_cfg['use_transformer'],
        linear_routers=nrf_cfg['linear_routers'],
        rf_routing=nrf_cfg['rf_routing'],
    ).to(local_device)

    _, _, oob_acc = train_tree(
        tree, X, y_enc,
        epochs=nrf_cfg['epochs'], batch_size=nrf_cfg['batch_size'],
        lr=nrf_cfg['lr'], weight_decay=nrf_cfg['weight_decay'],
        bootstrap=True, patience=nrf_cfg['patience'],
        min_epochs=nrf_cfg['min_epochs'],
        temp_start=nrf_cfg['temp_start'], temp_end=nrf_cfg['temp_end'],
        entropy_reg=nrf_cfg['entropy_reg'],
        noise_std=noise_std, verbose=False,
        temp_schedule=nrf_cfg.get('temp_schedule', 'linear'),
    )

    # Convert to numpy before returning. Sending torch tensors through
    # mp.Queue triggers torch's FD-sharing reductions, which race with worker
    # exit and crash the parent with ConnectionResetError on Queue.get().
    # Numpy arrays pickle as raw bytes — no resource_sharer involvement.
    cpu_state = {k: v.detach().cpu().numpy() for k, v in tree.state_dict().items()}
    return tree_idx, cpu_state, oob_acc, tree_depth, feat_idx


def _worker_entry(gpu_id, worker_args, out_q):
    '''Pin process to one GPU then train its assigned trees sequentially.
    Setting CUDA_VISIBLE_DEVICES after torch is already imported (which
    happens during spawn re-import of the main script) is unreliable, so we
    use torch.cuda.set_device() and redirect the module-level `device`
    global instead. train_tree captures `device` at call time, so the
    redirect propagates correctly.'''
    import torch
    import src.model
    torch.cuda.set_device(gpu_id)
    src.model.device = torch.device(f'cuda:{gpu_id}')
    from src.model import _fit_one_tree
    for args in worker_args:
        try:
            result = _fit_one_tree(args)
            out_q.put(('ok', result))
        except Exception as e:
            import traceback
            out_q.put(('err', args[0], f'{repr(e)}\n{traceback.format_exc()}'))


def _parallel_fit(args_list, n_gpus, verbose=True):
    '''Spawn n_gpus workers; round-robin trees across them.'''
    import torch.multiprocessing as mp
    # Defensive: use file_system sharing rather than file_descriptor sharing
    # for any torch tensors that end up on the queue (e.g. inside wrapped
    # exception payloads). The FD scheme races with worker exit and crashes
    # the parent with ConnectionResetError. file_system uses /tmp-backed
    # shared memory which survives worker exit.
    try:
        mp.set_sharing_strategy('file_system')
    except RuntimeError:
        # Strategy already locked elsewhere in this process; ignore.
        pass
    ctx = mp.get_context('spawn')
    out_q = ctx.Queue()

    per_gpu = [[] for _ in range(n_gpus)]
    for args in args_list:
        per_gpu[args[0] % n_gpus].append(args)

    procs = []
    for gpu_id in range(n_gpus):
        if not per_gpu[gpu_id]:
            continue
        p = ctx.Process(target=_worker_entry,
                        args=(gpu_id, per_gpu[gpu_id], out_q))
        p.start()
        procs.append(p)

    results = []
    try:
        for i in range(len(args_list)):
            while True:
                try:
                    msg = out_q.get(timeout=30)
                    break
                except Exception:
                    # Timeout — check whether any worker died without sending a result.
                    crashed = [p for p in procs if not p.is_alive() and p.exitcode not in (None, 0)]
                    if crashed:
                        for p in procs:
                            if p.is_alive():
                                p.terminate()
                        codes = [p.exitcode for p in crashed]
                        raise RuntimeError(
                            f'Worker process(es) crashed (exit codes {codes}). '
                            f'Check for CUDA errors or OOM kills in dmesg.'
                        )
            if msg[0] == 'err':
                tag, tree_idx, info = msg
                # Kill remaining workers; surface to the parent.
                for p in procs:
                    if p.is_alive():
                        p.terminate()
                raise RuntimeError(f'tree {tree_idx} failed in worker:\n{info}')
            _, result = msg
            results.append(result)
            if verbose:
                print(f'  Tree {i+1}/{len(args_list)} done')
    finally:
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join()
    return results


class NRF:
    '''
    Ensemble of SoftNeuralTrees with bootstrap aggregating.
    [ENS-1] OOB-weighted prediction  [ENS-2] Depth diversity  [ARCH-3] Per-tree noise
    '''

    def __init__(self, n_trees=25, depth=3, h_dim=32,
                 router_hidden=16, solver_hidden=16,
                 dropout=0.1, router_dropout=0.05,
                 lr=0.003, weight_decay=1e-4,
                 epochs=150, batch_size=512,
                 patience=20, min_epochs=60,
                 temp_start=1.0, temp_end=0.3, entropy_reg=0.01,
                 feature_subset=False,
                 max_noise_std=0.05,
                 depth_range=None,
                 use_transformer=True, linear_routers=False,
                 rf_routing=False,
                 temp_schedule='linear', noise_mode='uniform'):
        self.n_trees        = n_trees
        self.depth          = depth
        self.h_dim          = h_dim
        self.router_hidden  = router_hidden
        self.solver_hidden  = solver_hidden
        self.dropout        = dropout
        self.router_dropout = router_dropout
        self.lr             = lr
        self.weight_decay   = weight_decay
        self.epochs         = epochs
        self.batch_size     = batch_size
        self.patience       = patience
        self.min_epochs     = min_epochs
        self.temp_start     = temp_start
        self.temp_end       = temp_end
        self.entropy_reg    = entropy_reg
        self.feature_subset  = feature_subset
        self.max_noise_std   = max_noise_std
        self.depth_range     = depth_range
        self.use_transformer = use_transformer
        self.linear_routers  = linear_routers
        self.rf_routing      = rf_routing
        # [DEF-Q3] temp_schedule: 'linear' | 'exp' | 'const_end'
        # [DEF-Q5] noise_mode: 'uniform' (σ_t ~ U[0, max_noise_std], canonical)
        #          | 'fixed' (σ_t = max_noise_std, one shared value for all trees)
        self.temp_schedule   = temp_schedule
        self.noise_mode      = noise_mode
        self.trees_          = []
        self.feat_subsets_   = []
        self.tree_weights_   = []
        self.classes_        = None
        self.input_dim_      = None
        self.tree_depths_    = []

    def fit(self, X, y, verbose=True, n_gpus=None):
        '''Train the ensemble. Trees are independent; with n_gpus > 1 they are
        trained in parallel via spawned worker processes, one per GPU.
        n_gpus=None auto-detects (NRF_N_GPUS env var, else torch.cuda.device_count()).'''
        self.classes_ = np.unique(y)
        C    = len(self.classes_)
        n, m = X.shape
        self.input_dim_ = m
        self.le    = LabelEncoder()
        y_enc      = self.le.fit_transform(y)
        r = max(1, int(np.sqrt(m))) if self.feature_subset else m

        if n_gpus is None:
            env_n = int(os.environ.get('NRF_N_GPUS', '0') or 0)
            n_gpus = env_n if env_n > 0 else (torch.cuda.device_count() or 1)
        n_gpus = max(1, int(n_gpus))

        if verbose:
            print(f'NRF | {self.n_trees} trees, depth={self.depth}, h={self.h_dim}')
            print(f'  {n}x{m} input | C={C} | subset={self.feature_subset}')
            print(f'  T: {self.temp_start}->{self.temp_end} | patience={self.patience} | min_ep={self.min_epochs}')
            dr = self.depth_range if self.depth_range else (self.depth, self.depth)
            print(f'  depth_range={dr} | max_noise={self.max_noise_std:.2f} | router_drop={self.router_dropout:.2f}')
            print(f'  parallelism: {n_gpus} GPU{"s" if n_gpus > 1 else ""}')

        # Pre-sample all per-tree randomness in the parent so the np.random
        # state evolves identically to the original sequential loop, preserving
        # reproducibility against SEED=42 on the single-GPU path.
        feat_subsets = [
            (np.sort(np.random.choice(m, size=r, replace=False))
             if self.feature_subset else None)
            for _ in range(self.n_trees)
        ]
        if self.depth_range is not None:
            tree_depths = [int(np.random.randint(self.depth_range[0],
                                                 self.depth_range[1] + 1))
                           for _ in range(self.n_trees)]
        else:
            tree_depths = [self.depth] * self.n_trees
        if self.noise_mode == 'fixed':
            # [DEF-Q5] one shared noise level applied identically to every tree
            noise_stds = [float(self.max_noise_std)] * self.n_trees
        else:
            noise_stds = [float(np.random.uniform(0.0, self.max_noise_std))
                          for _ in range(self.n_trees)]

        nrf_cfg = {k: getattr(self, k) for k in (
            'h_dim', 'router_hidden', 'solver_hidden', 'dropout', 'router_dropout',
            'epochs', 'batch_size', 'lr', 'weight_decay', 'patience', 'min_epochs',
            'temp_start', 'temp_end', 'entropy_reg',
            'use_transformer', 'linear_routers', 'rf_routing',
            'temp_schedule',
        )}

        args_list = [
            (i, X, y_enc, tree_depths[i], feat_subsets[i], noise_stds[i],
             C, m, SEED, nrf_cfg)
            for i in range(self.n_trees)
        ]

        t0 = time.time()
        if n_gpus <= 1:
            # Run trees concurrently on the same GPU using separate CUDA streams.
            # Threads share the CUDA context so their kernels overlap; the GPU
            # hardware interleaves them. Each slot gets a dedicated stream to
            # avoid serializing on the default stream.
            # Note: bootstrap/noise RNG is not perfectly reproducible under
            # concurrency (np.random global state races between threads), but
            # training correctness is unaffected.
            N_CONCURRENT = min(self.n_trees, 8)
            use_cuda_streams = torch.cuda.is_available()
            if use_cuda_streams:
                streams = [torch.cuda.Stream(device=device) for _ in range(N_CONCURRENT)]
            done_count = [0]
            lock = threading.Lock()

            def _run_with_stream(packed):
                slot_idx, args = packed
                if use_cuda_streams:
                    with torch.cuda.stream(streams[slot_idx % N_CONCURRENT]):
                        result = _fit_one_tree(args)
                else:
                    result = _fit_one_tree(args)
                #with torch.cuda.stream(streams[slot_idx % N_CONCURRENT]):
                    #result = _fit_one_tree(args)
                if verbose:
                    with lock:
                        done_count[0] += 1
                        cnt = done_count[0]
                    if cnt % max(1, self.n_trees // 4) == 0:
                        elapsed = time.time() - t0
                        print(f'  Tree {cnt}/{self.n_trees}'
                              f' | OOB={result[2]:.3f} | {elapsed:.1f}s')
                return result

            with ThreadPoolExecutor(max_workers=N_CONCURRENT) as ex:
                results = list(ex.map(_run_with_stream,
                                      [(i, a) for i, a in enumerate(args_list)]))
        else:
            results = _parallel_fit(args_list, n_gpus, verbose=verbose)

        results.sort(key=lambda r: r[0])

        self.trees_        = []
        self.feat_subsets_ = []
        self.tree_weights_ = []
        self.tree_depths_  = []
        for tree_idx, cpu_state, oob_acc, tree_depth, feat_idx in results:
            tree = SoftNeuralTree(
                input_dim=m, n_classes=C, depth=tree_depth,
                h_dim=self.h_dim, router_hidden=self.router_hidden,
                solver_hidden=self.solver_hidden, dropout=self.dropout,
                router_dropout=self.router_dropout,
                feature_indices=(None if self.rf_routing else feat_idx),
                use_transformer=self.use_transformer,
                linear_routers=self.linear_routers,
                rf_routing=self.rf_routing,
            ).to(device)
            # cpu_state values are numpy arrays (see _fit_one_tree note);
            # convert back to torch tensors before load_state_dict.
            torch_state = {k: torch.from_numpy(v) for k, v in cpu_state.items()}
            tree.load_state_dict(torch_state)
            tree.eval()
            self.trees_.append(tree)
            self.feat_subsets_.append(feat_idx)
            self.tree_depths_.append(tree_depth)
            self.tree_weights_.append(max(oob_acc, 0.5))

        if verbose:
            w = np.array(self.tree_weights_)
            print(f'Done in {time.time()-t0:.1f}s ({n_gpus} GPU'
                  f'{"s" if n_gpus > 1 else ""}) | OOB weights '
                  f'min={w.min():.3f} mean={w.mean():.3f} max={w.max():.3f}')
        return self

    def save(self, path):
        if not self.trees_:
            raise RuntimeError('Cannot save NRF before fit().')
        if self.input_dim_ is None:
            raise RuntimeError('Cannot save NRF: missing input_dim_.')

        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        payload = {
            'meta': {
                'n_trees': self.n_trees,
                'depth': self.depth,
                'h_dim': self.h_dim,
                'router_hidden': self.router_hidden,
                'solver_hidden': self.solver_hidden,
                'dropout': self.dropout,
                'router_dropout': self.router_dropout,
                'lr': self.lr,
                'weight_decay': self.weight_decay,
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'patience': self.patience,
                'min_epochs': self.min_epochs,
                'temp_start': self.temp_start,
                'temp_end': self.temp_end,
                'entropy_reg': self.entropy_reg,
                'feature_subset': self.feature_subset,
                'max_noise_std': self.max_noise_std,
                'depth_range': self.depth_range,
                'use_transformer': self.use_transformer,
                'linear_routers': self.linear_routers,
                'rf_routing': self.rf_routing,
                'temp_schedule': self.temp_schedule,
                'noise_mode': self.noise_mode,
            },
            'input_dim': self.input_dim_,
            'classes': self.classes_.tolist() if self.classes_ is not None else None,
            'label_classes': self.le.classes_.tolist(),
            'feat_subsets': [
                fs.tolist() if fs is not None else None
                for fs in self.feat_subsets_
            ],
            'tree_depths': list(self.tree_depths_),
            'tree_weights': list(self.tree_weights_),
            'tree_states': [tree.state_dict() for tree in self.trees_],
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path, map_location=None):
        ckpt = torch.load(path, map_location=(map_location or device))
        model = cls(**ckpt['meta'])
        model.input_dim_ = int(ckpt['input_dim'])
        model.classes_ = np.array(ckpt['classes'])

        model.le = LabelEncoder()
        model.le.classes_ = np.array(ckpt['label_classes'])

        model.feat_subsets_ = [
            np.array(fs, dtype=np.int64) if fs is not None else None
            for fs in ckpt['feat_subsets']
        ]
        model.tree_depths_ = [int(d) for d in ckpt['tree_depths']]
        model.tree_weights_ = [float(w) for w in ckpt['tree_weights']]
        model.trees_ = []

        for feat_idx, tree_depth, state in zip(
            model.feat_subsets_, model.tree_depths_, ckpt['tree_states']
        ):
            tree = SoftNeuralTree(
                input_dim=model.input_dim_,
                n_classes=len(model.classes_),
                depth=tree_depth,
                h_dim=model.h_dim,
                router_hidden=model.router_hidden,
                solver_hidden=model.solver_hidden,
                dropout=model.dropout,
                router_dropout=model.router_dropout,
                feature_indices=(None if model.rf_routing else feat_idx),
                use_transformer=model.use_transformer,
                linear_routers=model.linear_routers,
                rf_routing=model.rf_routing,
            ).to(device)
            tree.load_state_dict(state)
            tree.eval()
            model.trees_.append(tree)
        return model

    def predict_proba(self, X, pred_batch_size=512):
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        proba     = np.empty((n_samples, n_classes), dtype=np.float64)
        weights   = np.array(self.tree_weights_)
        weights   = weights / weights.sum()
        # Plain Python floats — avoids any tensor↔scalar sync inside the loop.
        weights   = [float(w) for w in weights]

        # set eval mode once — not inside the batch loop
        for model in self.trees_:
            model.eval()

        nan_rows_total = 0
        with torch.no_grad():
            for start in range(0, n_samples, pred_batch_size):
                end = min(start + pred_batch_size, n_samples)
                X_t = torch.as_tensor(X[start:end], dtype=torch.float32, device=device)
                # Accumulate weighted tree outputs on GPU; one CPU sync per batch
                # instead of one per tree per batch.
                acc = torch.zeros(end - start, n_classes, device=device)
                weight_sum = torch.zeros(end - start, 1, device=device)
                for w, model in zip(weights, self.trees_):
                    out, _ = model(X_t, temperature=self.temp_end)
                    # Drop non-finite rows from this tree — a single rogue
                    # tree (NaN weights from training-time fp16 overflow) must
                    # not poison the rest of the ensemble for that row.
                    row_ok = torch.isfinite(out).all(dim=1, keepdim=True)
                    acc.add_(torch.where(row_ok, out, torch.zeros_like(out)),
                             alpha=w)
                    weight_sum.add_(row_ok.float() * w)
                row_ok = weight_sum > 0
                acc = torch.where(
                    row_ok,
                    acc / weight_sum.clamp_min(1e-12),
                    torch.full_like(acc, 1.0 / n_classes),
                )
                nan_rows_total += int((~row_ok).sum().item())
                proba[start:end] = acc.cpu().numpy()
        if nan_rows_total > 0:
            warnings.warn(
                f'NRF.predict_proba: {nan_rows_total} row(s) had no finite '
                'tree output; filled with uniform distribution. A tree likely '
                'went NaN during training (check use_transformer=False with '
                'large h_dim under fp16 autocast).',
                RuntimeWarning,
                stacklevel=2,
            )
        return proba

    def predict(self, X):
        return self.le.inverse_transform(np.argmax(self.predict_proba(X), axis=1))

