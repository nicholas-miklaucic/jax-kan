import functools as ft
import pickle
import time
from collections.abc import Mapping, Sequence
from random import shuffle
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import optuna
import pandas as pd
import rich
import rich.progress as rprog
from eins import EinsOp
from eins import ElementwiseOps as E
from flax import struct
from flax.training import train_state
from jaxtyping import Array, Bool, Float
from rich.progress import Progress, track
from rich.progress_bar import ProgressBar

from jax_kan.data import DataFrameDataLoader
from jax_kan.kan import KAN, KANLayer
from jax_kan.trainer import Trainer
from jax_kan.utils import Identity, debug_stat, debug_structure, flax_summary

console = rich.console.Console()

# -------------------------------

target = 'yield_featurized'
loss_norm_fn = jnp.abs
dataset_splits = (3, 4, 5, 6, 7)
batch_size = 16
use_prodigy = False
n_folds = 5
start_frac = 0.8
end_frac = 0.2
nesterov = True
warmup = 10
n_epochs = 300
dtype = jnp.float32
optimize = False

n_coef = 4
node_dropout = 0
order = 3
spline_input_map = lambda x: nn.tanh(x * 0.8)
hidden_dim = None
inner_dims = [32, 32]
normalization = Identity
base_act = nn.tanh
weight_decay = 0.03
base_lr = 4e-3
gamma = 0.99
alpha = None

# -------------------------------

target_transforms = {
    'bandgap': lambda x: x,
    'yield_raw': E.from_func(lambda x: E.expm1(x + 7.5)),
    'delta_e': lambda x: x * 4,
}

target_transforms['yield_featurized'] = target_transforms['yield_raw']

if target in ('bandgap', 'delta_e'):
    df = pd.read_feather('datasets/mpc_full_feats_scaled_split.feather')
    df = df[df['dataset_split'].isin(dataset_splits)]
    df = df.select_dtypes('number').drop(columns=['TSNE_x', 'TSNE_y', 'umap_x', 'umap_y', 'dataset_split'])
elif target == 'yield_raw':
    df = pd.read_feather('datasets/steels_raw.feather')
elif target == 'yield_featurized':
    df = pd.read_feather('datasets/steels_featurized.feather')
else:
    msg = f'Unknown target: {target}'
    raise ValueError(msg)


dl = DataFrameDataLoader(df, batch_size=batch_size, target_col=df.columns[-1])

steps_in_epoch = 4 * dl.num_batches // 5

kwargs = {
    'n_coef': n_coef,
    'inner_dims': inner_dims,
    'normalization': normalization,
    'hidden_dim': hidden_dim,
    'out_hidden_dim': 1,
    'layer_templ': KANLayer(
        1,
        1,
        order=order,
        dropout_rate=node_dropout,
        base_act=base_act,
        spline_input_map=spline_input_map,
        alpha=alpha,
    ),
}

sched = optax.cosine_onecycle_schedule(
    transition_steps=steps_in_epoch * n_epochs,
    peak_value=1 if use_prodigy else base_lr,
    pct_start=0.1,
    div_factor=1 / start_frac,
    final_div_factor=1 / end_frac,
)
if use_prodigy:
    opt = optax.contrib.prodigy(sched, weight_decay=weight_decay)
else:
    opt = optax.adamw(sched, weight_decay=weight_decay, nesterov=nesterov)
tx = optax.chain(
    opt,
    optax.clip_by_global_norm(max_norm=3.0),
)


kan = KAN(in_dim=dl.sample_batch().in_dim, out_dim=1, final_act=target_transforms[target], **kwargs)

trainer = Trainer(kan, dl, optimizer=tx)
