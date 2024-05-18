"""Trainer interface."""

from typing import Callable
from flax import linen as nn
from flax.training import train_state
import functools as ft
import jax
import optax
import jax.random as jr
import jax.numpy as jnp
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

from jax_kan.kan import KAN, KANLayer
from jax_kan.utils import Identity, debug_stat, debug_structure, flax_summary

from jax_kan.data import AbstractDataLoader, DataBatch, DataFrameDataLoader


class TrainState(train_state.TrainState):
    pass


class Trainer:
    """A trainer that wraps a model, dataset, and optimizer."""

    def __init__(
        self,
        model: nn.Module,
        data: DataFrameDataLoader,
        optimizer: optax.GradientTransformation,
        rng=None,
        loss_fn: Callable = lambda x, y: jnp.mean(jnp.abs(x - y)),
        show_progress: bool = True,
        show_subprogress: bool = False,
    ):
        self.model = model
        self.data = data
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.show_progress = show_progress
        self.show_subprogress = show_subprogress

        if rng is None:
            self.rng = jr.key(np.random.randint(0, 2048))
        else:
            self.rng = rng

        self.use_fold(0)

    def init_clean(self):
        self.epoch = 0

        model_state = self.model.init(self.rng, self.data.sample_batch().X, training=False)
        params = model_state.pop('params')

        self.state = TrainState.create(apply_fn=self.model.apply, params=params, tx=self.optimizer, **model_state)

        def apply_model(state: TrainState, batch: DataBatch, training: bool, dropout_key):
            dropout_train_key = jr.fold_in(key=dropout_key, data=state.step)

            def loss_fn(params):
                out = state.apply_fn(
                    {'params': params},
                    batch.X,
                    training=training,
                    rngs={'dropout': dropout_train_key},
                    mutable=training,
                )

                if training:
                    yhat, updates = out
                else:
                    yhat = out
                    updates = {}

                err = self.loss_fn(jnp.squeeze(yhat, -1), batch.y) * batch.mask
                return jnp.sum(err) / jnp.sum(batch.mask), (updates, out)

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, (updates, out)), grad = grad_fn(state.params)
            return grad, loss, updates, out

        def step(state: TrainState, grad, updates):
            state = state.apply_gradients(grads=grad)
            for k, v in updates.items():
                if k != 'params':
                    state = state.replace(**{k: v})
            return state

        self.apply_model = jax.jit(apply_model, static_argnames='training')
        self.take_step = jax.jit(step)

    def use_fold(self, fold: int, k: int = 5):
        self.train_data, self.valid_data = self.data.train_valid_split(k=k, fold=fold)
        self.init_clean()

    def fit(self, n_epochs: int):
        start_t = time.time()
        dropout_state = jr.fold_in(self.rng, 123)
        epoch_df = []

        steps_in_epoch = self.train_data.num_batches
        steps_in_valid_epoch = self.valid_data.num_batches

        with Progress(
            rprog.TextColumn('[progress.description]{task.description}'),
            rprog.BarColumn(120, 'light_pink3', 'deep_sky_blue4', 'green'),
            rprog.MofNCompleteColumn(),
            rprog.TimeElapsedColumn(),
            rprog.TimeRemainingColumn(),
            rprog.SpinnerColumn(),
            refresh_per_second=3,
            expand=True,
            disable=not self.show_progress,
        ) as prog:
            epochs = prog.add_task('Training', total=n_epochs)

            for epoch_i in range(n_epochs):
                epoch_bar = prog.add_task(f'Train {epoch_i}...', total=steps_in_epoch, visible=self.show_subprogress)
                values = {'train_loss': [], 'grad_norm': [], 'valid_loss': []}
                for batch in self.train_data.epoch_batches():
                    grad, loss, updates, out = self.apply_model(
                        self.state, batch, training=True, dropout_key=dropout_state
                    )

                    # debug_stat(grad)
                    values['train_loss'].append(loss)
                    values['grad_norm'].append(optax.global_norm(grad))
                    self.state = self.take_step(self.state, grad, updates)
                    prog.update(epoch_bar, advance=1)

                valid_bar = prog.add_task(
                    f'Valid {epoch_i}...', total=steps_in_valid_epoch, visible=self.show_subprogress
                )
                for batch in self.valid_data.epoch_batches():
                    grad, loss, updates, out = self.apply_model(
                        self.state, batch, training=False, dropout_key=dropout_state
                    )
                    values['valid_loss'].append(loss)
                    prog.update(valid_bar, advance=1)

                prog.update(epoch_bar, visible=False, completed=True)
                prog.update(valid_bar, visible=False, completed=True)

                epoch_row = {k: float(np.mean(v)) for k, v in values.items()}
                epoch_row['epoch'] = self.epoch
                epoch_df.append(epoch_row)
                self.epoch += 1

                prog.update(
                    epochs,
                    advance=1,
                    description='Train: {:>8.03f}\tValid: {:>8.03f}'.format(
                        epoch_row['train_loss'], epoch_row['valid_loss']
                    ),
                )

        end_t = time.time()

        duration = end_t - start_t

        self.epoch_df = pd.DataFrame(epoch_df)

        return duration


if __name__ == '__main__':
    target = 'expt_gap'
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
        'expt_gap': nn.elu,
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
    elif target == 'expt_gap':
        df = pd.read_feather('datasets/mb_expt_gap.feather')
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

    trainer.fit(n_epochs=n_epochs)

    print(trainer.epoch_df.iloc[-1].to_string())
