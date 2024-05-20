"""Trainer interface."""

import functools as ft
import pickle
import time
from collections import defaultdict
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
from flax import linen as nn
from flax import struct
from flax.training import train_state
from jaxtyping import Array, Bool, Float
from rich.progress import Progress, track
from rich.progress_bar import ProgressBar
from sklearn.metrics import roc_auc_score

from jax_kan.data import AbstractDataLoader, DataBatch, DataFrameDataLoader
from jax_kan.function_basis import FixedInputMap, InputMap
from jax_kan.kan import KAN, KANLayer
from jax_kan.utils import Identity, debug_stat, debug_structure, flax_summary


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
        show_progress: bool = True,
        show_subprogress: bool = False,
        coef_ab=(1, 1),
    ):
        self.model = model
        self.data = data
        self.optimizer = optimizer
        self.show_progress = show_progress
        self.show_subprogress = show_subprogress
        self.coef_ab = coef_ab

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

        def apply_model(state: TrainState, batch: DataBatch, training: bool, dropout_key, pct_done):
            dropout_train_key = jr.fold_in(key=dropout_key, data=state.step)

            n_eff_coefs = 1 - (1 - pct_done ** self.coef_ab[0]) ** self.coef_ab[1]

            def loss_fn(params):
                out = state.apply_fn(
                    {'params': params},
                    batch.X,
                    training=training,
                    n_eff_coefs=n_eff_coefs,
                    rngs={'dropout': dropout_train_key},
                    mutable=training,
                )

                if training:
                    yhat, updates = out
                else:
                    yhat = out
                    updates = {}

                # debug_stat(yhat=yhat, y=batch.y)

                err = optax.softmax_cross_entropy_with_integer_labels(yhat, batch.y) * batch.mask
                return batch.masked_mean(err), (updates, yhat)

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
                values = defaultdict(list)
                for batch in self.train_data.epoch_batches():
                    grad, loss, updates, out = self.apply_model(
                        self.state,
                        batch,
                        training=True,
                        dropout_key=dropout_state,
                        pct_done=epoch_i / (n_epochs - 1),
                    )

                    # debug_stat(grad)

                    values['train_loss'].append(loss)
                    values['train_acc'].append(batch.masked_mean(jnp.argmax(out, -1) == batch.y))
                    values['grad_norm'].append(optax.global_norm(grad))
                    self.state = self.take_step(self.state, grad, updates)
                    prog.update(epoch_bar, advance=1)

                valid_bar = prog.add_task(
                    f'Valid {epoch_i}...', total=steps_in_valid_epoch, visible=self.show_subprogress
                )
                for batch in self.valid_data.epoch_batches():
                    grad, loss, updates, out = self.apply_model(
                        self.state, batch, training=False, dropout_key=dropout_state, pct_done=epoch_i / (n_epochs - 1)
                    )
                    values['valid_loss'].append(loss)
                    values['valid_acc'].append(batch.masked_mean(jnp.argmax(out, -1) == batch.y))
                    prog.update(valid_bar, advance=1)

                prog.update(epoch_bar, visible=False, completed=True)
                prog.update(valid_bar, visible=False, completed=True)

                aggfuncs = {'grad_norm': np.max}
                epoch_row = {k: float(aggfuncs.get(k, np.mean)(v)) for k, v in values.items()}
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
    # jax.config.update('jax_debug_nans', True)
    batch_size = 16
    n_folds = 5
    start_frac = 0.8
    end_frac = 0.2
    warmup = 10
    n_epochs = 300
    dtype = jnp.float32

    n_coef = 4
    node_dropout = 0
    hidden_dim = None
    inner_dims = [32, 32]
    normalization = Identity
    base_act = nn.tanh
    weight_decay = 0.03
    base_lr = 4e-3
    input_map = FixedInputMap(
        stretch_base=1,
        stretch_trainable=False,
        map_type='tanh',
    )

    # -------------------------------

    df = pd.read_csv('datasets/one-hundred-plants.csv', index_col='id')
    df['Class'] = df['Class'].astype(int)
    dl = DataFrameDataLoader(df, batch_size=batch_size, target_col='Class')

    steps_in_epoch = 4 * dl.num_batches // 5

    kwargs = {
        'in_dim': dl.sample_batch().in_dim,
        'out_dim': max(dl.dataset.y.tolist()) + 1,
        'final_act': lambda x: x,
        'inner_dims': inner_dims,
        'normalization': normalization,
        'hidden_dim': hidden_dim,
        'out_hidden_dim': None,
        'layer_templ': KANLayer(1, 1, n_coef=n_coef, dropout_rate=node_dropout, base_act=base_act, input_map=input_map),
    }

    sched = optax.cosine_onecycle_schedule(
        transition_steps=steps_in_epoch * n_epochs,
        peak_value=base_lr,
        pct_start=0.1,
        div_factor=1 / start_frac,
        final_div_factor=1 / end_frac,
    )
    opt = optax.nadamw(sched, weight_decay=weight_decay)
    tx = optax.chain(opt)

    kan = KAN(**kwargs)

    trainer = Trainer(kan, dl, optimizer=tx)

    trainer.fit(n_epochs=n_epochs)

    print(trainer.epoch_df.iloc[-1].to_string())
