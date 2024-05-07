import functools as ft
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

from jax_kan.kan import KANLayer
from jax_kan.utils import Identity, debug_structure, flax_summary

console = rich.console.Console()

# -------------------------------

target = 'yield_featurized'
loss_norm_fn = jnp.abs
dataset_splits = (3, 4, 5, 6, 7)
batch_size = 16
valid_prop = 0.2
start_frac = 0.1
end_frac = 0.2
base_lr = 5e-3
weight_decay = 0.03
nesterov = True
warmup = 10
n_epochs = 500
dtype = jnp.float32

kwargs = {
    'n_grid': 5,
    'inner_dims': [128, 128],
    'normalization': nn.LayerNorm,
    'hidden_dim': None,
    'out_hidden_dim': 1,
    'train_knots': False,
    'layer_templ': KANLayer(1, 1, order=2, dropout_rate=0.1, base_act=lambda x: jnp.tanh(x / 2)),
}


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

valid_size = int(round(df.shape[0] * valid_prop / batch_size)) * batch_size
valid_inds = np.random.default_rng(seed=123).choice(df.index, valid_size, replace=False)
is_valid = df.index.isin(valid_inds)


if target == 'bandgap':
    # switch bandgap and delta_e
    df = df[[*df.columns[:-2], df.columns[-1], df.columns[-2]]]


datasets = []
for sub in df[is_valid], df[~is_valid]:
    Xy = jnp.array(sub.values, dtype=dtype)
    num_pad = -Xy.shape[0] % batch_size
    mask = jnp.concat([jnp.ones(Xy.shape[0]), jnp.zeros(num_pad)]).astype(jnp.bool)
    Xy = jnp.concat([Xy, Xy[:num_pad]])
    datasets.append((Xy, mask))

datasets = {'train': datasets[1], 'valid': datasets[0]}

steps_in_epoch = datasets['train'][0].shape[0] // batch_size
steps_in_valid_epoch = datasets['valid'][0].shape[0] // batch_size


@struct.dataclass
class TrainBatch:
    X: Float[Array, 'batch in_dim']
    y: Float[Array, ' batch']
    mask: Bool[Array, ' batch']

    def as_dict(self):
        return {'X': self.X, 'y': self.y, 'mask': self.mask}


def data_loader(split='train', infinite=True):
    data, mas = datasets[split]

    data = jnp.array(data)
    mas = jnp.array(mas)

    n_splits = data.shape[0] // batch_size
    data_batches = jnp.split(data, n_splits)
    mas_batches = jnp.split(mas, n_splits)

    batches = [TrainBatch(X=d[..., :-1], y=d[..., -1], mask=m) for d, m in zip(data_batches, mas_batches)]

    first_time = True
    while first_time or infinite:
        first_time = False

        shuffle(batches)

        yield from batches


sample_batch = next(data_loader())
# debug_structure(sample_batch)


train_dl = data_loader('valid')

# for _i in traclk(list(range(1000))):
#     sample_batch = next(train_dl)


warmup_steps = steps_in_epoch * min(warmup, n_epochs // 4)
sched = optax.warmup_cosine_decay_schedule(
    init_value=start_frac * base_lr,
    peak_value=base_lr,
    warmup_steps=warmup_steps,
    decay_steps=steps_in_epoch * n_epochs,
    end_value=end_frac * base_lr,
)


class TrainState(train_state.TrainState):
    pass


def create_train_state(model: nn.Module, rng):
    model_state = model.init(rng, jnp.ones((sample_batch.X.shape[1],)), training=False)
    params = model_state.pop('params')
    tx = optax.adamw(sched, weight_decay=weight_decay, nesterov=nesterov)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx, **model_state)


steps_per_log = steps_in_epoch


@ft.partial(jax.jit, static_argnames='training')
def apply_model(state: TrainState, batch: Mapping[str, jax.Array], training: bool, dropout_key):
    dropout_train_key = jr.fold_in(key=dropout_key, data=state.step)

    def loss_fn(params):
        out = state.apply_fn(
            {'params': params},
            batch['X'],
            training=training,
            rngs={'dropout': dropout_train_key},
            mutable=training,
        )

        if training:
            yhat, updates = out
        else:
            yhat = out
            updates = {}

        err = loss_norm_fn(jnp.squeeze(yhat) - batch['y']) * batch['mask']
        return jnp.mean(err), updates

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, updates), grad = grad_fn(state.params)
    return grad, loss, updates


def step(state, grad, updates):
    state = state.apply_gradients(grads=grad)
    state = state.replace(**updates)
    return state


def train_model(model: nn.Module):
    start_t = time.time()
    param_state, dropout_state = jr.split(jr.key(np.random.randint(0, 1000)), 2)
    state = create_train_state(model, param_state)
    console.print(model.__class__.__name__)
    epoch_df = []

    train_dl = data_loader()
    valid_dl = data_loader(split='valid')

    with Progress(
        rprog.TextColumn('[progress.description]{task.description}'),
        rprog.BarColumn(120, 'light_pink3', 'deep_sky_blue4', 'green'),
        rprog.MofNCompleteColumn(),
        rprog.TimeElapsedColumn(),
        rprog.TimeRemainingColumn(),
        rprog.SpinnerColumn(),
        refresh_per_second=3,
        console=console,
        expand=True,
    ) as prog:
        epochs = prog.add_task('Training', total=n_epochs)

        for epoch_i in range(n_epochs):
            epoch_bar = prog.add_task(f'Train {epoch_i}...', total=steps_in_epoch)
            losses = []
            for _, batch in zip(range(steps_in_epoch), train_dl):
                grad, loss, updates = jax.vmap(
                    lambda bat, state: apply_model(state, bat, training=True, dropout_key=dropout_state),
                    in_axes=(
                        {'X': 0, 'y': 0, 'mask': 0},
                        None,
                    ),
                )(batch.as_dict(), state)
                losses.append(loss)
                state = step(state, grad, updates)
                prog.update(epoch_bar, advance=1)

            train_loss = np.mean(losses)

            valid_bar = prog.add_task(f'Valid {epoch_i}...', total=steps_in_valid_epoch)
            losses = []
            for _, batch in zip(range(steps_in_valid_epoch), valid_dl):
                grad, loss, updates = apply_model(state, batch, training=False, dropout_key=dropout_state)
                losses.append(loss)
                prog.update(valid_bar, advance=1)

            prog.update(epoch_bar, visible=False, completed=True)
            prog.update(valid_bar, visible=False, completed=True)

            valid_loss = np.mean(losses)
            epoch_df.append({'train': train_loss, 'valid': valid_loss})

            prog.update(epochs, advance=1, description=f'Train: {train_loss:.03f}\tValid: {valid_loss:.03f}')

    end_t = time.time()

    duration = end_t - start_t

    return state, pd.DataFrame(epoch_df), duration


if __name__ == '__main__':
    from rich.table import Table

    from jax_kan.kan import KAN

    kan = KAN(in_dim=sample_batch.X.shape[-1], out_dim=1, final_act=target_transforms[target], **kwargs)

    flax_summary(kan, x=sample_batch.X[0], compute_flops=True, compute_vjp_flops=True)

    # mlp_state, mlp_epochs = train_model(mlp)
    kan_state, kan_epochs, duration = train_model(kan)

    table = Table(title='Run')

    table.add_column('Duration', justify='right', style='cyan')
    table.add_column('Best Training', justify='right', style='magenta')
    table.add_column('Best Validation', justify='right', style='green')

    table.add_row(f'{duration:.3f}s', f'{kan_epochs["train"].min():.3f}', f'{kan_epochs["valid"].min():.3f}')
    console.print(table)
