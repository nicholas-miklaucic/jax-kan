import functools as ft
import time
from collections.abc import Mapping, Sequence
from random import shuffle
from typing import Any, Callable
import optuna

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
from jax_kan.utils import Identity, debug_stat, debug_structure, flax_summary

console = rich.console.Console()

# -------------------------------

target = 'yield_featurized'
loss_norm_fn = jnp.abs
dataset_splits = (3, 4, 5, 6, 7)
batch_size = 16
n_folds = 5
start_frac = 0.8
end_frac = 0.2
nesterov = True
warmup = 10
n_epochs = 200
dtype = jnp.float32
optimize = False

n_coef = 5
node_dropout = 0.3
order = 3
spline_input_map = lambda x: nn.tanh(x * 0.8)
hidden_dim = None
inner_dims = [512, 128, 64]
normalization = Identity
base_act = nn.tanh
weight_decay = 0
base_lr = 6e-3
gamma = 0.997

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


fold_ids = np.arange(df.shape[0]) % 5
valid_inds = np.random.default_rng(seed=43).permutation(fold_ids)


if target == 'bandgap':
    # switch bandgap and delta_e
    df = df[[*df.columns[:-2], df.columns[-1], df.columns[-2]]]


datasets = []
for fold in range(n_folds):
    sub = df[valid_inds == fold]
    Xy = jnp.array(sub.values, dtype=dtype)
    num_pad = -Xy.shape[0] % batch_size
    mask = jnp.concat([jnp.ones(Xy.shape[0]), jnp.zeros(num_pad)]).astype(jnp.bool)
    Xy = jnp.concat([Xy, Xy[:num_pad]])
    datasets.append((Xy, mask))

# print([sum(fold_ids == i) for i in range(n_folds)])

steps_in_valid_epoch = datasets[0][0].shape[0] // batch_size
steps_in_epoch = steps_in_valid_epoch * (n_folds - 1)

# print(datasets[0][0].shape[0])
# print(sum([datasets[i][0].shape[0] for i in range(1, n_folds)]))

@struct.dataclass
class TrainBatch:
    X: Float[Array, 'batch in_dim']
    y: Float[Array, ' batch']
    mask: Bool[Array, ' batch']

    def as_dict(self):
        return {'X': self.X, 'y': self.y, 'mask': self.mask}


def data_loader(split='train', fold=0, infinite=True):
    if split == 'valid':
        data, mas = datasets[fold]

        data = jnp.array(data)
        mas = jnp.array(mas)
    else:
        datas = []
        mass = []
        for i, (data, mas) in enumerate(datasets):
            if i != fold:
                datas.append(jnp.array(data))
                mass.append(jnp.array(mas))
        data = jnp.concatenate(datas)
        mas = jnp.concatenate(mass)

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

# for _i in traclk(list(range(1000))):
#     sample_batch = next(train_dl)


warmup_steps = steps_in_epoch * min(warmup, n_epochs // 4)
# sched = optax.warmup_cosine_decay_schedule(
#     init_value=start_frac * base_lr,
#     peak_value=base_lr,
#     warmup_steps=warmup_steps,
#     decay_steps=steps_in_epoch * n_epochs,
#     end_value=end_frac * base_lr,
# )


class TrainState(train_state.TrainState):
    pass


def create_train_state(model: nn.Module, rng, sched, weight_decay=0, nesterov=True, sample_X=None):
    if sample_X is None:
        sample_X = sample_batch.X
    model_state = model.init(rng, sample_X, training=False)
    params = model_state.pop('params')
    tx = optax.adamw(sched, weight_decay=weight_decay, nesterov=nesterov)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx, **model_state)


steps_per_log = steps_in_epoch


@ft.partial(jax.jit, static_argnames='training')
def apply_model(state: TrainState, batch: TrainBatch, training: bool, dropout_key):
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

        err = loss_norm_fn(jnp.squeeze(yhat, -1) - batch.y) * batch.mask
        return jnp.sum(err) / jnp.sum(batch.mask), (updates, out)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (updates, out)), grad = grad_fn(state.params)
    return grad, loss, updates, out


@jax.jit
def step(state, grad, updates):
    state = state.apply_gradients(grads=grad)
    for k, v in updates.items():
        if k != 'params':
            state = state.replace(**{k: v})
    return state


def train_model(
    model: nn.Module,
    sched=None,
    weight_decay=0,
    nesterov=True,
    show_progress=True,
    show_sub_progress=False,
    fold=0,
    gamma=0.99,
    train_dl=None,
    valid_dl=None,
):
    start_t = time.time()
    param_state, dropout_state = jr.split(jr.key(np.random.randint(0, 1000)), 2)
    state = create_train_state(model, param_state, sched=sched, weight_decay=weight_decay, nesterov=nesterov)
    # console.print(model.__class__.__name__)
    epoch_df = []

    ema_params = state.params

    if train_dl is None:
        train_dl = data_loader(split='train', fold=fold)
    if valid_dl is None:
        valid_dl = data_loader(split='valid', fold=fold)

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
        disable=not show_progress,
    ) as prog:
        epochs = prog.add_task('Training', total=n_epochs)

        for epoch_i in range(n_epochs):
            epoch_bar = prog.add_task(f'Train {epoch_i}...', total=steps_in_epoch, visible=show_sub_progress)
            losses = []
            for _, batch in zip(range(steps_in_epoch), train_dl):
                grad, loss, updates, out = apply_model(state, batch, training=True, dropout_key=dropout_state)
                # debug_stat(grad)
                losses.append(loss)
                state = step(state, grad, updates)
                ema_params = jax.tree_map(lambda x, y: gamma * x + (1 - gamma) * y, ema_params, state.params)
                prog.update(epoch_bar, advance=1)

            train_loss = float(np.mean(losses))

            valid_bar = prog.add_task(f'Valid {epoch_i}...', total=steps_in_valid_epoch, visible=show_sub_progress)
            losses = []
            for _, batch in zip(range(steps_in_valid_epoch), valid_dl):
                grad, loss, updates, out = apply_model(state, batch, training=False, dropout_key=dropout_state)
                losses.append(loss)
                prog.update(valid_bar, advance=1)

            prog.update(epoch_bar, visible=False, completed=True)
            prog.update(valid_bar, visible=False, completed=True)

            valid_loss = float(np.mean(losses))
            epoch_df.append({'train': train_loss, 'valid': valid_loss})

            prog.update(epochs, advance=1, description=f'Train: {train_loss:>8.03f}\tValid: {valid_loss:>8.03f}')

    end_t = time.time()

    duration = end_t - start_t

    return state.replace(params=ema_params), pd.DataFrame(epoch_df), duration


if __name__ == '__main__':
    from rich.table import Table

    from jax_kan.kan import KAN

    import optuna


    def objective(trial: optuna.Trial):
        n_coef = trial.suggest_int('n_coef', 4, 7)
        spline_input_scale = trial.suggest_float('spline_input_scale', 0.75, 0.85)
        # gamma = 1 - trial.suggest_float('gamma', 1e-3, 1e-2)
        gamma = 0.997
        base_lr = trial.suggest_float('base_lr', 2e-3, 6e-3)
        final_scale = trial.suggest_float('final_scale', 1, 5)
        num_layers = trial.suggest_int('num_layers', 2, 4)

        dims = []
        for i in range(num_layers):
            if i == 0:
                dims.append(trial.suggest_int(f'inner_dims_{i+1}', 256, 512, log=True))
            else:
                mult = trial.suggest_float(f'inner_dim_scale_{i+1}', 0.25, 0.6)
                dims.append(int(round(mult * dims[-1])))

        spline_input_map = lambda x, scale=spline_input_scale: jnp.tanh(x * scale)
        

        kwargs = {
            'n_coef': n_coef,
            'inner_dims': dims,
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
            ),
        }

        sched = optax.warmup_cosine_decay_schedule(
            init_value=start_frac * base_lr,
            peak_value=base_lr,
            warmup_steps=warmup_steps,
            decay_steps=steps_in_epoch * n_epochs,
            end_value=end_frac * base_lr,
        )


        table_df = []

        kan = KAN(in_dim=sample_batch.X.shape[-1], out_dim=1, final_act=lambda x, s=final_scale: target_transforms[target](x * s), **kwargs)

        for fold in range(n_folds):
            ema_state, kan_epochs, duration = train_model(
                kan,
                sched=sched,
                weight_decay=weight_decay,
                nesterov=nesterov,
                fold=fold,
                gamma=gamma
            )

            losses = []
            for batch in data_loader(fold=fold, split='valid', infinite=False):
                grad, loss, updates, out = apply_model(ema_state, batch, training=False, dropout_key=jr.key(0))
                losses.append(loss)

            best_valid = np.mean(losses).item()
            console.print(f'EMA: {best_valid:.3f}')

            losses = []
            for batch in data_loader(fold=fold, split='train', infinite=False):
                grad, loss, updates, out = apply_model(ema_state, batch, training=False, dropout_key=jr.key(0))
                losses.append(loss)

            best_train = np.mean(losses).item()
            console.print(f'EMA (Training): {best_train:.3f}')

            best_train = kan_epochs["train"].min()
            # best_valid = kan_epochs["valid"].min()

            table_df.append({'Duration': duration, 'Training': best_train, 'Validation': best_valid, 'Fold': fold})


            trial.report(best_valid, fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return pd.DataFrame(table_df).mean()['Validation']

    from rich.logging import RichHandler


    
    if optimize:
        optuna.logging.get_logger("optuna").addHandler(RichHandler(console=console))
        study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=50)
    else:
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
            ),
        }

        sched = optax.warmup_cosine_decay_schedule(
            init_value=start_frac * base_lr,
            peak_value=base_lr,
            warmup_steps=warmup_steps,
            decay_steps=steps_in_epoch * n_epochs,
            end_value=end_frac * base_lr,
        )

        kan = KAN(in_dim=sample_batch.X.shape[-1], out_dim=1, final_act=target_transforms[target], **kwargs)

        sample_out, params = kan.init_with_output(jr.key(0), sample_batch.X)

        print(steps_in_epoch)        

        # debug_stat(jnp.abs(sample_out.squeeze() - sample_batch.y))
        # debug_structure(sample_out)
        # debug_structure(params)

        flax_summary(kan, x=sample_batch.X, compute_flops=True, compute_vjp_flops=True)        

        table_df = []
        table = Table(title='Run')

        table.add_column('Duration', justify='right', style='cyan')
        table.add_column('Best Training', justify='right', style='magenta')
        table.add_column('Best Validation', justify='right', style='green')


        for fold in range(n_folds):
            kan = KAN(in_dim=sample_batch.X.shape[-1], out_dim=1, final_act=target_transforms[target], **kwargs)
            ema_state, kan_epochs, duration = train_model(
                kan,
                sched=sched,
                weight_decay=weight_decay,
                nesterov=nesterov,
                fold=fold,
                gamma=gamma
            )

            # ema_params = jax.tree_map(lambda *xs: jnp.mean(ema_gamma * xs, axis=0), *[state.params for state in kan_states])

            losses = []
            for batch in data_loader(fold=fold, split='valid', infinite=False):
                grad, loss, updates, out = apply_model(ema_state, batch, training=False, dropout_key=jr.key(0))
                losses.append(loss)

            best_valid = np.mean(losses).item()
            console.print(f'EMA: {best_valid:.3f}')

            losses = []
            for batch in data_loader(fold=fold, split='train', infinite=False):
                grad, loss, updates, out = apply_model(ema_state, batch, training=False, dropout_key=jr.key(0))
                losses.append(loss)

            best_train = np.mean(losses).item()
            console.print(f'EMA (Training): {best_train:.3f}')

            # debug_stat(ema_state.params)

            # best_train = kan_epochs["train"].min()
            # best_valid = kan_epochs["valid"].min()

            table.add_row(f'{duration:.3f}s', f'{best_train:.3f}', f'{best_valid:.3f}')
            table_df.append({'Duration': duration, 'Training': best_train, 'Validation': best_valid, 'Fold': fold})


        console.print(table)

        console.print(pd.DataFrame(table_df).mean())
