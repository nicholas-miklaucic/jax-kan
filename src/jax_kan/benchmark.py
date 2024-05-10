from collections.abc import Sequence
from typing import Any, Callable, Optional

import numpy as np
from jax_kan.kan import KAN, KANLayer
import optax
import flax.linen as nn
from jax_kan.train import TrainBatch, apply_model, create_train_state, step, df, target_transforms, train_model
from jax_kan.utils import Identity, flax_summary
import jax
import jax.random as jr
import jax.numpy as jnp
from dataclasses import dataclass
import pandas as pd
import matminer
from pymatgen.core import Composition
from jax_kan.train import n_coef, node_dropout, order, spline_input_map, hidden_dim, inner_dims, normalization, base_act, weight_decay, base_lr, gamma, nesterov, start_frac, end_frac, batch_size, n_epochs, warmup

from matminer.featurizers.base import MultipleFeaturizer
import matminer.featurizers.composition as cf

preset = cf.ElementProperty.from_preset('magpie')
megnet = cf.ElementProperty.from_preset('megnet_el')

feats = MultipleFeaturizer([
    cf.ElementProperty(preset.data_source, preset.features, stats=['mean', 'avg_dev', 'minimum', 'maximum']),
    cf.ElementProperty(megnet.data_source, megnet.features, stats=['mean', 'avg_dev', 'minimum', 'maximum']),
    cf.Miedema(),
    cf.WenAlloys()   
])

columns = df.columns[:-1]


def mb_data_loader(X_data, y_data, batch_size, infinite=True, shuffle=True, emulate_artifacts=True):
    comps: list[Composition] = [Composition(x) for x in X_data]

    feat_df = pd.DataFrame(feats.transform(comps), columns=feats.feature_labels()).select_dtypes('number')
    feat_z = (feat_df - feat_df.mean()) / (feat_df.std() + 1e-12)
    X = feat_z[columns]
    X = jnp.array(X.values).clip(-4, 4)

    y = jnp.array(y_data)[:, None]

    Xy = jnp.concat([X, y], axis=1)
    if emulate_artifacts:
        datasets = []
        masks = []
        fold_i = jnp.arange(Xy.shape[0]) % 4
        for fold in range(4):
            sub = Xy[fold_i == fold]            
            num_pad = -sub.shape[0] % batch_size
            mask = jnp.concat([jnp.ones(sub.shape[0]), jnp.zeros(num_pad)]).astype(jnp.bool)
            sub = jnp.concat([sub, sub[:num_pad]])
            datasets.append(sub)
            masks.append(mask)
        
        Xy = jnp.concat(datasets)
        mask = jnp.concat(masks)
    else:        
        num_pad = -Xy.shape[0] % batch_size
        mask = jnp.concat([jnp.ones(Xy.shape[0]), jnp.zeros(num_pad)]).astype(jnp.bool)
        Xy = jnp.concat([Xy, Xy[:num_pad]])
    
    data = Xy    
    mas = mask     

    n_splits = data.shape[0] // batch_size

    yield data, data[:, :-1].shape, n_splits
    data_batches = jnp.split(data, n_splits)
    mas_batches = jnp.split(mas, n_splits)

    batches = [TrainBatch(X=d[..., :-1], y=d[..., -1], mask=m) for d, m in zip(data_batches, mas_batches)]

    first_time = True
    while first_time or infinite:
        first_time = False

        if shuffle:
            np.random.shuffle(batches)

        yield from batches


@dataclass
class KANModel:
    X: Sequence[str]
    y: Sequence[float]
    n_coef: int = n_coef
    node_dropout: float = node_dropout
    spline_input_map: Callable = spline_input_map    
    hidden_dim: Optional[int] = hidden_dim
    inner_dims: Sequence[int] = tuple(inner_dims)
    normalization: Any = normalization    
    weight_decay: float = weight_decay
    base_lr: float = base_lr    
    gamma: float = gamma
    batch_size: int = batch_size
    start_frac: float = start_frac
    end_frac: float = end_frac
    nesterov: bool = nesterov
    warmup: int = warmup
    n_epochs: int = n_epochs

    def __post_init__(self):        
        self.dl = mb_data_loader(self.X, self.y, self.batch_size)
        data, (n_data, n_feats), self.steps_in_epoch = next(self.dl)

        print(self.steps_in_epoch)

        # self.steps_in_epoch = 12

        warmup_steps = self.warmup * self.steps_in_epoch
        self.sched = optax.warmup_cosine_decay_schedule(
            init_value=self.start_frac * self.base_lr,
            peak_value=self.base_lr,
            warmup_steps=warmup_steps,
            decay_steps=self.steps_in_epoch * self.n_epochs,
            end_value=self.end_frac * self.base_lr,
        )    
        self.layer_templ = KANLayer(
            1,
            1,            
            dropout_rate=self.node_dropout,            
            spline_input_map=lambda x: nn.tanh(x * 0.8),
        )

        self.kan = KAN(
            in_dim=n_feats, out_dim=1, final_act=target_transforms['yield_featurized'],
            n_coef=self.n_coef, hidden_dim=self.hidden_dim, normalization=self.normalization,
            out_hidden_dim=1, layer_templ=self.layer_templ, inner_dims=self.inner_dims
        )

        self.sample_X = jnp.ones((self.batch_size, n_feats))

    def fit(self):
        param_state, dropout_state = jr.split(jr.key(np.random.randint(0, 1000)), 2)        
        state = create_train_state(self.kan, param_state, sched=self.sched, 
                                   weight_decay=self.weight_decay, nesterov=self.nesterov,
                                   sample_X=self.sample_X)        
                
        ema_params = state.params
        for _epoch_i in range(self.n_epochs):
            losses = []
            for _i, batch in zip(range(self.steps_in_epoch), self.dl):
                grad, loss, updates, out = apply_model(state, batch, training=True, dropout_key=dropout_state)                
                losses.append(loss)
                state = step(state, grad, updates)
                ema_params = jax.tree_map(lambda x, y: self.gamma * x + (1 - self.gamma) * y, 
                                          ema_params, state.params)  
            # print(jnp.mean(jnp.array(losses)))

        self.ema_state = state.replace(params=ema_params)

    def fit_alt(self):
        ema_state, kan_epochs, duration = train_model(
            self.kan,
            sched=self.sched,
            weight_decay=self.weight_decay,
            nesterov=self.nesterov,            
            gamma=self.gamma,
            train_dl=self.dl
        )
        self.ema_state = ema_state
    
    def predict(self, X_data, y_data=None):
        if y_data is None:
            y_data = jnp.ones(X_data.shape[0])
        test_dl = mb_data_loader(X_data, y_data, X_data.shape[0], infinite=False, shuffle=False, emulate_artifacts=False)
        data, (n_data, n_feats), steps_in_epoch = next(test_dl)
        batch = next(test_dl)
        grad, loss, updates, out = apply_model(self.ema_state, batch, training=False, dropout_key=jr.key(0))
        
        yhat = np.array(out).reshape(-1)        
        yhat = yhat[batch.mask]

        return yhat


if __name__ == '__main__':
    import random
    import numpy as np
    # import torch
    # torch.manual_seed(1234)
    # torch.cuda.manual_seed(1234)
    # torch.cuda.manual_seed_all(1234)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(1234)
    random.seed(1234)



    from matbench.bench import MatbenchBenchmark
    mb = MatbenchBenchmark(autoload=False, subset=["matbench_steels"])
    for task in mb.tasks:
        task.load()
        for fold in task.folds:
            print(f'fold no: {fold}')
            train_inputs, train_outputs = task.get_train_and_val_data(fold)
            test_inputs = task.get_test_data(fold, include_target=False)
            model = KANModel(train_inputs, train_outputs) 

            # debug_stat(jnp.abs(sample_out.squeeze() - sample_batch.y))
            # debug_structure(sample_out)
            # debug_structure(params)            

            # dl = mb_data_loader(train_inputs, train_outputs, 30)
            # data, data_sh, sie = next(dl) 
            # print(data.mean(), data.std())
            # print(df.values.mean(), df.values.std())            

            model.fit_alt()
            train_hat = model.predict(train_inputs, train_outputs)
            print(jnp.mean(jnp.abs(train_hat.reshape(-1) - train_outputs.values.reshape(-1))))
            predictions = model.predict(test_inputs)
            task.record(fold, predictions)

    mb.to_file("results.json.gz")