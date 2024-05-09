# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rho_plus as rp
import seaborn as sns
import torch
import torch.nn.functional as F
from torch import nn

theme, cs = rp.mpl_setup(False)

torch.manual_seed(42)
device = 'cuda'

# %%
from torch.utils.data import DataLoader, Dataset, random_split

batch_size = 32
target = 'yield'


class MyDataset(Dataset):
    def __init__(self, inputs, target):
        self.inputs = inputs.values.astype(np.float32)
        self.labels = target.values.astype(np.float32)

        self.inputs = torch.from_numpy(self.inputs).to(device)
        self.labels = torch.from_numpy(self.labels).to(device)

    @property
    def dim_x(self) -> int:
        return self.inputs.shape[1]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index].unsqueeze(0)


all_data = pd.read_feather('datasets/steels_featurized.feather')

label_cols = ['yield']

input_cols = [c for c in all_data.columns if c not in label_cols]

inputs = all_data
print(inputs[input_cols].shape)

train_dataset = MyDataset(inputs[input_cols], inputs[target])

# use 20% of training data for validation
train_set_size = int(len(train_dataset) * 0.8)
valid_set_size = len(train_dataset) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(42)
train_dataset, valid_dataset = random_split(train_dataset, [train_set_size, valid_set_size], generator=seed)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# %%
import lightning as L


class PandasLogger(L.pytorch.loggers.Logger):
    def __init__(self):
        super().__init__()
        self.rows = []

    def log_hyperparams(self, params, *args, **kwargs):
        pass

    def log_metrics(self, metrics, step=None):
        model_type = 'none'
        row = {'step': step}
        for k in metrics:
            if k.startswith('valid_') and metrics[k] is not None:
                model_type = 'valid'
                row[k.removeprefix('valid_')] = metrics[k]
            elif k.startswith('train_') and metrics[k] is not None:
                model_type = 'train'
                row[k.removeprefix('train_')] = metrics[k]
            elif k.startswith('test_') and metrics[k] is not None:
                model_type = 'test'
                row[k.removeprefix('test_')] = metrics[k]
            else:
                row[k] = metrics[k]

        row['step'] = step
        row['kind'] = model_type
        self.rows.append(row)

    @property
    def version(self):
        return 0

    @property
    def name(self):
        return 'pandas'

    def finalize(self, status):
        self.df = pd.DataFrame(self.rows)


class Cone(torch.nn.Module):
    def forward(self, x):
        return 1 - torch.abs(x - 1)


class Identity(torch.nn.Module):
    def __init__(self, _dim):
        super().__init__()

    def forward(self, x):
        return x


activations = {
    'ReLU': nn.ReLU,
    'SiLU': nn.SiLU,
    'Sigmoid': nn.Sigmoid,
    'Cone': Cone,
    'LeakyReLU': nn.LeakyReLU,
}

norms = {'BatchNorm': nn.BatchNorm1d, 'LayerNorm': nn.LayerNorm, 'Identity': Identity}


class MLP(L.LightningModule):
    #'128-64-16'
    def __init__(
        self,
        input_size,
        out_dim=1,
        dims=(1024, 512),
        lr=1e-4,
        act_name='SiLU',
        norm_name='BatchNorm',
        dropout_rate=0,
        weight_decay=0.03,
        b1=0.9,
        eps=1e-8,
        use_nesterov=True,
    ):
        super().__init__()
        self.dims = dims
        self.lr = lr
        self.weight_decay = weight_decay
        self.b1 = b1
        self.eps = eps
        if use_nesterov:
            self.optim = torch.optim.NAdam
        else:
            self.optim = torch.optim.Adam

        self.layers = []
        i = 0
        prev_dim = input_size
        for dim in (*dims, out_dim):
            fc = nn.Linear(prev_dim, dim)
            bn = norms[norm_name](dim)
            dropout = nn.Dropout(dropout_rate)
            act = activations[act_name]()
            mod = nn.Sequential(fc, bn, dropout, act)
            self.add_module(f'layer{i}', mod)
            self.layers.append(mod)

            prev_dim = dim
            i += 1

        self.out = nn.Linear(dim, 1)  # type: ignore

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)

        return torch.expm1(self.out(out) + 7.5)

    def loss(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs)
        loss = F.l1_loss(output, target, reduction='mean')
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.loss(batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.loss(batch, batch_idx)
        self.log('valid_loss', loss)
        return loss

    def configure_optimizers(self):
        optim = self.optim(
            self.parameters(), lr=self.lr, betas=(self.b1, 0.999), eps=self.eps, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.995)
        return {'optimizer': optim, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}


# %%

n_hidden = 2

act_space = list(activations.keys())
norm_space = list(norms.keys())


def sample_hyperparameters(seed):
    rng = np.random.default_rng(seed)

    return {
        'lr': 10 ** rng.uniform(-4, -2),
        'act_name': rng.choice(act_space),
        'norm_name': rng.choice(norm_space),
        'dropout_rate': rng.uniform(0, 0.5) ** 2 * 2,
        'weight_decay': 10.0 ** rng.uniform(-4, -1.5),
        'eps': 10 ** rng.uniform(-9, -7),
        'b1': rng.uniform(0.8, 0.99),
        'use_nesterov': rng.choice([True, False]),
        'dims': [2 ** rng.integers(6, 12) for _ in range(n_hidden)],
    }


n_trials = 2048

# %%
import logging
import logging.config
from copy import deepcopy
from warnings import filterwarnings

from lightning.pytorch import callbacks
from rich.progress import track

logging.config.dictConfig(
    {
        'version': 1,
        # Other configs ...
        'disable_existing_loggers': True,
    }
)

import os
import sys


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


filterwarnings('ignore', message='.*bottleneck.*')


for name, logger in logging.root.manager.loggerDict.items():
    logger.disabled = True


data = []
for seed in track(np.arange(n_trials) + 512, total=n_trials):
    with HiddenPrints():
        torch.manual_seed(123)
        space = sample_hyperparameters(seed)
        model = MLP(len(input_cols), **space)
        trainer = L.Trainer(
            logger=PandasLogger(),
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            callbacks=[
                # callbacks.LearningRateFinder(min_lr=1e-5, max_lr=1e-2, mode='exponential'),
                callbacks.EarlyStopping(monitor='valid_loss', mode='min', patience=30, strict=True),
            ],
            max_epochs=200,
            log_every_n_steps=2,
            check_val_every_n_epoch=4,
        )
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
        last_epoch = trainer.current_epoch
        best_valid_loss = trainer.logger.df.query('kind == "valid"')['loss'].min()
        best_train_loss = trainer.logger.df.query('kind == "train"')['loss'].min()

        data.append({**space, 'last_epoch': last_epoch, 'best_train': best_train_loss, 'best_valid': best_valid_loss})

        if seed % 10 == 0:
            pd.DataFrame(data).to_csv('mlp_hyperparams_2.csv')

        torch.cuda.empty_cache()

data_df = pd.DataFrame(data)
data_df = pd.concat(
    [
        pd.DataFrame(data_df['dims'].tolist(), columns=[f'hidden_{i+1}' for i in range(n_hidden)]),
        data_df.drop(columns='dims'),
    ],
    axis=1,
)

data_df.to_csv('mlp_hyperparams_2.csv')
