"""Data loading/training pipeline."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Optional, Self

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax import struct
from jaxtyping import Array, Bool, Float, Int, Shaped

from jax_kan.typing_utils import class_tcheck
from jax_kan.utils import debug_structure


def pad_to(new_batch_size: int, *arrs, mask=None):
    outs = []
    for arr in arrs:
        if arr is None:
            continue
        num_pad = -arr.shape[0] % new_batch_size
        if num_pad:
            new_arr = jnp.concat([arr, jnp.tile(arr[[0], ...], [num_pad, *[1 for _ in range(arr.ndim - 1)]])])
        else:
            new_arr = arr
        outs.append(new_arr)

    if mask is None:
        mask = jnp.concat([jnp.ones(arr.shape[0], dtype=jnp.bool), jnp.zeros(num_pad, dtype=jnp.bool)])
    else:
        mask = jnp.concat([mask, jnp.zeros(num_pad, dtype=mask.dtype)])
    outs.append(mask)
    return outs


@class_tcheck
@struct.dataclass
class DataBatch:
    """A (potentially padded/masked) batch of data."""

    # Input.
    X: Shaped[Array, 'batch in_dim']
    # Output.
    y: Shaped[Array, ' batch']
    mask: Bool[Array, ' batch']

    def as_dict(self):
        return {'X': self.X, 'y': self.y, 'mask': self.mask}

    @property
    def size(self) -> int:
        return self.X.shape[0]

    @property
    def in_dim(self) -> int:
        return self.X.shape[1]

    @classmethod
    def new(cls, X: Shaped[Array, 'size in_dim'], y: Shaped[Array, ' size'], batch_size: Optional[int] = None) -> Self:
        """Pads the inputs to the given size if necessary."""
        if batch_size is None:
            return cls(X=X, y=y, mask=jnp.ones(X.shape[0], dtype=jnp.bool))
        else:
            X, y, mask = pad_to(batch_size, X, y)
            return cls(X=X, y=y, mask=mask)

    @classmethod
    def new_empty(cls, batch: int, in_dim: int, X_dtype=jnp.float32, y_dtype=jnp.float32) -> Self:
        return cls(
            X=jnp.zeros((batch, in_dim), dtype=X_dtype),
            y=jnp.zeros((batch,), dtype=y_dtype),
            mask=jnp.ones((batch,), dtype=jnp.bool),
        )

    def reorder(self, order: Sequence[int] | Int[Array, ' batch']) -> Self:
        """Transposes the batch, shuffling the order of inputs."""
        return DataBatch(X=self.X[order], y=self.y[order], mask=self.mask[order])

    def split(self, new_batch_size: int) -> Sequence[Self]:
        """Splits the batch into chunks, padding as necessary to ensure outputs are the same
        shape."""

        X, y, mask = pad_to(new_batch_size, self.X, self.y, mask=self.mask)
        chunks = []
        for i in range(0, X.shape[0], new_batch_size):
            j = i + new_batch_size
            chunks.append(DataBatch(X=X[i:j], y=y[i:j], mask=mask[i:j]))

        return tuple(chunks)

    def masked_mean(self, vals):
        mask_w = self.mask.astype(float)
        mask_w = mask_w / jnp.sum(mask_w)
        return jnp.dot(vals, mask_w)


class AbstractDataLoader:
    """Abstract data loader. Iterates over input batches."""

    @property
    def num_batches(self) -> int:
        """Gets number of batches per epoch."""
        raise NotImplementedError

    def sample_batch(self) -> DataBatch:
        """Returns a pytree of the same shape as a batch."""
        raise NotImplementedError

    def epoch_batches(self) -> Iterable[DataBatch]:
        """Returns a container of batches for the next epoch."""
        raise NotImplementedError


@dataclass
class DataFrameDataLoader(AbstractDataLoader):
    """Loads data from a DataFrame."""

    df: pd.DataFrame
    batch_size: int
    target_col: str
    shuffle_seed: Optional[int] = 42
    standardize: bool = True
    exclude_cols: Sequence[str] = ()

    def __post_init__(self):
        if self.shuffle_seed is not None:
            self.rng = np.random.default_rng(self.shuffle_seed)

        X_df = self.df.drop(columns=list(self.exclude_cols))
        y = jnp.array(X_df.pop(self.target_col).values)
        X = jnp.array(X_df.values)
        X = jax.nn.standardize(X, axis=0) if self.standardize else X

        self.dataset = DataBatch.new(X, y, self.batch_size)

    @property
    def num_batches(self) -> int:
        return self.dataset.size // self.batch_size

    def sample_batch(self) -> DataBatch:
        return DataBatch.new_empty(
            self.batch_size, self.dataset.in_dim, X_dtype=self.dataset.X.dtype, y_dtype=self.dataset.y.dtype
        )

    def epoch_batches(self) -> Iterable[DataBatch]:
        if self.shuffle_seed is not None:
            shuffle_perm = jnp.array(self.rng.permutation(self.dataset.size))
            self.dataset = self.dataset.reorder(shuffle_perm)

        return self.dataset.split(self.batch_size)

    def train_valid_split(self, k: int = 5, fold: int = 0) -> tuple[Self, Self]:
        """Splits into a train and validation data loader."""
        fold_ids = np.arange(self.df.shape[0]) % k

        df_valid = self.df[fold_ids == fold]
        df_train = self.df[fold_ids != fold]

        kwargs = {'batch_size': self.batch_size, 'target_col': self.target_col, 'exclude_cols': self.exclude_cols}

        return (
            DataFrameDataLoader(df=df_train, shuffle_seed=self.shuffle_seed, **kwargs),
            DataFrameDataLoader(df=df_valid, shuffle_seed=None, **kwargs),
        )


if __name__ == '__main__':
    from tqdm import tqdm

    df = pd.read_csv('datasets/one-hundred-plants.csv', index_col='id')
    df['Class'] = df['Class'].astype(int)
    dl = DataFrameDataLoader(
        df=df,
        batch_size=8,
        target_col=df.columns[-1],
    )

    samp = dl.sample_batch()

    debug_structure(samp)

    for epoch in range(3):
        for i, _b in enumerate(tqdm(dl.epoch_batches())):
            if i == 1 and epoch == 0:
                debug_structure(samp)
