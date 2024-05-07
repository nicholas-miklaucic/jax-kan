"""Flax modules defining the KAN and a single KAN layer."""

import functools as ft
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Callable, Optional, Self

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from eins import EinsOp
from eins import Reductions as R
from flax import linen as nn
from jaxtyping import Array, Float

from jax_kan.spline import BSpline
from jax_kan.typing_utils import class_tcheck, tcheck
from jax_kan.utils import flax_summary


@class_tcheck
class KANLayer(nn.Module):
    in_dim: int
    out_dim: int
    order: int = 2
    dropout_rate: float = 0.0
    kernel_init: Callable = nn.initializers.normal(stddev=1)
    resid_scale_trainable: bool = True
    resid_scale_init: Callable = nn.initializers.ones
    spline_scale_trainable: bool = False
    spline_scale_init: Callable = nn.initializers.zeros
    base_act: Callable = nn.tanh
    spline_input_map: Callable = jnp.tanh
    knots: Float[Array, ' grid'] = field(default_factory=lambda: jnp.array([-1, -0.5, 0, 0.5, 1]))

    def setup(self):
        self.size = self.in_dim * self.out_dim
        self.grid = self.knots

        self.spline = BSpline(self.grid, self.order)
        self.coefs = nn.Einsum((self.in_dim, self.out_dim, self.spline.n_coefs), 'ic,ioc->io', use_bias=False)

        if self.resid_scale_trainable:
            self.resid_scale = self.param('resid_scale', self.resid_scale_init, (self.in_dim, self.out_dim))
        else:
            self.resid_scale = 1

        if self.spline_scale_trainable:
            self.spline_scale = self.param('spline_scale', self.spline_scale_init, (self.in_dim, self.out_dim))
        else:
            self.spline_scale = 1

        self.dropout = nn.Dropout(self.dropout_rate)

    def __call__(self: Self, x: Float[Array, 'in_dim'], training: bool = False) -> Float[Array, ' out_dim']:
        dm = jax.vmap(self.spline.design_matrix)(self.spline_input_map(x))
        # in coefs
        y = self.coefs(dm)
        # in_dim, out_dim
        y = self.resid_scale * self.base_act(x[..., None]) + self.spline_scale * y
        y = self.dropout(y, deterministic=not training)
        # y = EinsOp('in out -> out', reduce='mean')(y)
        y = R.mean(y, axis=0)

        return y


@class_tcheck
class KAN(nn.Module):
    in_dim: int
    out_dim: int
    inner_dims: Sequence[int]

    n_grid: int = 5
    knot_dtype: jnp.dtype = jnp.float32
    train_knots: bool = True
    layer_dropout_rate: float = 0.0
    hidden_dim: Optional[int] = None
    out_hidden_dim: Optional[int] = None
    normalization: type[nn.Module] = nn.LayerNorm
    layer_templ: KANLayer = KANLayer(in_dim=1, out_dim=1)
    final_act: Callable = lambda x: x

    def setup(self):
        norms = []
        dropouts = []
        layers = []
        in_dim = self.hidden_dim or self.in_dim
        out_dim = self.out_hidden_dim or self.out_dim
        out_dims = (*self.inner_dims, out_dim)

        knot_init = np.linspace(-1, 1, self.n_grid, dtype=self.knot_dtype)
        knot_init = np.sign(knot_init) * np.sqrt(np.abs(knot_init))
        self.knots = knot_init

        for out_dim in out_dims:
            layers.append(self.layer_templ.copy(in_dim=in_dim, out_dim=out_dim, knots=self.knots))
            norms.append(self.normalization())
            dropouts.append(nn.Dropout(self.layer_dropout_rate))
            in_dim = out_dim

        if self.hidden_dim is not None:
            self.in_proj = nn.Dense(self.hidden_dim)
        else:
            self.in_proj = lambda x: x

        if self.out_hidden_dim is not None:
            self.out_proj = nn.DenseGeneral(self.out_dim)
        else:
            self.out_proj = lambda x: x

        self.norms = norms
        self.layers = layers
        self.dropouts = dropouts
        self.network = nn.Sequential(self.layers)

    def single_output(self, x: Float[Array, 'in_dim'], training: bool = False) -> Float[Array, ' out_dim']:
        curr_x = self.in_proj(x)
        for layer, norm, dropout in zip(self.layers, self.norms, self.dropouts):
            curr_x = norm(curr_x)
            curr_x = layer(curr_x, training=training)
            curr_x = dropout(curr_x, deterministic=not training)
        y = self.out_proj(curr_x)
        y = self.final_act(curr_x)
        return y

    def __call__(self, x: Float[Array, 'batch in_dim'], training: bool = False) -> Float[Array, ' batch out_dim']:
        return jax.vmap(lambda single: self.single_output(single, training))(x)


if __name__ == '__main__':
    in_dim = 4
    rng = jr.key(0)
    xtest = jr.normal(rng, (in_dim,))

    kan = KAN(in_dim=in_dim, out_dim=6, inner_dims=[5], hidden_dim=15, out_hidden_dim=12)
    flax_summary(kan, x=xtest, compute_flops=True, compute_vjp_flops=True)
