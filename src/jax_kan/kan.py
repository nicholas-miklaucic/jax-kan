"""Flax modules defining the KAN and a single KAN layer."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Callable, Optional, Self

import jax
import jax.numpy as jnp
import jax.random as jr
from eins import EinsOp
from flax import linen as nn
from jaxtyping import Array, Float

from jax_kan.spline import design_matrix, eval_spline
from jax_kan.typing_utils import class_tcheck, tcheck


@class_tcheck
@dataclass
class KANLayerOutput:
    y: Float[Array, ' out_dim']
    postacts: Float[Array, ' out_dim in_dim']
    postspline: Float[Array, ' out_dim in_dim']


@class_tcheck
class KANLayer(nn.Module):
    in_dim: int
    out_dim: int
    order: int = 2
    dropout_rate: float = 0.0
    kernel_init: Callable = nn.initializers.normal(stddev=0.5)
    resid_scale_trainable: bool = False
    resid_scale_init: Callable = nn.initializers.ones
    spline_scale_trainable: bool = False
    spline_scale_init: Callable = nn.initializers.ones
    base_act: Callable = nn.tanh
    knots: Float[Array, ' grid'] = field(default_factory=lambda: jnp.array([-1, -0.5, 0, 0.5, 1]))

    def setup(self):
        self.size = self.in_dim * self.out_dim
        n_grid = len(self.knots)
        self.grid = self.knots

        def spline_init(*args, **kwargs):
            noise = self.kernel_init(*args, **kwargs)
            # debug_structure(grid=self.grid, noise=noise)
            # return curve2coef(self.grid.T, noise.T, self.grid, self.order)
            return noise

        self.coef = self.param('coef', spline_init, (self.size, n_grid + self.order - 1))
        if self.resid_scale_trainable:
            self.resid_scale = self.param('resid_scale', self.resid_scale_init, (self.size,))
        else:
            self.resid_scale = 1

        if self.spline_scale_trainable:
            self.spline_scale = self.param('spline_scale', self.spline_scale_init, (self.size,))
        else:
            self.spline_scale = 1

        self.dropout = nn.Dropout(self.dropout_rate)

    def full_output(self: Self, x: Float[Array, 'in_dim'], training: bool = False) -> KANLayerOutput:
        # splines: (out_dim in_dim)
        x = jnp.tile(x, (self.out_dim, 1)).reshape(-1)
        y = jax.vmap(eval_spline, in_axes=(0, None, 0, None))(jnp.tanh(x), self.grid, self.coef, self.order)
        postspline = y.reshape(self.out_dim, self.in_dim)

        y = self.resid_scale * self.base_act(x) + self.spline_scale * y
        postacts = y.reshape(self.out_dim, self.in_dim)

        y = self.dropout(y, deterministic=not training)
        y = EinsOp('(out in) -> out', reduce='mean', symbol_values={'out': self.out_dim})(y)

        return KANLayerOutput(y=y, postacts=postacts, postspline=postspline)

    def __call__(self: Self, x: Float[Array, 'batch in_dim'], training: bool = False) -> Float[Array, 'batch out_dim']:
        out = lambda x: self.full_output(x, training=training).y
        return jax.vmap(out)(x)


@class_tcheck
class KAN(nn.Module):
    in_dim: int
    out_dim: int
    inner_dims: Sequence[int]

    n_grid: int = 16
    knot_dtype: jnp.dtype = jnp.float32
    train_knots: bool = True
    layer_dropout_rate: float = 0.0
    hidden_dim: Optional[int] = None
    out_hidden_dim: Optional[int] = None
    normalization: Callable = nn.LayerNorm
    layer_templ: KANLayer = KANLayer(in_dim=1, out_dim=1)
    final_act: Callable = lambda x: x

    def setup(self):
        norms = []
        dropouts = []
        layers = []
        in_dim = self.hidden_dim or self.in_dim
        out_dim = self.out_hidden_dim or self.out_dim
        out_dims = (*self.inner_dims, out_dim)

        knot_start = jnp.linspace(-1, 1, self.n_grid, dtype=self.knot_dtype)[1:-1]
        knot_start = jnp.sign(knot_start) * jnp.sqrt(jnp.abs(knot_start))
        if self.train_knots:
            self.knots = self.param(
                'knots',
                lambda _rng, _shape: knot_start,
                (in_dim,),
            )
        else:
            self.knots = knot_start

        for out_dim in out_dims:
            layers.append(self.layer_templ.copy(in_dim=in_dim, out_dim=out_dim, knots=self.grid_points()))
            norms.append(self.normalization())
            dropouts.append(nn.Dropout(self.layer_dropout_rate))
            in_dim = out_dim

        if self.hidden_dim is not None:
            self.in_proj = nn.Dense(self.hidden_dim)
        else:
            self.in_proj = lambda x: x

        if self.hidden_dim is not None:
            self.out_proj = nn.Dense(self.out_dim, kernel_init=nn.initializers.ones, bias_init=nn.initializers.zeros)
        else:
            self.out_proj = lambda x: x

        self.norms = norms
        self.layers = layers
        self.dropouts = dropouts
        self.network = nn.Sequential(self.layers)

    def grid_points(self):
        return jnp.concat([jnp.array([-1]), self.knots, jnp.array([1])])

    def full_outputs(
        self, x: Float[Array, 'in_dim'], training: bool = False
    ) -> tuple[Float[Array, ' out_dim'], Sequence[KANLayerOutput]]:
        outputs = []
        curr_x = self.in_proj(x)
        for layer, norm, dropout in zip(self.layers, self.norms, self.dropouts):
            curr_x = norm(curr_x)
            outputs.append(layer.full_output(curr_x, training=training))
            curr_x = dropout(outputs[-1].y, deterministic=not training)
        y = self.out_proj(curr_x)
        y = self.final_act(curr_x)
        return y, outputs

    def __call__(self, x: Float[Array, 'batch in_dim'], training: bool = False) -> Float[Array, 'batch out_dim']:
        out = lambda b: self.full_outputs(b, training=training)[0]
        return jax.vmap(out)(x)


if __name__ == '__main__':
    in_dim = 4
    rng = jr.key(0)
    xtest = jr.normal(rng, (14, in_dim))

    kan = KAN(in_dim=in_dim, out_dim=6, inner_dims=[5], hidden_dim=15, out_hidden_dim=12)
    print(kan.tabulate(rng, xtest))
