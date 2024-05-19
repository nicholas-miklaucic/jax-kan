"""Flax modules defining the KAN and a single KAN layer."""

from collections import defaultdict
import functools as ft
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Self

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from eins import EinsOp, ElementwiseOps as E
from eins import Reductions as R
from eins.elementwise import ElementwiseFunc
from flax import linen as nn
from jaxtyping import Array, Float

from jax_kan.function_basis import Chebyshev, FunctionBasis, InputMap, InputMapType, Jacobi
from jax_kan.typing_utils import class_tcheck, tcheck
from jax_kan.utils import debug_stat, debug_structure, flax_summary


# @class_tcheck
class KANLayer(nn.Module):
    in_dim: int
    out_dim: int
    n_coef: int = 5
    dropout_rate: float = 0.0
    kernel_init: Callable = nn.initializers.normal(stddev=1)
    resid_scale_trainable: bool = False
    resid_scale_init: Callable = nn.initializers.ones
    spline_kind: type[FunctionBasis] = Jacobi
    spline_scale_trainable: bool = False
    spline_scale_init: Callable = nn.initializers.zeros
    base_act: ElementwiseFunc = E.positive
    input_map: InputMap = InputMap()
    spline_params_init: Mapping[str, nn.initializers.Initializer] = field(default_factory=dict)
    spline_params_share: bool = False

    def setup(self):
        self.size = self.in_dim * self.out_dim

        params = {
            name: self.spline_params_init.get(name, nn.initializers.zeros) for name in self.spline_kind.param_names()
        }

        shape = () if self.spline_params_share else (self.in_dim,)
        self.spline_params = {name: self.param(name, init, shape) for name, init in params.items()}

        # self.spline = BSpline(self.grid, self.order)
        self.coefs = nn.Einsum((self.in_dim, self.out_dim, self.n_coef), 'ic,ioc->io', use_bias=True)

        if self.resid_scale_trainable:
            self.resid_scale = self.param('resid_scale', self.resid_scale_init, (self.in_dim, 1))
        else:
            self.resid_scale = 0

        if self.spline_scale_trainable:
            self.spline_scale = self.param('spline_scale', self.spline_scale_init, (self.in_dim, self.out_dim))
        else:
            self.spline_scale = 1

        self.dropout = nn.Dropout(self.dropout_rate)

    def __call__(self: Self, x: Float[Array, 'in_dim'], training: bool = False) -> Float[Array, ' out_dim']:
        # spline = Chebyshev(n_coefs=self.n_coef)

        def design_matrix(x, params):
            spline = self.spline_kind(self.n_coef, **params)
            return spline.design_matrix(x)

        if self.spline_params_share:
            in_axes = (0, None)
        else:
            in_axes = (0, 0)
        dm = jax.vmap(design_matrix, in_axes=in_axes)(self.input_map(x, self.spline_kind), self.spline_params)
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

        for out_dim in out_dims:
            layers.append(self.layer_templ.copy(in_dim=in_dim, out_dim=out_dim))
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
    in_dim = 64
    rng = jr.key(0)
    xtest = jr.normal(
        rng,
        (32, in_dim),
    )

    kan = KAN(in_dim=in_dim, out_dim=6, inner_dims=[32, 16], layer_templ=KANLayer(1, 1, spline_params_share=True))

    out, params = kan.init_with_output(rng, xtest, training=False)
    debug_stat(out=jnp.abs(out), params=params)
    flax_summary(kan, x=xtest, compute_flops=True, compute_vjp_flops=True)
