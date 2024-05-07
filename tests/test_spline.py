"""Tests the splines."""

import functools as ft

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from scipy.interpolate import BSpline as ScipyBSpline

from jax_kan.spline import BSpline
from jax_kan.typing_utils import tcheck


@tcheck
def _test_eval_spline(seed):
    """Tests the eval_spline function."""
    rng = jax.random.key(seed)

    x = jax.random.uniform(rng, (100,), minval=-1, maxval=1)
    grid = jnp.linspace(-1, 1, 32)
    order = 3
    coef = jax.random.normal(rng, (grid.shape[0] + order - 1,))

    BatchSpline = nn.vmap(BSpline, in_axes=(0, None), variable_axes={'params': None}, split_rngs={'params': False})  # noqa: N806
    spl = BatchSpline(grid, order)
    spl_params = spl.init(rng, x, coef)

    out = spl.apply(spl_params, x, coef)

    h = (grid[-1] - grid[0]) / (grid.shape[0] - 1)
    pad_start = jnp.tile(grid[0], (order,)) - h
    pad_end = jnp.tile(grid[-1], (order,)) + h
    grid = jnp.concat([pad_start, grid, pad_end], axis=0)

    spl2 = ScipyBSpline(grid, coef, order, extrapolate=False)

    out2 = spl2(x)

    assert np.allclose(out, out2, atol=1e-6), f'{out} != {out2}, {np.max(np.abs(out - out2))}'


def test_eval_spline():
    """Tests the eval_spline function."""
    for seed in np.random.randint(0, 1000, 5):
        _test_eval_spline(seed)


if __name__ == '__main__':
    test_eval_spline()
