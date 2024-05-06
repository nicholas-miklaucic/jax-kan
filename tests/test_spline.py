"""Tests the splines."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import BSpline

from jax_kan.spline import design_matrix, eval_spline


def _test_eval_spline(seed):
    """Tests the eval_spline function."""
    rng = jax.random.key(seed)

    x = jax.random.uniform(rng, (100,), minval=-1, maxval=1)
    grid = jnp.linspace(-1, 1, 32)
    order = 3
    coef = jax.random.normal(rng, (grid.shape[0] + order - 1,))

    dm = jax.vmap(design_matrix, in_axes=(0, None, None))(x, grid, order)

    out = jax.vmap(eval_spline, in_axes=(0, None, None, None))(x, grid, coef, order)

    h = (grid[-1] - grid[0]) / (grid.shape[0] - 1)
    pad_start = jnp.tile(grid[0], (order,)) - h
    pad_end = jnp.tile(grid[-1], (order,)) + h
    grid = jnp.concat([pad_start, grid, pad_end], axis=0)
    spl = BSpline(grid, coef, order, extrapolate=False)
    dm2 = spl.design_matrix(x, grid, order).todense()

    out2 = spl(x)

    assert np.allclose(dm, dm2)
    assert np.allclose(out, out2, atol=1e-6), f'{out} != {out2}, {np.max(np.abs(out - out2))}'


def test_eval_spline():
    """Tests the eval_spline function."""
    for seed in np.random.randint(0, 1000, 5):
        _test_eval_spline(seed)
