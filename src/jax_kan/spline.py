"""JAX utilties for working with B-Splines."""

import functools as ft

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from jax_kan.typing import tcheck


@tcheck
def design_matrix(
    x: Float[Array, ''], grid: Float[Array, ' grid'], order: int | Int[Array, ''], eps: float = 1e-12
) -> Float[Array, 'coefs=grid+{order}-1']:
    """
    Compute the design matrix of basis splines for the input.

    Parameters
    ----------
    x: Float[]
        The value for which to compute the splines for.
    grid: Float[grid]
        The knots on which to compute the splines.
    order: int
        The order of the splines.
    eps: float
        The epsilon added to avoid dividing by 0 if two knots are extremely close.

    Returns
    -------
    Float[grid + p - 1]
        The values of the spline basis at the input x.
    """
    h = (grid[-1] - grid[0]) / (grid.shape[0] - 1)
    pad_start = jnp.tile(grid[0], (order,)) - h
    pad_end = jnp.tile(grid[-1], (order,)) + h
    grid = jnp.concat([pad_start, grid, pad_end], axis=0)

    b_prev = ((x >= grid[:-1]) * (x < grid[1:])).astype(x.dtype)

    for k in range(1, order + 1):
        value = x - grid[: -(k + 1)]
        value = value / (grid[k:-1] - grid[: -(k + 1)] + eps)
        value = value * b_prev[:-1]
        value = value + (grid[k + 1 :] - x) / (grid[k + 1 :] - grid[1:(-k)] + eps) * b_prev[1:]
        b_prev = value
    return b_prev


@tcheck
def eval_spline(
    x: Float[Array, ''],
    grid: Float[Array, ' grid'],
    coef: Float[Array, ' coefs=grid+{order}-1'],
    order: int | Int[Array, ''],
) -> Float[Array, '']:
    """
    Evaluates the spline defined on grid with coefficients coef at x.

    Parameters
    ----------
    x: Float[]
        The value for which to compute the splines for.
    grid: Float[grid]
        The knots on which to compute the splines.
    coef: Float[coefs]
        The coefficients of the splines.
    order: int
        The order of the splines.

    Returns
    -------
    Float[]
        The value of the spline curve at x.
    """

    return jnp.dot(design_matrix(x, grid, order), coef)
