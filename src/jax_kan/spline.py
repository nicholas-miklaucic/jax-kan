"""JAX utilties for working with B-Splines and similar concepts."""

import functools as ft

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax import struct
from jaxtyping import Array, Float, Int

from jax_kan.typing_utils import class_tcheck

EPS = 1e-12


# @class_tcheck
class BSpline:
    """B-spline."""

    def __init__(self, grid: Float[Array, ' grid'], order: int = 2):
        self.grid = grid
        self.order = order
        self.n_grid = self.grid.shape[0]

        def pad_grid(grid, order):
            h = (grid[-1] - grid[0]) / (grid.shape[0] - 1)
            pad_start = jnp.tile(grid[0], (order,)) - h
            pad_end = jnp.tile(grid[-1], (order,)) + h
            return jnp.concat([pad_start, grid, pad_end], axis=0)

        self.knots = pad_grid(self.grid, self.order)

    @property
    def n_coefs(self) -> int:
        return self.n_grid + self.order - 1

    def design_matrix(self, x: Float[Array, '']) -> Float[Array, 'coefs={self.n_grid}+{self.order}-1']:
        """
        Compute the design matrix of basis splines for the input.

        Parameters
        ----------
        x: Float[]
            The value for which to compute the splines for.

        Returns
        -------
        Float[grid + order - 1]
            The values of the spline basis at the input x.
        """
        grid = self.knots

        b_prev = ((x >= grid[:-1]) * (x < grid[1:])).astype(x.dtype)

        for k in range(1, self.order + 1):
            value = x - grid[: -(k + 1)]
            value = value / (grid[k:-1] - grid[: -(k + 1)] + EPS)
            value = value * b_prev[:-1]
            value = value + (grid[k + 1 :] - x) / (grid[k + 1 :] - grid[1:(-k)] + EPS) * b_prev[1:]
            b_prev = value
        return b_prev

    def __call__(
        self,
        x: Float[Array, ''],
        coefs: Float[Array, ' coefs={self.n_grid}+{self.order}-1'],
    ) -> Float[Array, '']:
        """
        Evaluates the spline defined on grid with coefficients coef at x.

        Parameters
        ----------
        x: Float[]
            The value for which to compute the splines for.
        coef: Float[coefs]
            The coefficients of the splines.

        Returns
        -------
        Float[]
            The value of the spline curve at x.
        """

        return jnp.dot(self.design_matrix(x), coefs)
