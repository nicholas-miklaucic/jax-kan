"""JAX utilties for working with B-Splines and similar concepts."""

import functools as ft

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax import struct
from jaxtyping import Array, Float, Int
from orthax import chebyshev as C
from orthax import legendre as L

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


class Chebyshev(BSpline):
    """Chebyshev polynomials of the first kind."""

    def __init__(self, n_coefs: int):
        self._n_coefs = n_coefs

    @property
    def n_coefs(self) -> int:
        return self._n_coefs

    def design_matrix(self, x: Float[Array, '']) -> Float[Array, 'coefs={self.n_coefs}']:
        # x0 = jnp.ones_like(x, dtype=x.dtype)[None]
        polys = [1, x]

        for _n in range(2, self.n_coefs):
            # https://www.wikiwand.com/en/Chebyshev_polynomials#Recurrence_definition
            polys.append(2 * x * polys[-1] - polys[-2])
        return jnp.array(polys[: self.n_coefs])


class Jacobi(Chebyshev):
    """Jacobi polynomials, scaled so the value at 1 is 1."""

    def __init__(self, alpha: Float[Array, ''], beta: Float[Array, ''], n_coefs: int):
        super().__init__(n_coefs)
        self.alpha = alpha
        self.beta = beta

        # self.scaling = jnp.max(jnp.abs(jax.vmap(self._design_matrix)(jnp.linspace(-1, 1, 8))))
        self.scaling = 1

    def _design_matrix(self, x: Float[Array, '']) -> Float[Array, 'coefs={self.n_coefs}']:
        a = self.alpha
        b = self.beta

        # x0 = jnp.ones_like(x, dtype=x.dtype)[None]
        x0 = x
        polys = [x0, (a + 1) * (a + b + 2) * (x0 - 1) / 2]

        for n in range(2, self.n_coefs):
            # https://www.wikiwand.com/en/Jacobi_polynomials#Recurrence_relations
            a = n + self.alpha
            b = n + self.beta
            c = a + b

            den = 2 * n * (c - n) * (c - 2)
            coef1 = ((c - 1) * (c * (c - 2) * x + (a - b) * (c - 2 * n))) / den
            coef2 = (-2 * (a - 1) * (b - 1) * c) / den

            polys.append(coef1 * polys[-1] - coef2 * polys[-2])

        return jnp.array(polys[: self.n_coefs])

    def design_matrix(self, x: Float[Array, '']) -> Float[Array, 'coefs={self.n_coefs}']:
        return self._design_matrix(x) / self.scaling
