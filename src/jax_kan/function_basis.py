"""Unified interface for basis functions."""

import functools as ft
from typing import Literal, Self, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from jaxtyping import Array, Float, Int

from jax_kan.typing_utils import class_tcheck

EPS = 1e-12


class FunctionBasis:
    """Set of basis functions that represent smooth univariate functions."""

    def __init__(self, n_coefs: int, **params):
        self._n_coefs = n_coefs
        self.params = params

    @classmethod
    def param_names(cls) -> Sequence[str]:
        """Names of the parameters of the basis set."""
        return ()

    @classmethod
    def domain(cls) -> tuple[float, float]:
        """Domain of the function, as an interval. Can include ±∞."""
        raise NotImplementedError

    @property
    def n_coefs(self) -> int:
        """Number of coefficients in the basis."""
        return self._n_coefs

    def design_matrix(self, x: Float[Array, '']) -> Float[Array, 'coefs={self.n_coefs}']:
        """Compute the design matrix of basis functions for the input."""
        raise NotImplementedError

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

    def extend(self, new_n_coefs: int) -> Self:
        """
        Create a new basis with the given number of coefficients.

        Parameters
        ----------
        new_n_coefs: int
            The number of coefficients in the new basis.

        Returns
        -------
        Self
            The new basis.
        """
        return type(self)(new_n_coefs, **self.params)

    def best_fit_coefs(self, x: Float[Array, ' dim'], y: Float[Array, ' dim']) -> Float[Array, ' coefs={self.n_coefs}']:
        """Find the best-fit coefficients for the given (x, y) pairs."""

        return jnp.linalg.lstsq(jax.vmap(self.design_matrix)(x), y)[0]


class Chebyshev(FunctionBasis):
    """Chebyshev polynomials of the first kind."""

    def design_matrix(self, x: Float[Array, '']) -> Float[Array, 'coefs={self.n_coefs}']:
        # x0 = jnp.ones_like(x, dtype=x.dtype)[None]
        polys = [x, 2 * x**2 - 1]

        for _n in range(2, self.n_coefs):
            # https://www.wikiwand.com/en/Chebyshev_polynomials#Recurrence_definition
            polys.append(2 * x * polys[-1] - polys[-2])
        return jnp.array(polys[: self.n_coefs])

    @classmethod
    def domain(cls) -> tuple[float, float]:
        return (-1, 1)


class Legendre(FunctionBasis):
    """Legendre polynomials."""

    def __init__(self, n_coefs):
        super().__init__(n_coefs)

    def design_matrix(self, x: Float[Array, '']) -> Float[Array, 'coefs={self.n_coefs}']:
        return Gegenbauer(self.n_coefs, 0).design_matrix(x)

    @classmethod
    def domain(cls) -> tuple[float, float]:
        return (-1, 1)


def binom(n, k):
    fact = jax.scipy.special.factorial
    return fact(n) / (fact(k) * fact(n - k))


class Jacobi(FunctionBasis):
    """Jacobi polynomials, without normalization."""

    def __init__(self, n_coefs: int, alpha: Float[Array, ''], beta: Float[Array, '']):
        super().__init__(n_coefs, alpha=jnp.tanh(alpha), beta=jnp.tanh(beta))

    @classmethod
    def param_names(cls) -> Sequence[str]:
        return ('alpha', 'beta')

    def design_matrix(self, x: Float[Array, '']) -> Float[Array, 'coefs={self.n_coefs}']:
        alpha = self.params['alpha']
        beta = self.params['beta']

        # x0 = jnp.ones_like(x, dtype=x.dtype)[None]
        x0 = x
        polys = [x0, (alpha + 1) * (alpha + beta + 2) * (x0 - 1) / 2]

        for n in range(2, self.n_coefs):
            # https://www.wikiwand.com/en/Jacobi_polynomials#Recurrence_relations
            a = n + alpha
            b = n + beta
            c = a + b

            den = 2 * n * (c - n) * (c - 2)
            coef1 = ((c - 1) * (c * (c - 2) * x + (a - b) * (c - 2 * n))) / den
            coef2 = (-2 * (a - 1) * (b - 1) * c) / den

            unscaled = coef1 * polys[-1] - coef2 * polys[-2]
            fact = jax.scipy.special.factorial
            scale = 2 ** (alpha + beta + 1) / (c + 1) * (fact(a + 1) * fact(b + 1)) / (fact(a + beta + 1) * fact(n))

            polys.append(unscaled / scale)

        return jnp.array(polys[: self.n_coefs])

    @classmethod
    def domain(cls) -> tuple[float, float]:
        return (-1, 1)


class Gegenbauer(Jacobi):
    """
    Jacobi polynomials, but with α = β fixed. (This means the parameters don't exactly line up with
    the classical notation.)
    """

    def __init__(self, n_coefs: int, alpha: Float[Array, '']):
        super().__init__(n_coefs, alpha=jnp.tanh(alpha), beta=jnp.tanh(alpha))

    @classmethod
    def param_names(cls) -> Sequence[str]:
        return ('alpha',)


class Hermite(FunctionBasis):
    """Hermite polynomials (physicist's version.)"""

    def __init__(self, n_coefs: int):
        super().__init__(n_coefs)

    @property
    def n_coefs(self) -> int:
        return self._n_coefs

    def design_matrix(self, x: Float[Array, '']) -> Float[Array, 'coefs={self.n_coefs}']:
        polys = [1, 2 * x]

        for n in range(2, self.n_coefs + 1):
            polys.append(2 * x * polys[-1] - 2 * (n - 1) * polys[-2])

        nn = jnp.arange(self.n_coefs + 1)
        # scale = jnp.sqrt(jnp.pi) * 2 ** (n - 1) * jax.scipy.special.gamma(n)
        scale = jnp.sqrt(jax.scipy.special.factorial(nn) * jnp.sqrt(jnp.pi) * 2 ** (nn))

        return jnp.array(polys[1 : self.n_coefs + 1]) / jnp.sqrt(scale[1 : self.n_coefs + 1])

    @classmethod
    def domain(cls) -> tuple[float, float]:
        return (-jnp.inf, jnp.inf)


class Fourier(FunctionBasis):
    """Fourier (sine) series."""

    def __init__(self, n_coefs: int):
        super().__init__(n_coefs)

    def design_matrix(self, x: Float[Array, '']) -> Float[Array, 'coefs={self.n_coefs}']:
        return jnp.sin(jnp.pi * x * (jnp.arange(self.n_coefs) + 1))

    @classmethod
    def domain(cls) -> tuple[float, float]:
        return (-jnp.inf, jnp.inf)


interval_maps = {
    'tanh': (jnp.tanh, -1, 1),
    'sigmoid': (jax.nn.sigmoid, 0, 1),
    'sin': (jnp.sin, -1, 1),
    'arctan': (jnp.arctan, -jnp.pi / 2, jnp.pi / 2),
    'rational': (lambda x: (2 * x) / (x**2 + 1), -1, 1),
}

InputMapType = Literal['tanh', 'sigmoid', 'sin', 'arctan', 'rational']


class InputMap(nn.Module):
    """Maps unconstrained reals to the domain of a basis function."""

    pass


class FixedInputMap(InputMap):
    """Maps unconstrained reals to the domain of a basis function."""

    stretch_base: float = 1
    stretch_trainable: bool = True
    map_type: InputMapType = 'tanh'

    def setup(self):
        if self.stretch_trainable:
            self.stretch = self.param('stretch', lambda _rng: jnp.array(self.stretch_base))
        else:
            self.stretch = jnp.array(self.stretch_base)

    def __call__(self, x: Float[Array, ''], basis: FunctionBasis) -> Float[Array, '']:
        interval = basis.domain()
        if interval == (-jnp.inf, jnp.inf):
            return x
        else:
            a2, b2 = interval
            fn, a1, b1 = interval_maps[self.map_type]

            scale = (b2 - a2) / (b1 - a1)

            return (fn(x / self.stretch) - a1) * scale + a2


class PolyInputMap(nn.Module):
    """Maps unconstrained reals to the domain of a basis function."""

    order: int = 3

    def setup(self):
        self.coefs = self.param('coefs', nn.initializers.normal(stddev=0.1), (self.order,))

    def __call__(self, x: Float[Array, ''], basis: FunctionBasis) -> Float[Array, '']:
        interval = basis.domain()
        if interval == (-jnp.inf, jnp.inf):
            return jnp.tanh(5 * x) * jnp.log1p(jnp.abs(x))
        else:
            lo, hi = interval

            raw = jnp.sin(
                (jnp.pi * x * jnp.polyval(self.coefs, x)) / (x ** (self.order + 1 + (self.order + 1) % 2) + 10)
            )

            return ((raw + 1) / 2) * (hi - lo) + lo
