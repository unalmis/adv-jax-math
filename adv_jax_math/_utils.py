"""General utilities."""

import operator
import warnings
from typing import Any, Type, Union

import jax
import jax.numpy as jnp
from jaxtyping import Array, Inexact
from numpy.typing import ArrayLike


class _Indexable:
    """Build an index tuple for selecting one axis of an array."""

    __slots__ = ()

    def __getitem__(self, index):
        return index

    @staticmethod
    def get(item, axis: int, ndim: int) -> tuple:
        """Return a full index tuple with ``item`` at ``axis``."""
        index = [slice(None)] * ndim
        index[axis] = item
        return tuple(index)


Index = _Indexable()


def isnonnegint(x: Any) -> bool:
    """Determine if x is a non-negative integer."""
    try:
        _ = operator.index(x)
    except TypeError:
        return False
    return x >= 0


def isposint(x: Any) -> bool:
    """Determine if x is a strictly positive integer."""
    return isnonnegint(x) and (x > 0)


def errorif(
    cond: Union[bool, jax.Array], err: Type[Exception] = ValueError, msg: str = ""
):
    """Raise an error if condition is met.

    Similar to assert but allows wider range of Error types, rather than
    just AssertionError.
    """
    if cond:
        raise err(msg)


def warnif(
    cond: Union[bool, jax.Array], err: Type[Warning] = UserWarning, msg: str = ""
):
    """Throw a warning if condition is met."""
    if cond:
        warnings.warn(msg, err)


def asarray_inexact(x: ArrayLike) -> Inexact[Array, "..."]:
    """Convert to jax array with floating point dtype."""
    x = jnp.asarray(x)
    if x.weak_type:  # preserve weakly typed things like scalars
        return x
    dtype = x.dtype
    if not jnp.issubdtype(dtype, jnp.inexact):
        dtype = jnp.result_type(x, jnp.array(1.0))
    return x.astype(dtype)


def setdefault(val, default, cond=None):
    """Return val if condition is met, otherwise default.

    If cond is None, then it checks if val is not None, returning val
    or default accordingly.
    """
    return val if cond or (cond is None and val is not None) else default


def ensure_tuple(x):
    """Return ``x`` as a tuple without nesting existing lists or tuples."""
    if isinstance(x, tuple):
        return x
    if isinstance(x, list):
        return tuple(x)
    return (x,)


def identity(x):
    """Return ``x`` unchanged."""
    return x
