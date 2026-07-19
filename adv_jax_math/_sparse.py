# Copyright (C) 2024 Kaya Unalmis
# SPDX-License-Identifier: LGPL-3.0

"""Sparsity-aware JAX derivative utilities."""

from functools import partial, wraps

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.tree_util import tree_leaves, tree_map
from packaging.version import Version

from ._batch import (
    _batch_and_remainder,
    _concat,
    _evaluate_in_chunks,
    _get_first_chunk,
    _scanmap,
    _unchunk,
)
from ._utils import identity


def _is_none(x):
    """Return whether ``x`` is ``None``."""
    return x is None


def _mul_cotangent(p, *, g):
    """Multiply a Jacobian-diagonal leaf by a broadcast cotangent."""
    # not doing sparse linear algebra here since we
    # do not assume the cotangent is sparse.
    # could do case fn expands leaf, but in that scenario
    # it is always the case user should diagonalize later so
    # better to just error.
    if p is None:
        return None
    shape = g.shape + (1,) * (p.ndim - g.ndim)
    return p * g.reshape(shape)


_USE_HIJAX = Version(jax.__version__) >= Version("0.11.0")

if _USE_HIJAX:  # noqa: C901
    from equinox import internal as eqxi
    from jax._src.interpreters.ad import add_tangents
    from jax.experimental.hijax import VJPHiPrimitive, Zero, jvp_from_lin

    def _contract_sparse_jvp(out_ndim, diagonal, tangent):
        """Contract one Jacobian-diagonal leaf with its tangent."""
        if diagonal is None:
            return None
        product = diagonal * tangent
        return product.sum(axis=tuple(range(out_ndim, product.ndim)))

    class _SparsePullbackPrimitive(VJPHiPrimitive):
        jvp = jvp_from_lin

        def __init__(self, y_aval, fn_aval, out_aval, *, fn_static, higher_order):
            self.in_avals = (y_aval, fn_aval)
            self.out_aval = out_aval
            self.params = {"fn_static": fn_static, "higher_order": higher_order}
            super().__init__()

        def _fn(self, fn_dynamic):
            return eqxi.hashable_combine(fn_dynamic, self.fn_static)

        def expand(self, y, fn_dynamic):
            return self._fn(fn_dynamic)(y)

        def batch_dim_rule(self, _axis_data, in_dims):
            return (
                jax.tree.broadcast(0, self.out_aval)
                if tree_leaves(in_dims)
                else eqx.filter(self.out_aval, False)
            )

        def lin(self, nzs_in, y, fn_dynamic):
            out, p = self.vjp_fwd(nzs_in, y, fn_dynamic)
            return out, p, any(tree_leaves(nzs_in[0]))

        def linearized(self, diagonal, y_dot, _fn_dot):
            contributions = tree_map(
                _contract_sparse_jvp,
                jax.tree.broadcast(self.out_aval.ndim, diagonal),
                diagonal,
                y_dot,
                is_leaf=_is_none,
            )
            return jax.tree.reduce(
                add_tangents,
                contributions,
                Zero(self.out_aval.to_tangent_aval()),
            )

        def vjp_fwd(self, nzs_in, y, fn_dynamic):
            out, vjp_fn = eqx.filter_vjp(self._fn(fn_dynamic), y)
            p = eqx.filter(vjp_fn(jnp.ones_like(out))[0], nzs_in[0])
            return self(y, fn_dynamic) if self.higher_order else out, p

        def vjp_bwd_retval(self, p, g):
            return (
                eqx.filter(self.in_avals, False)
                if isinstance(g, Zero)
                else (
                    tree_map(partial(_mul_cotangent, g=g), p),
                    eqx.filter(self.in_avals[1], False),
                )
            )

    def _sparse_pullback(y, *, fn, higher_order):
        fn_dynamic, fn_static = eqxi.hashable_partition(fn, eqx.is_array)
        return _SparsePullbackPrimitive(
            tree_map(jax.typeof, y),
            tree_map(jax.typeof, fn_dynamic),
            tree_map(jax.typeof, fn.out_struct),
            fn_static=fn_static,
            higher_order=higher_order,
        )(y, fn_dynamic)

else:

    @eqx.filter_custom_vjp
    def _sparse_pullback(y, *, fn, higher_order):
        return fn(y)

    @_sparse_pullback.def_fwd
    def _sparse_pullback_fwd(perturbed, y, *, fn, higher_order):
        out, vjp_fn = eqx.filter_vjp(fn, y)
        p = eqx.filter(vjp_fn(jnp.ones_like(out))[0], perturbed)
        return out, p

    @_sparse_pullback.def_bwd
    def _sparse_pullback_bwd(p, g, perturbed, y, *, fn, higher_order):
        return tree_map(partial(_mul_cotangent, g=g), p)


def sparse_pullback_map(fn, y, *, higher_order=False):
    """Wrapper for sparsity exploiting pullback.

    Wraps the given map with logic to ensure cotangents flow through the diagonal
    of its pullback. The derivatives will be exact for maps whose Jacobians are
    block diagonal.

    See Also
    --------
    sparse_pullback
        Applies the same transformation and immediatly returns its output.

    Parameters
    ----------
    fn : callable
        Vectorized map.
    y : pytree
        Example input used to closure-convert ``fn``.
    higher_order : bool
        Whether to support higher-order differentiation with the HiJAX backend
        at the expense of evaluating the primal ``fn`` twice. The custom-VJP
        backend supports higher-order differentiation regardless of this flag.
        Default is ``False``.

    Returns
    -------
    wrapper : callable
        Same forward map but with a sparsity exploiting pullback.

    Examples
    --------
    >>> fn = sparse_pullback_map(fn, y)
    >>> out = fn(y)

    """
    fn = eqx.filter_closure_convert(fn, y)
    return wraps(fn)(partial(_sparse_pullback, fn=fn, higher_order=higher_order))


def _sparse_pullback_sharded(fn, y, *, higher_order):
    return _sparse_pullback(
        y, fn=eqx.filter_closure_convert(fn, y), higher_order=higher_order
    )


def _sparse_pullback_sharded_map_stripped(fn, y, *, higher_order):
    return jax.vmap(
        partial(
            _sparse_pullback_sharded,
            fn,
            higher_order=higher_order,
        )
    )(y)


def sparse_pullback(
    fn,
    y,
    /,
    batch_size=None,
    *,
    reduction=None,
    chunk_reduction=identity,
    strip_dim0=False,
    shard_input_data=False,
    higher_order=False,
):
    """Compute ``chunk_reduction(fn(fun_input))`` in batches with sparse pullbacks.

    Wraps the given map with logic to ensure cotangents flow through the diagonal
    of its pullback. The derivatives will be exact for maps whose Jacobians are
    block diagonal.

    Notes
    -----
    This method does not automatically wrap ``fn`` with ``vmap``.
    Unless ``fn`` is already wrapped with ``vmap``, the leading dimension
    of ``y`` will not be stripped before it is passed into ``fn``.
    This can be inconvenient for nesting calls to ``sparse_pullback``, since
    only batching along the first axis is currently supported.
    However, the ``strip_dim0`` flag should cover the most common case
    of nesting calls where ``batch_size`` is one on the outermost call.

    See Also
    --------
    sparse_pullback_map
        Functional version.

    Parameters
    ----------
    fn : callable
        Vectorized map.
    y : pytree
        Data to split into batches to feed to ``fn``.
    batch_size : int or None
        Size of batches. If no batching should be done or the batch size is the
        full input then supply ``None``.
    reduction : callable or None
        Binary reduction operation.
        Should take two arguments and return one output, e.g. ``jnp.add``.
    chunk_reduction : callable
        Chunk-wise reduction operation.
        Should typically apply ``reduction`` along the mapped axis,
        e.g. ``jnp.add.reduce``.
    strip_dim0 : bool
        Whether to strip the leading dim of ``y`` before passing it
        to ``fn``; see notes. This flag only works if ``batch_size`` is one.
        It should be set to ``False`` if ``fn`` is wrapped in ``vmap``.
        Default is ``False``.
    shard_input_data : bool
        Whether to shard ``y`` across devices before applying chunked batching.
        The divisible prefix is split across devices; when supplied,
        ``batch_size`` bounds the batches processed on each device. A local
        remainder is evaluated once per device, and a final global remainder is
        evaluated once overall. The input length need not be divisible by either
        the device count or ``batch_size``. Default is ``False``.
    higher_order : bool
        Whether to support higher-order differentiation with the HiJAX backend
        at the expense of evaluating the primal ``fn`` twice. The custom-VJP
        backend supports higher-order differentiation regardless of this flag.
        Default is ``False``.

    Returns
    -------
    out : pytree
        Returns ``chunk_reduction(fn(y))``.

    Examples
    --------
    >>> out = sparse_pullback(fn, y)

    """
    if shard_input_data:
        sparse_fun = partial(
            (
                _sparse_pullback_sharded_map_stripped
                if strip_dim0 and batch_size == 1
                else _sparse_pullback_sharded
            ),
            fn,
            higher_order=higher_order,
        )

        return _evaluate_in_chunks(
            sparse_fun,
            batch_size,
            (0,),
            reduction,
            chunk_reduction,
            True,
            y,
        )

    if strip_dim0 and batch_size == 1:
        return _scanmap(
            sparse_pullback_map(
                fn,
                _get_first_chunk(y),
                higher_order=higher_order,
            ),
            0,
            reduction,
            identity,
        )(y)

    if batch_size is None or (n_elements := tree_leaves(y)[0].shape[0]) <= batch_size:
        return chunk_reduction(
            _sparse_pullback(
                y,
                fn=eqx.filter_closure_convert(fn, y),
                higher_order=higher_order,
            )
        )

    y, remain = _batch_and_remainder(y, batch_size)
    # Note that num_batches in _batch_and_remainder is always positive.

    y = _scanmap(
        sparse_pullback_map(
            fn,
            _get_first_chunk(y),
            higher_order=higher_order,
        ),
        0,
        reduction,
        chunk_reduction,
    )(y)

    if reduction is None:
        y = _unchunk(y)

    if n_elements % batch_size == 0:
        return y

    remain = chunk_reduction(
        _sparse_pullback(
            remain,
            fn=eqx.filter_closure_convert(fn, remain),
            higher_order=higher_order,
        )
    )

    if reduction is None:
        return _concat(y, remain)

    return reduction(y, remain)
