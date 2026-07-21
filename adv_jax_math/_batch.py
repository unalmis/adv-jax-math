# Copyright (C) 2024 Kaya Unalmis
# SPDX-License-Identifier: LGPL-3.0

"""Batched operations."""

from functools import partial

import jax
import jax.numpy as jnp
from jax import vmap
from jax._src import config
from jax._src.api import (
    _check_input_dtype_jacfwd,
    _check_input_dtype_jacrev,
    _check_output_dtype_jacfwd,
    _check_output_dtype_jacrev,
    _jacfwd_unravel,
    _jacrev_unravel,
    _std_basis,
)
from jax._src.api_util import _ensure_index, check_callable
from jax._src.lax.control_flow.loops import _batch_and_remainder
from jax._src.numpy.vectorize import (
    _apply_excluded,
    _check_output_dims,
    _parse_gufunc_signature,
    _parse_input_dimensions,
)
from jax._src.util import wraps
from jax.lax import scan
from jax.sharding import NamedSharding, PartitionSpec
from jax.tree_util import (
    tree_flatten,
    tree_leaves,
    tree_map,
    tree_structure,
    tree_transpose,
)
from packaging import version

from ._utils import errorif, identity, warnif

try:
    from jax._src.api_util import argnums_partial2
except ImportError:
    from jax._src.api_util import _ensure_inbounds, _ensure_index_tuple

    def argnums_partial2(fun, dyn_argnums, args, kwargs):
        """Partially apply arguments while keeping an ordinary callable."""
        dyn_argnums = _ensure_index_tuple(dyn_argnums)
        dyn_argnums = _ensure_inbounds(False, len(args), dyn_argnums)
        static_args = list(args)
        dyn_args = []
        for i in dyn_argnums:
            dyn_args.append(static_args[i])
            static_args[i] = None

        def wrapped(*dyn_args_):
            args_ = list(static_args)
            for i, arg in zip(dyn_argnums, dyn_args_):
                args_[i] = arg
            return fun(*args_, **kwargs)

        return wrapped, tuple(dyn_args)


try:
    from jax.sharding import AxisType, reshard
except ImportError:
    AxisType = None
    reshard = None

_JAX_VERSION = version.parse(jax.__version__)

# Sharded batching uses automatic mesh axes so that device placement does not
# leak an explicit mesh into pullback cotangent types. Older JAX versions fall
# back to ordinary chunking.
_SUPPORTS_SHARDED_BATCHING = (
    _JAX_VERSION >= version.parse("0.10.2")
    and AxisType is not None
    and reshard is not None
)

_unchunk = partial(tree_map, lambda y: y.reshape(-1, *y.shape[2:]))
_concat = partial(tree_map, lambda y1, y2: jnp.concatenate((y1, y2)))
_get_first_chunk = partial(tree_map, lambda x: x[0])


def _reshape_sharded(x, shape, spec, mesh):
    return jax.lax.with_sharding_constraint(x.reshape(shape), NamedSharding(mesh, spec))


def _make_automatic_mesh(num_devices):
    return jax.make_mesh(
        (num_devices,),
        ("x",),
        axis_types=(AxisType.Auto,),
    )


def _validate_sharding_mesh(mesh):
    errorif(
        len(mesh.axis_names) != 1,
        ValueError,
        "Sharded batching requires a one-dimensional mesh.",
    )
    errorif(
        mesh.axis_types != (AxisType.Auto,),
        ValueError,
        "Sharded batching requires its mesh axis to have type AxisType.Auto.",
    )
    return mesh


def _axis_name(mesh):
    return mesh.axis_names[0]


def _empty_leading_axis_like(x):
    return jnp.empty((0, *x.shape[1:]), dtype=x.dtype)


def _reshard_leaf_to_replicated(x, mesh):
    sharding = NamedSharding(mesh, PartitionSpec(*(None,) * x.ndim))
    return reshard(x, sharding)


def _has_explicit_sharding(x):
    sharding = getattr(jax.typeof(x), "sharding", None)
    return (
        AxisType is not None
        and isinstance(sharding, NamedSharding)
        and AxisType.Explicit in sharding.mesh.axis_types
    )


def _concat_resharded_to_replicated(x, y, mesh):
    fun = partial(_reshard_leaf_to_replicated, mesh=mesh)
    return _concat(tree_map(fun, x), tree_map(fun, y))


def _scan_append(f, x, reduction=None):
    """Evaluate f element-wise in x while appending the results."""

    def body(carry, x):
        return (), f(x)

    _, result = scan(body, (), x)
    return result


def _scan_reduce(f, x, reduction=None):
    """Evaluate f element-wise in x while reducing the results."""

    def body(carry, x):
        return reduction(carry, f(x)), None

    result, _ = scan(body, f(_get_first_chunk(x)), tree_map(lambda leaf: leaf[1:], x))
    return result


def _scanmap(fun, argnums=0, reduction=None, chunk_reduction=identity):
    """A helper function to wrap f with a scan_fun."""
    scan_fun = _scan_append if reduction is None else _scan_reduce

    def f_(*args, **kwargs):
        f_partial, dyn_args = argnums_partial2(fun, argnums, args, kwargs)
        return scan_fun(
            lambda x: chunk_reduction(f_partial(*x)),
            dyn_args,
            reduction,
        )

    return f_


def _split_shardable(f, mesh):
    leaves, treedef = tree_flatten(f)
    out = [_shard(leaf, mesh) for leaf in leaves]
    sf = treedef.unflatten(f[0] for f in out)
    rf = treedef.unflatten(f[1] for f in out)
    return sf, rf


def _duplicate_unmapped_explicit(f, mesh):
    f = tree_map(
        lambda leaf: (
            _reshard_leaf_to_replicated(leaf, mesh)
            if hasattr(leaf, "ndim") and _has_explicit_sharding(leaf)
            else leaf
        ),
        f,
    )
    return f, f


def _shard(f, mesh):
    num_devices = mesh.size
    has_explicit_sharding = _has_explicit_sharding(f)
    shardable_size = f.shape[0] - (f.shape[0] % num_devices)
    sharding = NamedSharding(
        mesh,
        PartitionSpec(_axis_name(mesh), *(None,) * (f.ndim - 1)),
    )
    if has_explicit_sharding:
        if shardable_size == 0:
            # Empty slices of Explicit arrays require a mesh context on JAX
            # 0.10.2. The empty prefix is not evaluated, so construct it
            # independently and replicate the full remainder directly.
            return (
                _empty_leading_axis_like(f),
                _reshard_leaf_to_replicated(f, mesh),
            )
        # Change the divisible prefix directly to the batching layout. Replicate
        # only the small global remainder that is evaluated on one device.
        sf = jax.device_put(
            f if shardable_size == f.shape[0] else f[:shardable_size], sharding
        )
        if shardable_size < f.shape[0]:
            rf = f[shardable_size:]
            rf = _reshard_leaf_to_replicated(rf, mesh)
        else:
            # Avoid slicing an empty Explicit array on JAX 0.10.2. The empty
            # remainder is never evaluated, so it does not need a device layout.
            rf = _empty_leading_axis_like(f)
    else:
        sf = f[:shardable_size]
        rf = f[shardable_size:]
        sf = jax.lax.with_sharding_constraint(sf, sharding)
    return sf, rf


def _to_device_local_leaf(x, mesh):
    local_size = x.shape[0] // mesh.size
    shape = (mesh.size, local_size, *x.shape[1:])
    # Device-local layout: one shard per device, with an unsharded local axis.
    spec = PartitionSpec(_axis_name(mesh), *(None,) * x.ndim)
    return _reshape_sharded(x, shape, spec, mesh)


def _flatten_device_local_leaf(x, mesh):
    shape = (x.shape[0] * x.shape[1], *x.shape[2:])
    spec = PartitionSpec(_axis_name(mesh), *(None,) * (x.ndim - 2))
    return _reshape_sharded(x, shape, spec, mesh)


def _flat_to_device_local_leaf(x, local_size, mesh):
    shape = (mesh.size, local_size, *x.shape[1:])
    spec = PartitionSpec(_axis_name(mesh), None, *(None,) * (x.ndim - 1))
    return _reshape_sharded(x, shape, spec, mesh)


def _batch_device_local_leaf(x, batch_size, mesh):
    local_size = x.shape[1]
    num_chunks = local_size // batch_size
    chunked_size = num_chunks * batch_size
    full = x[:, :chunked_size]
    # Scan chunk layout after moveaxis: (num_chunks, num_devices, batch_size, ...).
    full = _reshape_sharded(
        full,
        (x.shape[0], num_chunks, batch_size, *x.shape[2:]),
        PartitionSpec(
            _axis_name(mesh),
            None,
            None,
            *(None,) * (x.ndim - 2),
        ),
        mesh,
    )
    return jnp.moveaxis(full, 1, 0)


def _unbatch_device_local_leaf(y, mesh):
    y = jnp.moveaxis(y, 0, 1)
    shape = (y.shape[0], y.shape[1] * y.shape[2], *y.shape[3:])
    spec = PartitionSpec(_axis_name(mesh), None, *(None,) * (y.ndim - 3))
    return _reshape_sharded(y, shape, spec, mesh)


_concat_device_local = partial(
    tree_map, lambda y1, y2: jnp.concatenate((y1, y2), axis=1)
)


def _flatten_device_local(x, mesh):
    return tree_map(lambda y: _flatten_device_local_leaf(y, mesh), x)


def _unbatch_device_local(x, mesh):
    return tree_map(lambda y: _unbatch_device_local_leaf(y, mesh), x)


def _to_device_local(x, mesh):
    return tree_map(lambda y: _to_device_local_leaf(y, mesh), x)


def _flat_to_device_local(x, local_size, mesh):
    return tree_map(lambda y: _flat_to_device_local_leaf(y, local_size, mesh), x)


def _batch_device_local(x, batch_size, mesh):
    return tree_map(lambda y: _batch_device_local_leaf(y, batch_size, mesh), x)


def _device_local_remainder(x, chunked_size):
    return tree_map(lambda y: y[:, chunked_size:], x)


def _scan_device_local_chunks(
    fun,
    argnums,
    reduction,
    chunk_reduction,
    batch_size,
    mesh,
    *args,
    **kwargs,
):
    f_partial, dyn_args = argnums_partial2(fun, argnums, args, kwargs)

    def chunk_fun(x):
        x = _flatten_device_local(x, mesh)
        y = chunk_reduction(f_partial(*x))
        if reduction is None:
            y = _flat_to_device_local(y, batch_size, mesh)
        return y

    scan_fun = _scan_append if reduction is None else _scan_reduce
    return scan_fun(chunk_fun, dyn_args, reduction)


def _evaluate_device_local_in_chunks(
    fun,
    batch_size,
    argnums,
    reduction,
    chunk_reduction,
    mesh,
    *args,
    **kwargs,
):
    local_size = tree_leaves(args[argnums[0]])[0].shape[1]

    if local_size <= batch_size:
        flat_args = tuple(
            _flatten_device_local(a, mesh) if i in argnums else a
            for i, a in enumerate(args)
        )
        y = chunk_reduction(fun(*flat_args, **kwargs))
        if reduction is None:
            y = _flat_to_device_local(y, local_size, mesh)
        return y

    scan_x = tuple(
        _batch_device_local(a, batch_size, mesh) if i in argnums else a
        for i, a in enumerate(args)
    )
    local_remainder = local_size % batch_size
    y = _scan_device_local_chunks(
        fun,
        argnums,
        reduction,
        chunk_reduction,
        batch_size,
        mesh,
        *scan_x,
        **kwargs,
    )

    if reduction is None:
        y = _unbatch_device_local(y, mesh)

    if local_remainder == 0:
        return y

    chunked_size = local_size - local_remainder
    remain_x = tuple(
        _device_local_remainder(a, chunked_size) if i in argnums else a
        for i, a in enumerate(args)
    )
    flat_remain_x = tuple(
        _flatten_device_local(a, mesh) if i in argnums else a
        for i, a in enumerate(remain_x)
    )
    remain_y = chunk_reduction(fun(*flat_remain_x, **kwargs))
    if reduction is None:
        remain_y = _flat_to_device_local(remain_y, local_remainder, mesh)
        return _concat_device_local(y, remain_y)

    return reduction(y, remain_y)


def _evaluate_sharded(
    fun,
    batch_size,
    argnums,
    reduction,
    chunk_reduction,
    mesh,
    *args,
    **kwargs,
):
    mesh = (
        _make_automatic_mesh(jax.device_count())
        if mesh is None
        else _validate_sharding_mesh(mesh)
    )
    return _evaluate_sharded_on_mesh(
        fun,
        batch_size,
        argnums,
        reduction,
        chunk_reduction,
        mesh,
        *args,
        **kwargs,
    )


def _evaluate_on_first_device(
    fun,
    batch_size,
    argnums,
    reduction,
    chunk_reduction,
    mesh,
    *args,
    **kwargs,
):
    """Evaluate replicated inputs on mesh index zero and broadcast the output."""
    axis_name = _axis_name(mesh)

    def evaluate(args_):
        return _evaluate_in_chunks(
            fun,
            batch_size,
            argnums,
            reduction,
            chunk_reduction,
            False,
            None,
            *args_,
            **kwargs,
        )

    out_struct = jax.eval_shape(evaluate, args)

    def evaluate_on_first(*args_):
        out = jax.lax.cond(
            jax.lax.axis_index(axis_name) == 0,
            evaluate,
            lambda _: tree_map(jnp.zeros_like, out_struct),
            args_,
        )
        # Unlike a sum, the transpose of this gather sends output cotangents
        # back only to the device that evaluated ``fun``. This avoids both
        # duplicate primal work and duplicate gradient contributions.
        return tree_map(
            lambda leaf: jax.lax.all_gather(leaf, axis_name, axis=0, tiled=False)[0],
            out,
        )

    in_specs, out_specs = tree_map(
        lambda _: PartitionSpec(),
        (args, out_struct),
    )
    out = jax.shard_map(
        evaluate_on_first,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        axis_names={axis_name},
        # The gather makes every physical result identical, but its varying
        # transpose is required when this result is joined to sharded output.
        check_vma=False,
    )(*args)
    return out


def _evaluate_sharded_on_mesh(
    fun,
    batch_size,
    argnums,
    reduction,
    chunk_reduction,
    mesh,
    *args,
    **kwargs,
):
    args_shardable, args_remainder = zip(
        *[
            (
                _split_shardable(a, mesh)
                if i in argnums
                else _duplicate_unmapped_explicit(a, mesh)
            )
            for i, a in enumerate(args)
        ]
    )
    n_shardable = tree_leaves(args_shardable[argnums[0]])[0].shape[0]
    n_remainder = tree_leaves(args_remainder[argnums[0]])[0].shape[0]

    if n_shardable == 0:
        return _evaluate_on_first_device(
            fun,
            batch_size,
            argnums,
            reduction,
            chunk_reduction,
            mesh,
            *args_remainder,
            **kwargs,
        )

    # Global sharded layout: the divisible prefix is partitioned over axis 0.
    if batch_size is None:
        out_shardable = chunk_reduction(fun(*args_shardable, **kwargs))
    else:
        args_local = tuple(
            _to_device_local(a, mesh) if i in argnums else a
            for i, a in enumerate(args_shardable)
        )
        out_shardable = _evaluate_device_local_in_chunks(
            fun,
            batch_size,
            argnums,
            reduction,
            chunk_reduction,
            mesh,
            *args_local,
            **kwargs,
        )
        if reduction is None:
            out_shardable = _flatten_device_local(out_shardable, mesh)

    if n_remainder == 0:
        return out_shardable

    out_remainder = _evaluate_on_first_device(
        fun,
        batch_size,
        argnums,
        reduction,
        chunk_reduction,
        mesh,
        *args_remainder,
        **kwargs,
    )
    return (
        _concat_resharded_to_replicated(out_shardable, out_remainder, mesh)
        if reduction is None
        else reduction(out_shardable, out_remainder)
    )


def _evaluate_in_chunks(
    vmapped_fun,
    batch_size,
    argnums,
    reduction=None,
    chunk_reduction=identity,
    shard=False,
    mesh=None,
    *args,
    **kwargs,
):
    warnif(
        shard and not _SUPPORTS_SHARDED_BATCHING,
        RuntimeWarning,
        "shard=True requires JAX 0.10.2 or newer; falling back to ordinary "
        "chunking.",
    )
    if shard and _SUPPORTS_SHARDED_BATCHING:
        return _evaluate_sharded(
            vmapped_fun,
            batch_size,
            argnums,
            reduction,
            chunk_reduction,
            mesh,
            *args,
            **kwargs,
        )

    if batch_size is None:
        return chunk_reduction(vmapped_fun(*args, **kwargs))

    n_elements = tree_leaves(args[argnums[0]])[0].shape[0]
    if n_elements <= batch_size:
        return chunk_reduction(vmapped_fun(*args, **kwargs))

    scan_x, remain_x = zip(
        *[
            _batch_and_remainder(a, batch_size) if i in argnums else (a, a)
            for i, a in enumerate(args)
        ]
    )
    # Note that num_batches in _batch_and_remainder is always positive.
    scan_y = _scanmap(vmapped_fun, argnums, reduction, chunk_reduction)(
        *scan_x, **kwargs
    )
    if reduction is None:
        scan_y = _unchunk(scan_y)

    if n_elements % batch_size == 0:
        return scan_y

    remain_y = chunk_reduction(vmapped_fun(*remain_x, **kwargs))
    if reduction is None:
        return _concat(scan_y, remain_y)

    return reduction(scan_y, remain_y)


def _parse_in_axes(in_axes):
    in_axes = (in_axes,) if isinstance(in_axes, int) else in_axes

    errorif(
        not set(in_axes).issubset((0, None)),
        NotImplementedError,
        f"Only in_axes 0/None are currently supported, but got {in_axes}.",
    )
    argnums = tuple(i for i, axis in enumerate(in_axes) if axis is not None)
    return in_axes, argnums


def _parse_batch_size(batch_size, kwargs):
    unexpected = kwargs.keys() - {"chunk_size"}
    errorif(
        unexpected,
        ValueError,
        f"Unexpected keyword argument(s): {', '.join(sorted(unexpected))}",
    )
    if batch_size is None and "chunk_size" in kwargs:
        batch_size = kwargs["chunk_size"]
    return batch_size


def batch_vmap(
    f,
    /,
    in_axes=0,
    *,
    batch_size=None,
    reduction=None,
    chunk_reduction=identity,
    shard=False,
    mesh=None,
    **kwargs,
):
    """Behaves like ``vmap`` but uses scan to chunk the computations in smaller chunks.

    Warnings
    --------
    - https://github.com/jax-ml/jax/issues/26689
    - https://github.com/jax-ml/jax/issues/27591
    - https://github.com/jax-ml/jax/issues/31919
    - https://docs.jax.dev/en/latest/jep/2026-custom-derivatives.html
      #main-problem-descriptions.
    - Only ``out_axes=0`` is supported.

    See Also
    --------
    batch_map
        If the function supports native vectorization, use ``batch_map`` instead
        for the reasons discussed the docstring.

    Parameters
    ----------
    f : callable
        The function to be vectorised.
    in_axes : int or None
        The axes that should be scanned along. Only supports ``0`` or ``None``.
    batch_size : int or None
        Size to split computation into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``.
        ``chunk_size`` is accepted as an alias when ``batch_size`` is ``None``.
    reduction : callable or None
        Binary reduction operation.
        Should take two arguments and return one output, e.g. ``jnp.add``, and
        must be associative because partial chunk results are combined in an
        implementation-dependent order.
    chunk_reduction : callable
        Chunk-wise reduction operation.
        Should summarize a chunk compatibly with ``reduction`` along the mapped
        axis, e.g. ``jnp.add.reduce``.
    shard : bool
        Whether to shard mapped input data across devices before applying
        chunked batching. The divisible prefix is split across devices; when
        supplied, ``batch_size`` bounds the batches processed on each device. A
        local remainder is evaluated once per device, and a final global
        remainder is evaluated once overall. The mapped length need not be
        divisible by either the device count or ``batch_size``. Default is
        ``False``. If a non-reduced output has a global remainder, the final
        concatenated output is replicated across the mesh.
    mesh : jax.sharding.Mesh or None
        Optional one-dimensional mesh with an ``AxisType.Auto`` axis. Supplying
        a mesh selects the devices and topology used by ``shard=True`` and lets
        already-sharded inputs retain a compatible layout. Default is ``None``,
        which creates a mesh over all available devices.

    Returns
    -------
    f : callable
        A vectorised and chunked function.

    """
    batch_size = _parse_batch_size(batch_size, kwargs)
    in_axes, argnums = _parse_in_axes(in_axes)
    f = vmap(f, in_axes=in_axes)
    if batch_size is None and not shard:
        return lambda *args, **kwargs: chunk_reduction(f(*args, **kwargs))
    return partial(
        _evaluate_in_chunks,
        f,
        batch_size,
        argnums,
        reduction,
        chunk_reduction,
        shard,
        mesh,
    )


def batch_map(
    fun,
    fun_input,
    /,
    batch_size=None,
    *,
    reduction=None,
    chunk_reduction=identity,
    strip_dim0=False,
    shard=False,
    mesh=None,
    **kwargs,
):
    """Compute ``chunk_reduction(fun(fun_input))`` in batches.

    Notes
    -----
    This method does not automatically wrap ``fun`` with ``vmap``.
    Unless ``fun`` is already wrapped with ``vmap``, the leading dimension
    of ``fun_input`` will not be stripped before it is passed into ``fun``.
    This can be inconvenient for nesting calls to ``batch_map``,
    since only batching along the first axis is supported.
    However, the ``strip_dim0`` flag should cover the most common case
    of nesting calls where ``batch_size`` is one on the outermost call.

    For a ``fun`` that accepts scalar and batched inputs elementwise,
    ``batch_map(fun, inputs)`` is expected to match
    ``batch_vmap(fun)(inputs)``. Functions whose results depend on other examples
    in a chunk or on the chunk length do not satisfy this elementwise contract.

    If ``fun`` is natively vectorized, this can be preferable to ``batch_vmap``
    to reduce compilation time, avoid issues such as executing all branches of
    code conditioned on dynamic values, or avoid messing up the behavior of
    jvp's and vjp's under vmap, e.g.
    https://docs.jax.dev/en/latest/jep/
    2026-custom-derivatives.html#main-problem-descriptions.

    Only out axes = 0 is supported.

    See Also
    --------
    batch_vmap
        If the function does not support native vectorization.

    Parameters
    ----------
    fun : callable
        Vectorized function.
    fun_input : pytree
        Data to split into batches to feed to ``fun``.
    batch_size : int or None
        Size of batches. If no batching should be done or the batch size is the
        full input then supply ``None``.
        ``chunk_size`` is accepted as an alias when ``batch_size`` is ``None``.
    reduction : callable or None
        Binary reduction operation.
        Should take two arguments and return one output, e.g. ``jnp.add``, and
        must be associative because partial chunk results are combined in an
        implementation-dependent order.
    chunk_reduction : callable
        Chunk-wise reduction operation.
        Should summarize a chunk compatibly with ``reduction`` along the mapped
        axis, e.g. ``jnp.add.reduce``.
    strip_dim0 : bool
        Whether to strip the leading dim of ``fun_input`` before passing it
        to ``fun``; see notes. This flag only works if ``batch_size`` is one.
        It should be set to ``False`` if ``fun`` is wrapped in ``vmap``.
        Default is ``False``.
    shard : bool
        Whether to shard ``fun_input`` across devices before applying chunked
        batching. The divisible prefix is split across devices; when supplied,
        ``batch_size`` bounds the batches processed on each device. A local
        remainder is evaluated once per device, and a final global remainder is
        evaluated once overall. The input length need not be divisible by either
        the device count or ``batch_size``. Default is ``False``. If a non-reduced
        output has a global remainder, the final concatenated output is replicated
        across the mesh.
    mesh : jax.sharding.Mesh or None
        Optional one-dimensional mesh with an ``AxisType.Auto`` axis. Supplying
        a mesh selects the devices and topology used by ``shard=True`` and lets
        already-sharded inputs retain a compatible layout. Default is ``None``,
        which creates a mesh over all available devices.

    Returns
    -------
    fun_output
        Returns ``chunk_reduction(fun(fun_input))``.

    """
    batch_size = _parse_batch_size(batch_size, kwargs)
    if batch_size is None and not shard:
        return chunk_reduction(fun(fun_input))
    if strip_dim0 and batch_size == 1:
        if shard:
            return _evaluate_in_chunks(
                vmap(fun),
                batch_size,
                (0,),
                reduction,
                chunk_reduction if reduction is not None else identity,
                True,
                mesh,
                fun_input,
            )
        return _scanmap(fun, 0, reduction, identity)(fun_input)

    return _evaluate_in_chunks(
        fun,
        batch_size,
        (0,),
        reduction,
        chunk_reduction,
        shard,
        mesh,
        fun_input,
    )


def batch_vectorize(  # noqa: C901
    pyfunc, *, excluded=frozenset(), signature=None, batch_size=None, **kwargs
):
    """Define a vectorized function with broadcasting and batching.

    References
    ----------
    The original copyright notice is as follows
    Copyright 2020 The JAX Authors.
    Licensed under the Apache License, Version 2.0 (the "License");
    https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/vectorize.py.

    Notes
    -----
    :func:`vectorize` is a convenience wrapper for defining vectorized
    functions with broadcasting, in the style of NumPy's
    `generalized universal functions
    <https://numpy.org/doc/stable/reference/c-api/generalized-ufuncs.html>`_.
    It allows for defining functions that are automatically repeated across
    any leading dimensions, without the implementation of the function needing to
    be concerned about how to handle higher dimensional inputs.

    :func:`jax.numpy.vectorize` has the same interface as
    :class:`numpy.vectorize`, but it is syntactic sugar for an auto-batching
    transformation (:func:`vmap`) rather than a Python loop. This should be
    considerably more efficient, but the implementation must be written in terms
    of functions that act on JAX arrays.

    Parameters
    ----------
    pyfunc : callable
        Function to vectorize.
    excluded : set of int or str, optional
        Positional or keyword arguments that will not be vectorized. These are
        passed directly to ``pyfunc`` without modification.
    signature : str, optional
        Generalized universal function signature, such as ``(m,n),(n)->(m)``
        for vectorized matrix-vector multiplication. If provided, ``pyfunc``
        receives and must return arrays whose shapes match the corresponding
        core dimensions.
    batch_size : int, optional
        Number of mapped elements evaluated per batch. By default, all mapped
        elements are evaluated together, matching :func:`jax.numpy.vectorize`.
        ``chunk_size`` is accepted as an alias when ``batch_size`` is ``None``.

    Returns
    -------
    callable
        Batch-vectorized version of the given function.

    """
    batch_size = _parse_batch_size(batch_size, kwargs)
    errorif(
        any(not isinstance(exclude, (str, int)) for exclude in excluded),
        TypeError,
        "jax.numpy.vectorize can only exclude integer or string arguments, "
        "but excluded={!r}".format(excluded),
    )
    errorif(
        any(isinstance(e, int) and e < 0 for e in excluded),
        msg=f"excluded={excluded!r} contains negative numbers",
    )

    @wraps(pyfunc)
    def wrapped(*args, **kwargs):
        error_context = (
            "on vectorized function with excluded={!r} and "
            "signature={!r}".format(excluded, signature)
        )
        excluded_func, args, kwargs = _apply_excluded(pyfunc, excluded, args, kwargs)

        if signature is not None:
            input_core_dims, output_core_dims = _parse_gufunc_signature(signature)
        else:
            input_core_dims = [()] * len(args)
            output_core_dims = None

        none_args = {i for i, arg in enumerate(args) if arg is None}
        if none_args:
            errorif(
                any(input_core_dims[i] != () for i in none_args),
                msg=f"Cannot pass None at locations {none_args} with {signature=}",
            )
            excluded_func, args, _ = _apply_excluded(excluded_func, none_args, args, {})
            input_core_dims = [
                dim for i, dim in enumerate(input_core_dims) if i not in none_args
            ]

        args = tuple(map(jnp.asarray, args))

        broadcast_shape, dim_sizes = _parse_input_dimensions(
            args, input_core_dims, error_context
        )

        checked_func = (
            excluded_func
            if output_core_dims is None
            else _check_output_dims(
                excluded_func, dim_sizes, output_core_dims, error_context
            )
        )

        # Detect implicit rank promotion.
        if config.numpy_rank_promotion.value != "allow":
            ranks = [
                arg.ndim - len(core_dims)
                for arg, core_dims in zip(args, input_core_dims)
                if arg.ndim != 0
            ]
            if len(set(ranks)) > 1:
                msg = (
                    f"operands with shapes {[arg.shape for arg in args]} require rank"
                    f" promotion for jnp.vectorize function with signature {signature}."
                    " Set the jax_numpy_rank_promotion config option to 'allow' to"
                    " disable this message; for more information, see"
                    " https://docs.jax.dev/en/latest/rank_promotion_warning.html."
                )
                warnif(config.numpy_rank_promotion.value == "warn", msg=msg)
                errorif(config.numpy_rank_promotion.value == "raise", msg=msg)

        # Rather than broadcasting all arguments to full broadcast shapes, prefer
        # expanding dimensions using vmap. By pushing broadcasting
        # into vmap, we can make use of more efficient batching rules for
        # primitives where only some arguments are batched (e.g., for
        # lax_linalg.triangular_solve), and avoid instantiating large broadcasted
        # arrays.

        squeezed_args = []
        rev_filled_shapes = []

        for arg, core_dims in zip(args, input_core_dims):
            noncore_shape = arg.shape[: arg.ndim - len(core_dims)]

            pad_ndim = len(broadcast_shape) - len(noncore_shape)
            filled_shape = pad_ndim * (1,) + noncore_shape
            rev_filled_shapes.append(filled_shape[::-1])

            squeeze_indices = tuple(
                i for i, size in enumerate(noncore_shape) if size == 1
            )
            squeezed_arg = jnp.squeeze(arg, axis=squeeze_indices)
            squeezed_args.append(squeezed_arg)

        vectorized_func = checked_func
        dims_to_expand = []
        for negdim, axis_sizes in enumerate(zip(*rev_filled_shapes)):
            in_axes = tuple(None if size == 1 else 0 for size in axis_sizes)
            if all(axis is None for axis in in_axes):
                dims_to_expand.append(len(broadcast_shape) - 1 - negdim)
            else:
                vectorized_func = batch_vmap(
                    vectorized_func, in_axes, batch_size=batch_size
                )
        result = vectorized_func(*squeezed_args)

        if not dims_to_expand:
            return result
        elif isinstance(result, tuple):
            return tuple(jnp.expand_dims(r, axis=dims_to_expand) for r in result)
        else:
            return jnp.expand_dims(result, axis=dims_to_expand)

    return wrapped


def batch_jacfwd(
    fun,
    argnums=0,
    has_aux=False,
    holomorphic=False,
    *,
    batch_size=None,
    **kwargs,
):
    """Jacobian of ``fun`` evaluated column-by-column using forward-mode AD.

    Refrences
    ---------
    The original copyright notice is as follows
    Copyright 2018 The JAX Authors.
    Licensed under the Apache License, Version 2.0 (the "License");
    https://github.com/jax-ml/jax/blob/main/jax/_src/api.py.

    Parameters
    ----------
    fun: callable
        Function whose Jacobian is to be computed.
    argnums: Optional, integer or sequence of integers.
        Specifies which positional argument(s) to differentiate with respect to
        (default ``0``).
    has_aux: Optional, bool.
        Indicates whether ``fun`` returns a pair where the first element is considered
        the output of the mathematical function to be differentiated and the second
        element is auxiliary data. Default False.
    holomorphic: Optional, bool.
        Indicates whether ``fun`` is promised to be holomorphic. Default False.
    batch_size: int
        The size of the batches to pass to vmap. If None, defaults to the largest
        possible batch_size.
        ``chunk_size`` is accepted as an alias when ``batch_size`` is ``None``.

    Returns
    -------
    jac: callable
        A function with the same arguments as ``fun``, that evaluates the Jacobian of
        ``fun`` using forward-mode automatic differentiation. If ``has_aux`` is True
        then a pair of (jacobian, auxiliary_data) is returned.

    """
    batch_size = _parse_batch_size(batch_size, kwargs)
    check_callable(fun)
    argnums = _ensure_index(argnums)

    docstr = (
        "Jacobian of {fun} with respect to positional argument(s) "
        "{argnums}. Takes the same arguments as {fun} but returns the "
        "jacobian of the output with respect to the arguments at "
        "positions {argnums}."
    )

    @wraps(fun, docstr=docstr, argnums=argnums)
    def jacfun(*args, **kwargs):
        f_partial, dyn_args = argnums_partial2(fun, argnums, args, kwargs)
        tree_map(partial(_check_input_dtype_jacfwd, holomorphic), dyn_args)
        pushfwd = partial(jax.jvp, f_partial, dyn_args, has_aux=has_aux)
        if has_aux:
            y, jac, aux = batch_vmap(pushfwd, batch_size=batch_size)(
                _std_basis(dyn_args)
            )
            aux = tree_map(lambda x: x[0], aux)
        else:
            y, jac = batch_vmap(pushfwd, batch_size=batch_size)(_std_basis(dyn_args))
            aux = None

        y = tree_map(lambda x: x[0], y)
        jac = tree_map(lambda x: jnp.moveaxis(x, 0, -1), jac)
        tree_map(partial(_check_output_dtype_jacfwd, holomorphic), y)
        example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
        jac_tree = tree_map(partial(_jacfwd_unravel, example_args), y, jac)
        return (jac_tree, aux) if has_aux else jac_tree

    return jacfun


def batch_jacrev(
    fun,
    argnums=0,
    has_aux=False,
    holomorphic=False,
    allow_int=False,
    *,
    batch_size=None,
    **kwargs,
):
    """Jacobian of ``fun`` evaluated row-by-row using reverse-mode AD.

    Refrences
    ---------
    The original copyright notice is as follows
    Copyright 2018 The JAX Authors.
    Licensed under the Apache License, Version 2.0 (the "License");
    https://github.com/jax-ml/jax/blob/main/jax/_src/api.py.

    Parameters
    ----------
    fun: callable
        Function whose Jacobian is to be computed.
    argnums: Optional, integer or sequence of integers.
        Specifies which positional argument(s) to differentiate with respect to
        (default ``0``).
    has_aux: Optional, bool.
        Indicates whether ``fun`` returns a pair where the first element is considered
        the output of the mathematical function to be differentiated and the second
        element is auxiliary data. Default False.
    holomorphic: Optional, bool.
        Indicates whether ``fun`` is promised to be holomorphic. Default False.
    allow_int: Optional, bool.
        Whether to allow differentiating with respect to integer valued inputs. The
        gradient of an integer input will have a trivial vector-space dtype (float0).
        Default False.
    batch_size: int
        The size of the batches to pass to vmap. If None, defaults to the largest
        possible batch_size.
        ``chunk_size`` is accepted as an alias when ``batch_size`` is ``None``.

    Returns
    -------
    jac: callable
        A function with the same arguments as ``fun``, that evaluates the Jacobian of
        ``fun`` using reverse-mode automatic differentiation. If ``has_aux`` is True
        then a pair of (jacobian, auxiliary_data) is returned.

    """
    batch_size = _parse_batch_size(batch_size, kwargs)
    check_callable(fun)
    argnums = _ensure_index(argnums)

    docstr = (
        "Jacobian of {fun} with respect to positional argument(s) "
        "{argnums}. Takes the same arguments as {fun} but returns the "
        "jacobian of the output with respect to the arguments at "
        "positions {argnums}."
    )

    @wraps(fun, docstr=docstr, argnums=argnums)
    def jacfun(*args, **kwargs):
        f_partial, dyn_args = argnums_partial2(fun, argnums, args, kwargs)
        tree_map(partial(_check_input_dtype_jacrev, holomorphic, allow_int), dyn_args)
        y, pullback, *maybe_aux = jax.vjp(f_partial, *dyn_args, has_aux=has_aux)
        tree_map(partial(_check_output_dtype_jacrev, holomorphic), y)
        jac = batch_vmap(pullback, batch_size=batch_size)(_std_basis(y))
        jac = jac[0] if isinstance(argnums, int) else jac
        example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
        jac_tree = tree_map(partial(_jacrev_unravel, y), example_args, jac)
        jac_tree = tree_transpose(
            tree_structure(example_args), tree_structure(y), jac_tree
        )
        return (jac_tree, *maybe_aux) if has_aux else jac_tree

    return jacfun
