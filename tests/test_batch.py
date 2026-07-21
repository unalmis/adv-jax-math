# Copyright (C) 2024 Kaya Unalmis
# SPDX-License-Identifier: LGPL-3.0

"""Tests for batching functions."""

import importlib.util
import os
import subprocess
import sys
import textwrap
from contextlib import nullcontext

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from packaging import version

from adv_jax_math._batch import (
    _SUPPORTS_SHARDED_BATCHING,
    batch_jacfwd,
    batch_jacrev,
    batch_map,
    batch_vectorize,
    batch_vmap,
)

_CALLER_MESH_KWARGS = (
    {"axis_types": (jax.sharding.AxisType.Auto,)}
    if version.parse("0.8.1") <= version.parse(jax.__version__) < version.parse("0.9.0")
    else {}
)


def _run_forced_cpu_devices(code, num_devices=4):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_devices}"
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        env=env,
    )


def _assert_tree_allclose(actual, expected):
    assert eqx.tree_equal(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "reduce_output"),
    (
        pytest.param({}, False, id="unbatched"),
        pytest.param({"batch_size": 2}, False, id="chunked-remainder"),
        pytest.param({"batch_size": 5}, False, id="full-batch"),
        pytest.param(
            {"batch_size": 1, "strip_dim0": True},
            False,
            id="stripped",
        ),
        pytest.param(
            {
                "batch_size": 2,
                "reduction": jnp.add,
                "chunk_reduction": jnp.sum,
            },
            True,
            id="reduced",
        ),
        pytest.param(
            {"shard": True},
            False,
            id="sharded",
        ),
        pytest.param(
            {"batch_size": 2, "shard": True},
            False,
            id="sharded-chunked",
        ),
        pytest.param(
            {"batch_size": 8, "shard": True},
            False,
            id="sharded-full-batch",
        ),
        pytest.param(
            {
                "batch_size": 1,
                "strip_dim0": True,
                "shard": True,
            },
            False,
            id="sharded-stripped",
        ),
        pytest.param(
            {
                "batch_size": 2,
                "reduction": jnp.add,
                "chunk_reduction": jnp.sum,
                "shard": True,
            },
            True,
            id="sharded-reduced",
        ),
        pytest.param(
            {
                "batch_size": 8,
                "reduction": jnp.add,
                "chunk_reduction": jnp.sum,
                "shard": True,
            },
            True,
            id="sharded-full-batch-reduced",
        ),
    ),
)
def test_batch_map_modes(kwargs, reduce_output):
    """Batching modes should preserve values and requested reductions."""
    x = jnp.arange(5.0)
    expected = jnp.sum(x + 1) if reduce_output else x + 1
    warning = (
        pytest.warns(RuntimeWarning, match="requires JAX 0.10.2 or newer")
        if kwargs.get("shard") and not _SUPPORTS_SHARDED_BATCHING
        else nullcontext()
    )
    with warning:
        actual = batch_map(lambda y: y + 1, x, **kwargs)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.unit
@pytest.mark.parametrize("batch_size", (None, 1, 2, 8))
def test_batch_vmap_matches_vmap(batch_size):
    """Batch sizes should not change mapped results with static arguments."""
    x = jnp.arange(5.0)
    actual = batch_vmap(
        lambda value, scale: value * scale,
        in_axes=(0, None),
        batch_size=batch_size,
    )(x, 3.0)
    np.testing.assert_allclose(actual, x * 3)


@pytest.mark.unit
def test_batch_vmap_reduction():
    """Chunk and cross-chunk reductions should compose."""
    x = jnp.arange(5.0)
    actual = batch_vmap(
        lambda value: value**2,
        batch_size=2,
        reduction=jnp.add,
        chunk_reduction=jnp.sum,
    )(x)
    np.testing.assert_allclose(actual, jnp.sum(x**2))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("reduction", "chunk_reduction", "values", "expected"),
    (
        pytest.param(
            jnp.multiply,
            jnp.prod,
            jnp.arange(1.0, 6.0),
            120.0,
            id="product",
        ),
        pytest.param(
            jnp.maximum,
            jnp.max,
            -jnp.arange(1.0, 6.0),
            -1.0,
            id="negative-maximum",
        ),
    ),
)
def test_chunk_reduction_does_not_assume_zero_identity(
    reduction,
    chunk_reduction,
    values,
    expected,
):
    """Cross-chunk reductions should seed from the first computed chunk."""
    vmapped = batch_vmap(
        lambda value: value,
        batch_size=2,
        reduction=reduction,
        chunk_reduction=chunk_reduction,
    )(values)
    mapped = batch_map(
        lambda value: value,
        values,
        batch_size=2,
        reduction=reduction,
        chunk_reduction=chunk_reduction,
    )

    np.testing.assert_allclose(vmapped, expected)
    np.testing.assert_allclose(mapped, expected)


@pytest.mark.unit
def test_chunk_size_alias_and_batch_size_precedence():
    """The legacy name should be used only when batch_size remains unset."""
    x = jnp.arange(6.0)

    def fill_with_batch_size(values):
        return jnp.full_like(values, values.shape[0])

    np.testing.assert_allclose(
        batch_map(fill_with_batch_size, x, chunk_size=2),
        jnp.full_like(x, 2),
    )
    np.testing.assert_allclose(
        batch_map(fill_with_batch_size, x, batch_size=3, chunk_size=2),
        jnp.full_like(x, 3),
    )


@pytest.mark.unit
def test_public_batch_transforms_accept_chunk_size():
    """All public batching transforms should accept the legacy keyword."""
    x = jnp.arange(5.0)
    expected = x**2 + 1

    np.testing.assert_allclose(
        batch_vmap(lambda value: value**2 + 1, chunk_size=2)(x), expected
    )
    np.testing.assert_allclose(
        batch_vectorize(lambda value: value**2 + 1, chunk_size=2)(x), expected
    )

    jac_input = jnp.array([1.0, 2.0, 3.0])
    fun = lambda value: jnp.array([value[0] * value[1], jnp.sin(value[2])])
    np.testing.assert_allclose(
        batch_jacfwd(fun, chunk_size=2)(jac_input),
        jax.jacfwd(fun)(jac_input),
    )
    np.testing.assert_allclose(
        batch_jacrev(fun, chunk_size=1)(jac_input),
        jax.jacrev(fun)(jac_input),
    )


@pytest.mark.unit
def test_batch_size_apis_reject_unexpected_keywords():
    """The compatibility kwargs should not swallow misspelled arguments."""
    with pytest.raises(ValueError, match="Unexpected keyword argument.*batch_sze"):
        batch_vmap(lambda value: value, batch_sze=2)


@pytest.mark.unit
def test_shard_input_data_is_not_supported():
    """The renamed sharding option should not retain a compatibility alias."""
    with pytest.raises(ValueError, match="Unexpected keyword.*shard_input_data"):
        batch_vmap(lambda value: value, shard_input_data=True)


@pytest.mark.unit
def test_batch_vmap_rejects_unsupported_axes():
    """Only mapped axis zero and unmapped inputs are supported."""
    with pytest.raises(NotImplementedError, match="Only in_axes 0/None"):
        batch_vmap(lambda value: value, in_axes=1)


@pytest.mark.unit
@pytest.mark.parametrize("batch_size", (None, 2))
def test_sharded_batching_falls_back_when_unsupported(monkeypatch, batch_size):
    """Older JAX installations should report the ordinary-chunking fallback."""
    import adv_jax_math._batch as batch_module

    monkeypatch.setattr(batch_module, "_SUPPORTS_SHARDED_BATCHING", False)
    x = jnp.arange(5.0)
    with pytest.warns(RuntimeWarning, match="requires JAX 0.10.2 or newer"):
        actual = batch_vmap(
            lambda value: value + 1,
            batch_size=batch_size,
            shard=True,
        )(x)
    np.testing.assert_allclose(actual, x + 1)


@pytest.mark.unit
def test_chunked_batching_preserves_caller_sharding():
    """Ordinary chunking should not introduce a library-owned mesh axis."""
    from jax.sharding import NamedSharding, PartitionSpec

    mesh = jax.make_mesh(
        (1,),
        ("caller",),
        devices=jax.devices()[:1],
        **_CALLER_MESH_KWARGS,
    )
    sharding = NamedSharding(mesh, PartitionSpec())
    x = jax.device_put(jnp.arange(8.0), sharding)

    actual = batch_vmap(lambda value: value + 1, batch_size=2)(x)

    np.testing.assert_allclose(actual, x + 1)
    assert actual.sharding.mesh == mesh
    assert actual.sharding.is_fully_replicated


@pytest.mark.unit
@pytest.mark.skipif(
    not _SUPPORTS_SHARDED_BATCHING,
    reason="Caller-provided batching meshes require JAX 0.10.2 or newer",
)
def test_sharded_batching_validates_caller_mesh():
    """Only one-dimensional automatic meshes should be accepted."""
    from jax.sharding import AxisType

    x = jnp.arange(4.0)
    devices = jax.devices()[:1]
    explicit_mesh = jax.make_mesh(
        (1,),
        ("explicit",),
        devices=devices,
        axis_types=(AxisType.Explicit,),
    )
    with pytest.raises(ValueError, match="AxisType.Auto"):
        batch_map(lambda values: values + 1, x, shard=True, mesh=explicit_mesh)

    two_dimensional_mesh = jax.make_mesh(
        (1, 1),
        ("rows", "columns"),
        devices=devices,
        axis_types=(AxisType.Auto, AxisType.Auto),
    )
    with pytest.raises(ValueError, match="one-dimensional"):
        batch_map(
            lambda values: values + 1,
            x,
            shard=True,
            mesh=two_dimensional_mesh,
        )

    auto_mesh = jax.make_mesh(
        (1,),
        ("data",),
        devices=devices,
        axis_types=(AxisType.Auto,),
    )
    actual = batch_map(
        lambda values: values + 1,
        x,
        batch_size=2,
        shard=True,
        mesh=auto_mesh,
    )
    np.testing.assert_allclose(actual, x + 1)
    assert actual.sharding.mesh == auto_mesh


@pytest.mark.unit
@pytest.mark.skipif(
    not _SUPPORTS_SHARDED_BATCHING,
    reason="Explicit sharding requires JAX 0.10.2 or newer",
)
def test_sharded_batching_preserves_explicit_mapped_and_unmapped_inputs():
    """Explicit inputs should retain values and cotangents through conversion."""
    from jax.sharding import AxisType, NamedSharding, PartitionSpec

    caller_mesh = jax.make_mesh(
        (1,),
        ("caller",),
        devices=jax.devices()[:1],
        axis_types=(AxisType.Explicit,),
    )
    sharding = NamedSharding(caller_mesh, PartitionSpec("caller"))
    x = jax.device_put(jnp.arange(5.0), sharding)
    weights = jax.device_put(jnp.arange(3.0), sharding)

    def fun(values, context):
        return batch_vmap(
            lambda value, config: (
                value + jnp.sum(config["weights"]) + config["offset"]
            ),
            in_axes=(0, None),
            batch_size=2,
            shard=True,
        )(values, context)

    context = {"weights": weights, "offset": 2.0}
    expected = x + jnp.sum(weights) + 2.0
    np.testing.assert_allclose(fun(x, context), expected)

    grad_x, grad_weights = jax.grad(
        lambda values, weights_: jnp.sum(
            fun(values, {"weights": weights_, "offset": 2.0})
        ),
        argnums=(0, 1),
    )(x, weights)
    np.testing.assert_allclose(grad_x, jnp.ones_like(x))
    np.testing.assert_allclose(grad_weights, x.size * jnp.ones_like(weights))


@pytest.mark.unit
def test_explicit_split_partition_logic(monkeypatch):
    """Explicit prefixes and remainders should partition every input exactly."""
    import adv_jax_math._batch as batch_module

    class Mesh:
        size = 4
        axis_names = ("data",)

    resharded_shapes = []
    monkeypatch.setattr(batch_module, "_has_explicit_sharding", lambda _: True)
    monkeypatch.setattr(batch_module, "NamedSharding", lambda *_: object())
    monkeypatch.setattr(batch_module.jax, "device_put", lambda value, _: value)

    def reshard(value, _mesh):
        resharded_shapes.append(value.shape)
        return value

    monkeypatch.setattr(batch_module, "_reshard_leaf_to_replicated", reshard)

    for size, expected_resharded_shapes in (
        (12, []),
        (13, [(1,)]),
        (3, [(3,)]),
    ):
        values = np.arange(size, dtype=np.float32)
        prefix, remainder = batch_module._shard(values, Mesh())
        combined = np.concatenate((np.asarray(prefix), np.asarray(remainder)))
        np.testing.assert_allclose(combined, values)
        assert prefix.shape[0] % Mesh.size == 0
        assert resharded_shapes == expected_resharded_shapes
        resharded_shapes.clear()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("split_at", "reduction", "chunk_reduction"),
    (
        pytest.param(0, None, lambda value: value, id="remainder-only"),
        pytest.param(4, None, lambda value: value, id="concatenated-remainder"),
        pytest.param(4, jnp.add, jnp.sum, id="reduced-remainder"),
    ),
)
def test_sharded_evaluation_combines_remainders(
    monkeypatch,
    split_at,
    reduction,
    chunk_reduction,
):
    """Sharded prefixes and globally evaluated remainders should recombine."""
    import adv_jax_math._batch as batch_module

    if not batch_module._SUPPORTS_SHARDED_BATCHING:
        pytest.skip("Sharded batching requires JAX 0.10.2 or newer")

    def split(value, _mesh):
        return value[:split_at], value[split_at:]

    monkeypatch.setattr(batch_module, "_split_shardable", split)
    mesh = batch_module._make_automatic_mesh(1)
    x = jnp.arange(5.0)
    actual = batch_module._evaluate_sharded_on_mesh(
        lambda value: value + 1,
        2,
        (0,),
        reduction,
        chunk_reduction,
        mesh,
        x,
    )
    expected = x + 1 if reduction is None else jnp.sum(x + 1)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.unit
def test_batched_jacobians_match_jax():
    """Batched forward- and reverse-mode Jacobians should match JAX."""
    x = jnp.array([1.0, 2.0, 3.0])

    def fun(y):
        return jnp.array([y[0] * y[1], jnp.sin(y[2])])

    np.testing.assert_allclose(
        batch_jacfwd(fun, batch_size=2)(x),
        jax.jacfwd(fun)(x),
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        batch_jacrev(fun, batch_size=1)(x),
        jax.jacrev(fun)(x),
        rtol=1e-6,
        atol=1e-7,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("batch_jacobian", "jax_jacobian"),
    ((batch_jacfwd, jax.jacfwd), (batch_jacrev, jax.jacrev)),
)
def test_batched_jacobians_multiple_arguments_and_aux(batch_jacobian, jax_jacobian):
    """Public JVP/VJP paths should preserve pytrees, argnums, and auxiliary data."""
    x = jnp.array([1.0, 2.0])
    y = jnp.array([3.0, 4.0])

    def fun(a, b, scale):
        output = {
            "product": scale * a * b,
            "sum": jnp.sum(a**2 + b),
        }
        return output, jnp.sum(a - b)

    kwargs = {"argnums": (0, 1), "has_aux": True}
    got = batch_jacobian(fun, batch_size=2, **kwargs)(x, y, 0.5)
    expected = jax_jacobian(fun, **kwargs)(x, y, 0.5)

    _assert_tree_allclose(got, expected)


@pytest.mark.unit
def test_older_jax_fallbacks(monkeypatch):
    """Compatibility fallbacks should preserve batching and Jacobian results."""
    from jax._src import api_util

    import adv_jax_math._batch as batch_module

    module_name = "adv_jax_math._batch_fallback_test"
    spec = importlib.util.spec_from_file_location(module_name, batch_module.__file__)
    assert spec is not None and spec.loader is not None
    fallback_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = fallback_module
    try:
        with monkeypatch.context() as patch:
            patch.delattr(api_util, "argnums_partial2", raising=False)
            patch.delattr(jax.sharding, "reshard", raising=False)
            spec.loader.exec_module(fallback_module)

        assert not fallback_module._SUPPORTS_SHARDED_BATCHING

        x = jnp.arange(5.0)
        with pytest.warns(RuntimeWarning, match="requires JAX 0.10.2 or newer"):
            actual = fallback_module.batch_vmap(
                lambda value: value + 1,
                batch_size=2,
                shard=True,
            )(x)
        np.testing.assert_allclose(actual, x + 1)

        def fun(x, y, *, scale):
            return scale * jnp.array([x * y, x + 2 * y])

        x, y = jnp.array(2.0), jnp.array(3.0)
        kwargs = {"argnums": (-2, -1), "batch_size": 1}
        expected_kwargs = {"argnums": (-2, -1)}
        np.testing.assert_allclose(
            fallback_module.batch_jacfwd(fun, **kwargs)(x, y, scale=0.5),
            jax.jacfwd(fun, **expected_kwargs)(x, y, scale=0.5),
        )
        np.testing.assert_allclose(
            fallback_module.batch_jacrev(fun, **kwargs)(x, y, scale=0.5),
            jax.jacrev(fun, **expected_kwargs)(x, y, scale=0.5),
        )
    finally:
        sys.modules.pop(module_name, None)


@pytest.mark.unit
def test_batch_vectorize_with_and_without_signature():
    """Batched vectorization should support scalar and gufunc signatures."""
    x = jnp.arange(5.0)
    scalar_fun = batch_vectorize(lambda y: y**2 + 1, batch_size=2)
    np.testing.assert_allclose(scalar_fun(x), x**2 + 1)

    dot = batch_vectorize(
        lambda y, z: y @ z,
        signature="(n),(n)->()",
        batch_size=2,
    )
    y = jnp.arange(12.0).reshape(4, 3)
    z = jnp.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(dot(y, z), y @ z)


@pytest.mark.unit
@pytest.mark.parametrize("batch_size", (None, 1, 2, 4))
def test_batch_vectorize_matches_jax_broadcasting(batch_size):
    """Batching should be the only difference from jax.numpy.vectorize."""

    def fun(x, y):
        return jnp.sin(x) + y**2

    x = jnp.arange(2.0).reshape(2, 1)
    y = jnp.arange(3.0)
    expected = jnp.vectorize(fun)(x, y)
    actual = batch_vectorize(fun, batch_size=batch_size)(x, y)
    _assert_tree_allclose(actual, expected)


@pytest.mark.unit
def test_batch_vectorize_matches_jax_special_arguments_and_outputs():
    """Excluded and None arguments and tuple gufunc outputs should match JAX."""

    def polynomial(x, power, *, offset):
        return x**power + offset

    excluded = frozenset((1, "offset"))
    x = jnp.arange(5.0)
    expected = jnp.vectorize(polynomial, excluded=excluded)(x, 2, offset=3)
    actual = batch_vectorize(polynomial, excluded=excluded, batch_size=2)(
        x, 2, offset=3
    )
    _assert_tree_allclose(actual, expected)

    def maybe_shift(value, shift):
        return value if shift is None else value + shift

    expected = jnp.vectorize(maybe_shift)(x, None)
    actual = batch_vectorize(maybe_shift, batch_size=2)(x, None)
    _assert_tree_allclose(actual, expected)

    def statistics(values):
        return jnp.sum(values), jnp.max(values)

    values = jnp.arange(12.0).reshape(4, 3)
    signature = "(n)->(),()"
    expected = jnp.vectorize(statistics, signature=signature)(values)
    actual = batch_vectorize(statistics, signature=signature, batch_size=2)(values)
    _assert_tree_allclose(actual, expected)


@pytest.mark.unit
@pytest.mark.parametrize(
    "excluded",
    (
        pytest.param(frozenset({1.5}), id="non-string-or-integer"),
        pytest.param(frozenset({-1}), id="negative-integer"),
    ),
)
def test_batch_vectorize_rejects_invalid_exclusions(excluded):
    """Excluded arguments should follow the jax.numpy.vectorize contract."""
    with pytest.raises((TypeError, ValueError)):
        batch_vectorize(jnp.add, excluded=excluded)


@pytest.mark.unit
def test_batch_vectorize_rejects_none_core_argument():
    """None is only valid for arguments without core dimensions."""
    vectorized = batch_vectorize(lambda value: value, signature="(n)->()")
    with pytest.raises(ValueError, match="Cannot pass None"):
        vectorized(None)


@pytest.mark.unit
@pytest.mark.parametrize(
    "fun",
    (
        pytest.param(lambda value: value + 1, id="array-output"),
        pytest.param(lambda value: (value + 1, value - 1), id="tuple-output"),
    ),
)
def test_batch_vectorize_preserves_singleton_dimensions(fun):
    """Singleton mapped dimensions should be restored after evaluation."""
    x = jnp.ones((1,))
    expected = jnp.vectorize(fun)(x)
    actual = batch_vectorize(fun, batch_size=1)(x)
    _assert_tree_allclose(actual, expected)


@pytest.mark.unit
def test_batch_vectorize_matches_jax_rank_promotion_policy():
    """Implicit rank-promotion errors should match jax.numpy.vectorize."""
    x = jnp.ones((2, 3))
    y = jnp.ones((3,))

    with jax.numpy_rank_promotion("raise"):
        with pytest.raises(ValueError, match="require rank promotion"):
            jnp.vectorize(jnp.add)(x, y)
        with pytest.raises(ValueError, match="require rank promotion"):
            batch_vectorize(jnp.add, batch_size=2)(x, y)

    with jax.numpy_rank_promotion("warn"):
        with pytest.warns(UserWarning, match="require rank promotion"):
            expected = jnp.vectorize(jnp.add)(x, y)
        with pytest.warns(UserWarning, match="require rank promotion"):
            actual = batch_vectorize(jnp.add, batch_size=2)(x, y)
    _assert_tree_allclose(actual, expected)

    same_rank = jnp.arange(6.0).reshape(2, 3)
    with jax.numpy_rank_promotion("raise"):
        expected = jnp.vectorize(jnp.add)(x, same_rank)
        actual = batch_vectorize(jnp.add, batch_size=2)(x, same_rank)
    _assert_tree_allclose(actual, expected)


@pytest.mark.unit
def test_sharded_chunked_batching():
    """Test chunked batching with sharded input data."""
    _run_forced_cpu_devices("""
        from collections import Counter

        import numpy as np

        import jax
        import jax.numpy as jnp
        from jax.sharding import NamedSharding, PartitionSpec
        from packaging import version

        from adv_jax_math._batch import batch_map, batch_vmap

        assert jax.device_count() == 4
        x = jnp.arange(13.0)

        cases = [
            (
                lambda y: batch_map(
                    lambda z: z + 1,
                    y,
                    batch_size=2,
                    shard=True,
                ),
                x + 1,
            ),
            (
                lambda y: batch_map(lambda z: z + 1, y, shard=True),
                x + 1,
            ),
            (
                lambda y: batch_map(
                    lambda z: z + 1,
                    y,
                    batch_size=1,
                    strip_dim0=True,
                    shard=True,
                ),
                x + 1,
            ),
            (
                lambda y: batch_vmap(
                    lambda z, scale: z * scale,
                    in_axes=(0, None),
                    batch_size=2,
                    shard=True,
                )(y, 3.0),
                x * 3,
            ),
            (
                lambda y: batch_vmap(
                    lambda z, scale: z * scale,
                    in_axes=(0, None),
                    shard=True,
                )(y, 3.0),
                x * 3,
            ),
            (
                lambda y: batch_map(
                    lambda z: z,
                    y,
                    batch_size=2,
                    reduction=jnp.add,
                    chunk_reduction=jnp.sum,
                    shard=True,
                ),
                jnp.sum(x),
            ),
        ]
        for fun, expected in cases:
            np.testing.assert_allclose(fun(x), expected)
            np.testing.assert_allclose(jax.jit(fun)(x), expected)

        two_inputs = lambda y, z: batch_vmap(
            lambda a, b: a - b,
            in_axes=(0, 0),
            batch_size=2,
            shard=True,
        )(y, z)
        np.testing.assert_allclose(two_inputs(x, x[::-1]), x - x[::-1])
        np.testing.assert_allclose(jax.jit(two_inputs)(x, x[::-1]), x - x[::-1])

        def elementwise(z):
            return jnp.sin(z) + z**2

        for batch_size in (None, 1, 2, 5):
            mapped = lambda y: batch_map(
                elementwise,
                y,
                batch_size=batch_size,
                shard=True,
            )
            vmapped = batch_vmap(
                elementwise,
                batch_size=batch_size,
                shard=True,
            )
            np.testing.assert_allclose(mapped(x), vmapped(x))
            np.testing.assert_allclose(jax.jit(mapped)(x), jax.jit(vmapped)(x))

        small_x = jnp.arange(3.0)
        small_fun = lambda y: batch_vmap(
            lambda z: z**2,
            batch_size=2,
            shard=True,
        )(y)
        np.testing.assert_allclose(small_fun(small_x), small_x**2)
        np.testing.assert_allclose(jax.jit(small_fun)(small_x), small_x**2)
        np.testing.assert_allclose(
            jax.grad(lambda y: jnp.sum(small_fun(y)))(small_x),
            2 * small_x,
        )

        if version.parse(jax.__version__) >= version.parse("0.10.2"):
            caller_mesh = jax.make_mesh((4,), ("caller",))
            explicit_small_x = jax.device_put(
                small_x,
                NamedSharding(caller_mesh, PartitionSpec()),
            )
            np.testing.assert_allclose(
                small_fun(explicit_small_x),
                explicit_small_x**2,
            )
            np.testing.assert_allclose(
                jax.jit(small_fun)(explicit_small_x),
                explicit_small_x**2,
            )
            np.testing.assert_allclose(
                jax.grad(lambda y: jnp.sum(small_fun(y)))(explicit_small_x),
                2 * explicit_small_x,
            )

            explicit_context = jax.device_put(
                jnp.arange(4.0),
                NamedSharding(caller_mesh, PartitionSpec("caller")),
            )
            with_context = lambda y, context: batch_vmap(
                lambda value, weights: value + jnp.sum(weights),
                in_axes=(0, None),
                batch_size=2,
                shard=True,
            )(y, context)
            expected = x + jnp.sum(explicit_context)
            np.testing.assert_allclose(with_context(x, explicit_context), expected)
            np.testing.assert_allclose(
                jax.jit(with_context)(x, explicit_context),
                expected,
            )
            grad_x, grad_context = jax.grad(
                lambda y, context: jnp.sum(with_context(y, context)),
                argnums=(0, 1),
            )(x, explicit_context)
            np.testing.assert_allclose(grad_x, jnp.ones_like(x))
            np.testing.assert_allclose(
                grad_context,
                x.size * jnp.ones_like(explicit_context),
            )

            calls = []

            def record(local, *, global_shape):
                calls.append((global_shape, tuple(local.shape)))

            def tracked(z):
                global_shape = tuple(z.shape)
                jax.debug.callback(partitioned=True)(
                    lambda local, global_shape=global_shape: record(
                        local, global_shape=global_shape
                    ),
                    z,
                )
                return z + 1

            execution_x = jnp.arange(35.0)
            execution_fun = lambda y: batch_map(
                tracked,
                y,
                batch_size=3,
                shard=True,
            )
            actual = jax.jit(execution_fun)(execution_x)
            actual.block_until_ready()
            np.testing.assert_allclose(actual, execution_x + 1)
            assert Counter(calls) == Counter({
                ((12,), (3,)): 8,
                ((8,), (2,)): 4,
                ((3,), (3,)): 1,
            })
        """)


@pytest.mark.unit
@pytest.mark.skipif(
    not _SUPPORTS_SHARDED_BATCHING,
    reason="Caller-provided batching meshes require JAX 0.10.2 or newer",
)
def test_sharded_batching_accepts_caller_auto_mesh():
    """A caller mesh should select devices and retain compatible input layout."""
    _run_forced_cpu_devices("""
        import numpy as np

        import jax
        import jax.numpy as jnp
        from jax.sharding import AxisType, NamedSharding, PartitionSpec

        from adv_jax_math._batch import batch_map, batch_vmap

        assert jax.device_count() == 4
        mesh = jax.make_mesh(
            (2,),
            ("data",),
            devices=jax.devices()[:2],
            axis_types=(AxisType.Auto,),
        )
        input_sharding = NamedSharding(mesh, PartitionSpec("data"))
        x = jax.device_put(jnp.arange(12.0), input_sharding)

        vmapped = batch_vmap(
            lambda value: value**2,
            batch_size=2,
            shard=True,
            mesh=mesh,
        )(x)
        mapped = batch_map(
            lambda values: values**2,
            x,
            batch_size=2,
            shard=True,
            mesh=mesh,
        )

        np.testing.assert_allclose(mapped, vmapped)
        np.testing.assert_allclose(mapped, x**2)
        assert mapped.sharding.mesh == mesh
        assert vmapped.sharding.mesh == mesh
        assert not mapped.sharding.is_fully_replicated
        assert not vmapped.sharding.is_fully_replicated
        """)


@pytest.mark.unit
@pytest.mark.skipif(
    not _SUPPORTS_SHARDED_BATCHING,
    reason="Explicit sharding preparation requires JAX 0.10.2 or newer",
)
def test_explicit_input_replicates_only_global_remainder():
    """Preparing explicit input should not replicate its divisible prefix."""
    _run_forced_cpu_devices("""
        import numpy as np

        import jax
        import jax.numpy as jnp
        from jax.sharding import NamedSharding, PartitionSpec

        import adv_jax_math._batch as batch_module

        assert jax.device_count() == 4
        caller_mesh = jax.make_mesh((4,), ("caller",))
        batching_mesh = batch_module._make_automatic_mesh(4)
        original_reshard = batch_module._reshard_leaf_to_replicated
        resharded_shapes = []

        def record_reshard(leaf, mesh):
            resharded_shapes.append(leaf.shape)
            return original_reshard(leaf, mesh)

        batch_module._reshard_leaf_to_replicated = record_reshard
        try:
            even = jax.device_put(
                jnp.arange(12.0),
                NamedSharding(caller_mesh, PartitionSpec("caller")),
            )
            prefix, remainder = batch_module._split_shardable(even, batching_mesh)
            assert not prefix.sharding.is_fully_replicated
            assert remainder.size == 0
            assert resharded_shapes == []

            odd = jax.device_put(
                jnp.arange(13.0),
                NamedSharding(caller_mesh, PartitionSpec()),
            )
            prefix, remainder = batch_module._split_shardable(odd, batching_mesh)
            assert not prefix.sharding.is_fully_replicated
            assert remainder.shape == (1,)
            assert remainder.sharding.is_fully_replicated
            assert resharded_shapes == [(1,)]

            small = jax.device_put(
                jnp.arange(3.0),
                NamedSharding(caller_mesh, PartitionSpec()),
            )
            prefix, remainder = batch_module._split_shardable(small, batching_mesh)
            assert prefix.shape == (0,)
            assert remainder.shape == (3,)
            assert remainder.sharding.is_fully_replicated
            np.testing.assert_allclose(remainder, small)
            assert resharded_shapes == [(1,), (3,)]

        finally:
            batch_module._reshard_leaf_to_replicated = original_reshard
        """)
