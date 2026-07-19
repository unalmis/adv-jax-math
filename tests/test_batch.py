# Copyright (C) 2024 Kaya Unalmis
# SPDX-License-Identifier: LGPL-3.0

"""Tests for batching functions."""

import os
import subprocess
import sys
import textwrap

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from adv_jax_math._batch import (
    batch_map,
    batched_vectorize,
    jacfwd_chunked,
    jacrev_chunked,
    make_shardable,
    vmap_chunked,
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
            {"shard_input_data": True},
            False,
            id="sharded",
        ),
        pytest.param(
            {"batch_size": 2, "shard_input_data": True},
            False,
            id="sharded-chunked",
        ),
        pytest.param(
            {"batch_size": 8, "shard_input_data": True},
            False,
            id="sharded-full-batch",
        ),
        pytest.param(
            {
                "batch_size": 1,
                "strip_dim0": True,
                "shard_input_data": True,
            },
            False,
            id="sharded-stripped",
        ),
        pytest.param(
            {
                "batch_size": 2,
                "reduction": jnp.add,
                "chunk_reduction": jnp.sum,
                "shard_input_data": True,
            },
            True,
            id="sharded-reduced",
        ),
    ),
)
def test_batch_map_modes(kwargs, reduce_output):
    """Batching modes should preserve values and requested reductions."""
    x = jnp.arange(5.0)
    expected = jnp.sum(x + 1) if reduce_output else x + 1
    np.testing.assert_allclose(batch_map(lambda y: y + 1, x, **kwargs), expected)


@pytest.mark.unit
@pytest.mark.parametrize("chunk_size", (None, 1, 2, 8))
def test_vmap_chunked_matches_vmap(chunk_size):
    """Chunk sizes should not change mapped results with static arguments."""
    x = jnp.arange(5.0)
    actual = vmap_chunked(
        lambda value, scale: value * scale,
        in_axes=(0, None),
        chunk_size=chunk_size,
    )(x, 3.0)
    np.testing.assert_allclose(actual, x * 3)


@pytest.mark.unit
def test_vmap_chunked_reduction():
    """Chunk and cross-chunk reductions should compose."""
    x = jnp.arange(5.0)
    actual = vmap_chunked(
        lambda value: value**2,
        chunk_size=2,
        reduction=jnp.add,
        chunk_reduction=jnp.sum,
    )(x)
    np.testing.assert_allclose(actual, jnp.sum(x**2))


@pytest.mark.unit
def test_vmap_chunked_rejects_unsupported_axes():
    """Only mapped axis zero and unmapped inputs are supported."""
    with pytest.raises(NotImplementedError, match="Only in_axes 0/None"):
        vmap_chunked(lambda value: value, in_axes=1)


@pytest.mark.unit
@pytest.mark.parametrize("chunk_size", (None, 2))
def test_sharded_batching_falls_back_when_unsupported(monkeypatch, chunk_size):
    """Older JAX installations should transparently use ordinary chunking."""
    import adv_jax_math._batch as batch_module

    monkeypatch.setattr(batch_module, "_SUPPORTS_SHARDED_BATCHING", False)
    x = jnp.arange(5.0)
    actual = vmap_chunked(
        lambda value: value + 1,
        chunk_size=chunk_size,
        shard_input_data=True,
    )(x)
    np.testing.assert_allclose(actual, x + 1)


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

    def split(value, _axis, _num_devices, _mesh, *, normalize_explicit=False):
        del normalize_explicit
        return value[:split_at], value[split_at:]

    monkeypatch.setattr(batch_module, "_make_shardable", split)
    mesh = batch_module._make_automatic_mesh(1)
    x = jnp.arange(5.0)
    actual = batch_module._evaluate_sharded_on_mesh(
        lambda value: value + 1,
        2,
        (0,),
        reduction,
        chunk_reduction,
        1,
        mesh,
        x,
    )
    expected = x + 1 if reduction is None else jnp.sum(x + 1)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.unit
def test_chunked_jacobians_match_jax():
    """Chunked forward- and reverse-mode Jacobians should match JAX."""
    x = jnp.array([1.0, 2.0, 3.0])

    def fun(y):
        return jnp.array([y[0] * y[1], jnp.sin(y[2])])

    np.testing.assert_allclose(
        jacfwd_chunked(fun, chunk_size=2)(x),
        jax.jacfwd(fun)(x),
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        jacrev_chunked(fun, chunk_size=1)(x),
        jax.jacrev(fun)(x),
        rtol=1e-6,
        atol=1e-7,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("chunked_jacobian", "jax_jacobian"),
    ((jacfwd_chunked, jax.jacfwd), (jacrev_chunked, jax.jacrev)),
)
def test_chunked_jacobians_multiple_arguments_and_aux(chunked_jacobian, jax_jacobian):
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
    got = chunked_jacobian(fun, chunk_size=2, **kwargs)(x, y, 0.5)
    expected = jax_jacobian(fun, **kwargs)(x, y, 0.5)

    _assert_tree_allclose(got, expected)


@pytest.mark.unit
def test_argnums_partial2_fallback():
    """Older JAX versions should use the callable argnums fallback."""
    _run_forced_cpu_devices(
        """
        import numpy as np

        import jax
        import jax.numpy as jnp
        from jax._src import api_util

        if hasattr(api_util, "argnums_partial2"):
            del api_util.argnums_partial2

        from adv_jax_math._batch import jacfwd_chunked, jacrev_chunked

        def fun(x, y, *, scale):
            return scale * jnp.array([x * y, x + 2 * y])

        x, y = jnp.array(2.0), jnp.array(3.0)
        kwargs = {"argnums": (-2, -1), "chunk_size": 1}
        expected_kwargs = {"argnums": (-2, -1)}
        np.testing.assert_allclose(
            jacfwd_chunked(fun, **kwargs)(x, y, scale=0.5),
            jax.jacfwd(fun, **expected_kwargs)(x, y, scale=0.5),
        )
        np.testing.assert_allclose(
            jacrev_chunked(fun, **kwargs)(x, y, scale=0.5),
            jax.jacrev(fun, **expected_kwargs)(x, y, scale=0.5),
        )
        """,
        num_devices=1,
    )


@pytest.mark.unit
def test_batched_vectorize_with_and_without_signature():
    """Batched vectorization should support scalar and gufunc signatures."""
    x = jnp.arange(5.0)
    scalar_fun = batched_vectorize(lambda y: y**2 + 1, chunk_size=2)
    np.testing.assert_allclose(scalar_fun(x), x**2 + 1)

    dot = batched_vectorize(
        lambda y, z: y @ z,
        signature="(n),(n)->()",
        chunk_size=2,
    )
    y = jnp.arange(12.0).reshape(4, 3)
    z = jnp.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(dot(y, z), y @ z)


@pytest.mark.unit
@pytest.mark.parametrize("chunk_size", (None, 1, 2, 4))
def test_batched_vectorize_matches_jax_broadcasting(chunk_size):
    """Chunking should be the only difference from jax.numpy.vectorize."""

    def fun(x, y):
        return jnp.sin(x) + y**2

    x = jnp.arange(2.0).reshape(2, 1)
    y = jnp.arange(3.0)
    expected = jnp.vectorize(fun)(x, y)
    actual = batched_vectorize(fun, chunk_size=chunk_size)(x, y)
    _assert_tree_allclose(actual, expected)


@pytest.mark.unit
def test_batched_vectorize_matches_jax_special_arguments_and_outputs():
    """Excluded and None arguments and tuple gufunc outputs should match JAX."""

    def polynomial(x, power, *, offset):
        return x**power + offset

    excluded = frozenset((1, "offset"))
    x = jnp.arange(5.0)
    expected = jnp.vectorize(polynomial, excluded=excluded)(x, 2, offset=3)
    actual = batched_vectorize(polynomial, excluded=excluded, chunk_size=2)(
        x, 2, offset=3
    )
    _assert_tree_allclose(actual, expected)

    def maybe_shift(value, shift):
        return value if shift is None else value + shift

    expected = jnp.vectorize(maybe_shift)(x, None)
    actual = batched_vectorize(maybe_shift, chunk_size=2)(x, None)
    _assert_tree_allclose(actual, expected)

    def statistics(values):
        return jnp.sum(values), jnp.max(values)

    values = jnp.arange(12.0).reshape(4, 3)
    signature = "(n)->(),()"
    expected = jnp.vectorize(statistics, signature=signature)(values)
    actual = batched_vectorize(statistics, signature=signature, chunk_size=2)(values)
    _assert_tree_allclose(actual, expected)


@pytest.mark.unit
@pytest.mark.parametrize(
    "excluded",
    (
        pytest.param(frozenset({1.5}), id="non-string-or-integer"),
        pytest.param(frozenset({-1}), id="negative-integer"),
    ),
)
def test_batched_vectorize_rejects_invalid_exclusions(excluded):
    """Excluded arguments should follow the jax.numpy.vectorize contract."""
    with pytest.raises((TypeError, ValueError)):
        batched_vectorize(jnp.add, excluded=excluded)


@pytest.mark.unit
def test_batched_vectorize_rejects_none_core_argument():
    """None is only valid for arguments without core dimensions."""
    vectorized = batched_vectorize(lambda value: value, signature="(n)->()")
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
def test_batched_vectorize_preserves_singleton_dimensions(fun):
    """Singleton mapped dimensions should be restored after evaluation."""
    x = jnp.ones((1,))
    expected = jnp.vectorize(fun)(x)
    actual = batched_vectorize(fun, chunk_size=1)(x)
    _assert_tree_allclose(actual, expected)


@pytest.mark.unit
def test_batched_vectorize_matches_jax_rank_promotion_policy():
    """Implicit rank-promotion errors should match jax.numpy.vectorize."""
    x = jnp.ones((2, 3))
    y = jnp.ones((3,))

    with jax.numpy_rank_promotion("raise"):
        with pytest.raises(ValueError, match="require rank promotion"):
            jnp.vectorize(jnp.add)(x, y)
        with pytest.raises(ValueError, match="require rank promotion"):
            batched_vectorize(jnp.add, chunk_size=2)(x, y)

    with jax.numpy_rank_promotion("warn"):
        with pytest.warns(UserWarning, match="require rank promotion"):
            expected = jnp.vectorize(jnp.add)(x, y)
        with pytest.warns(UserWarning, match="require rank promotion"):
            actual = batched_vectorize(jnp.add, chunk_size=2)(x, y)
    _assert_tree_allclose(actual, expected)


@pytest.mark.unit
def test_sharded_chunked_batching():
    """Test chunked batching with sharded input data."""
    _run_forced_cpu_devices("""
        from collections import Counter

        import numpy as np

        import jax
        import jax.numpy as jnp
        from packaging.version import Version

        from adv_jax_math._batch import batch_map, vmap_chunked

        assert jax.device_count() == 4
        x = jnp.arange(13.0)

        cases = [
            (
                lambda y: batch_map(
                    lambda z: z + 1,
                    y,
                    batch_size=2,
                    shard_input_data=True,
                ),
                x + 1,
            ),
            (
                lambda y: batch_map(lambda z: z + 1, y, shard_input_data=True),
                x + 1,
            ),
            (
                lambda y: batch_map(
                    lambda z: z + 1,
                    y,
                    batch_size=1,
                    strip_dim0=True,
                    shard_input_data=True,
                ),
                x + 1,
            ),
            (
                lambda y: vmap_chunked(
                    lambda z, scale: z * scale,
                    in_axes=(0, None),
                    chunk_size=2,
                    shard_input_data=True,
                )(y, 3.0),
                x * 3,
            ),
            (
                lambda y: vmap_chunked(
                    lambda z, scale: z * scale,
                    in_axes=(0, None),
                    shard_input_data=True,
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
                    shard_input_data=True,
                ),
                jnp.sum(x),
            ),
        ]
        for fun, expected in cases:
            np.testing.assert_allclose(fun(x), expected)
            np.testing.assert_allclose(jax.jit(fun)(x), expected)

        two_inputs = lambda y, z: vmap_chunked(
            lambda a, b: a - b,
            in_axes=(0, 0),
            chunk_size=2,
            shard_input_data=True,
        )(y, z)
        np.testing.assert_allclose(two_inputs(x, x[::-1]), x - x[::-1])
        np.testing.assert_allclose(jax.jit(two_inputs)(x, x[::-1]), x - x[::-1])

        if Version(jax.__version__) >= Version("0.10.2"):
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
                shard_input_data=True,
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
def test_make_shardable():
    """Test that sharding works."""
    _run_forced_cpu_devices("""
        import numpy as np

        import jax
        import jax.numpy as jnp

        from adv_jax_math._batch import make_shardable

        assert jax.device_count() == 4

        f = np.arange(21)
        sf, rf = make_shardable(f, num_devices=4)
        assert sf.size == 20
        assert rf.size == 1
        np.testing.assert_allclose(
            np.concatenate([np.asarray(jnp.sin(sf)), np.asarray(jnp.sin(rf))]),
            jnp.sin(f),
        )

        f = jnp.arange(20).reshape(2, 10)
        sf, rf = make_shardable(f, axis=1, num_devices=4)
        assert sf.shape == (2, 8)
        assert rf.shape == (2, 2)
        np.testing.assert_allclose(
            np.concatenate(
                [np.asarray(jnp.sin(sf)), np.asarray(jnp.sin(rf))], axis=1
            ),
            jnp.sin(f),
        )
        """)


@pytest.mark.unit
@pytest.mark.parametrize("axis", (0, -1))
def test_make_shardable_defaults_to_available_devices(axis):
    """The default device count should shard pytrees along normalized axes."""
    x = {"value": jnp.arange(12.0).reshape(3, 4)}
    sharded, remainder = make_shardable(x, axis=axis)
    normalized_axis = axis % x["value"].ndim
    combined = jnp.concatenate(
        (sharded["value"], remainder["value"]),
        axis=normalized_axis,
    )
    np.testing.assert_allclose(combined, x["value"])
