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

from adv_jax_math import (
    batch_map,
    batched_vectorize,
    jacfwd_chunked,
    jacrev_chunked,
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
def test_batch_map_with_chunk_size():
    """Test batch_map with a chunk size."""
    x = jnp.arange(5.0)
    np.testing.assert_allclose(batch_map(lambda y: y + 1, x, batch_size=2), x + 1)


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

        from adv_jax_math import jacfwd_chunked, jacrev_chunked

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
        import numpy as np

        import jax
        import jax.numpy as jnp

        from adv_jax_math import batch_map, vmap_chunked

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
        """)


@pytest.mark.unit
def test_make_shardable():
    """Test that sharding works."""
    _run_forced_cpu_devices("""
        import numpy as np

        import jax
        import jax.numpy as jnp

        from adv_jax_math import make_shardable

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
