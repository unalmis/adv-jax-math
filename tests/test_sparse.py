"""Tests for autodiff."""

import os
import subprocess
import sys
import textwrap
from functools import partial
from unittest.mock import Mock, patch

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from packaging.version import Version

from adv_jax_math import sparse_pullback, sparse_pullback_map

_HAS_HIJAX = Version(jax.__version__) >= Version("0.11.0")


def _cube(y):
    return y**3


def _jvp(fn, y, tangent):
    return jax.jvp(fn, (y,), (tangent,))


def _sum_output(fn, y):
    return jnp.sum(fn(y))


def _call_with_a(fn, b, a):
    return fn({"a": a, "b": b})


def _double(x):
    return 2 * x


def _stack_with_double(x):
    return jnp.stack((x, 2 * x))


def _pytree_fun(y):
    return jnp.sum(jnp.sin(y["a"]) + y["a"] ** 2, axis=1) + jnp.sum(
        jnp.exp(y["b"]), axis=1
    )


def _scaled_cube(scale, y):
    return scale * y**3


def _sparse_scaled_cube(scale, y):
    return sparse_pullback(partial(_scaled_cube, scale), y, batch_size=2)


def _sparse_scaled_cube_at_fixed_input(scale, y):
    return sparse_pullback_map(partial(_scaled_cube, scale), y)(y)


def _dynamic_closure_jvp(scale, x, tangent):
    return jax.jvp(partial(_sparse_scaled_cube, scale), (x,), (tangent,))


def _dynamic_closure_vjp(scale, x, cotangent):
    pullback = jax.vjp(partial(_sparse_scaled_cube, scale), x)[1]
    return pullback(cotangent)[0]


def _nested_batched_fun(y):
    a = y["a"]
    c = y["b"]["c"]
    return jnp.sum(jnp.sin(a) + a**2, axis=1) + jnp.sum(jnp.exp(c), axis=1)


def _nested_single_fun(y):
    a = y["a"]
    c = y["b"]["c"]
    return jnp.sum(jnp.sin(a) + a**2) + jnp.sum(jnp.exp(c))


def _vmap_call(fn, y):
    return jax.vmap(fn)(y)


def _scaled_ones(scale, leaf):
    return scale * jnp.ones_like(leaf)


def _assert_tree_allclose(got, expected):
    assert eqx.tree_equal(got, expected, rtol=1e-6, atol=1e-7)


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


class TestDerivative:
    """Tests Derivative classes."""

    @pytest.mark.unit
    def test_sparse_pullback_sharded_chunked(self):
        """Test sparse_pullback with chunking and sharded input data."""
        _run_forced_cpu_devices("""
            import numpy as np

            import jax
            import jax.numpy as jnp

            from adv_jax_math import sparse_pullback

            assert jax.device_count() == 4
            x = jnp.arange(13.0)

            cases = [
                (
                    lambda y: sparse_pullback(
                        lambda z: z**2,
                        y,
                        batch_size=2,
                        shard_input_data=True,
                    ),
                    x**2,
                ),
                (
                    lambda y: sparse_pullback(
                        lambda z: z**2,
                        y,
                        shard_input_data=True,
                    ),
                    x**2,
                ),
                (
                    lambda y: sparse_pullback(
                        lambda z: z**2,
                        y,
                        batch_size=1,
                        strip_dim0=True,
                        shard_input_data=True,
                    ),
                    x**2,
                ),
                (
                    lambda y: sparse_pullback(
                        lambda z: z**2,
                        y,
                        batch_size=2,
                        reduction=jnp.add,
                        chunk_reduction=jnp.sum,
                        shard_input_data=True,
                    ),
                    jnp.sum(x**2),
                ),
            ]
            for fun, expected in cases:
                np.testing.assert_allclose(fun(x), expected)
                np.testing.assert_allclose(jax.jit(fun)(x), expected)
            """)


@pytest.mark.unit
def test_sparse_pullback_map():
    """Functional sparse-pullback wrappers should preserve values and VJPs."""
    x = jnp.arange(1.0, 5.0)
    wrapped = sparse_pullback_map(_cube, x)

    out, pullback = jax.vjp(wrapped, x)

    np.testing.assert_allclose(out, x**3)
    np.testing.assert_allclose(pullback(jnp.ones_like(out))[0], 3 * x**2)


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_HIJAX, reason="HiJAX requires JAX 0.11 or newer")
def test_sparse_pullback_hijax_jvp_and_linearize():
    """The HiJAX backend should contract and reuse pytree linearizations."""
    x = {
        "a": jnp.linspace(-1.0, 1.0, 12).reshape(4, 3),
        "b": jnp.linspace(0.1, 0.8, 8).reshape(4, 2),
    }
    tangent = {
        "a": jnp.linspace(0.2, 0.8, 12).reshape(4, 3),
        "b": jnp.linspace(-0.3, 0.4, 8).reshape(4, 2),
    }

    wrapped = sparse_pullback_map(_pytree_fun, x)
    expected_out, expected_tangent = jax.jvp(_pytree_fun, (x,), (tangent,))
    out, out_tangent = jax.jvp(wrapped, (x,), (tangent,))
    assert eqx.tree_equal(out, expected_out, rtol=1e-6, atol=1e-7)
    assert eqx.tree_equal(out_tangent, expected_tangent, rtol=1e-6, atol=1e-7)

    only_a = partial(_call_with_a, wrapped, x["b"])
    partial_out, partial_tangent = jax.jvp(
        only_a,
        (x["a"],),
        (tangent["a"],),
    )
    expected_partial_tangent = jnp.sum(
        (jnp.cos(x["a"]) + 2 * x["a"]) * tangent["a"],
        axis=1,
    )
    assert eqx.tree_equal(partial_out, expected_out, rtol=1e-6, atol=1e-7)
    assert eqx.tree_equal(
        partial_tangent,
        expected_partial_tangent,
        rtol=1e-6,
        atol=1e-7,
    )

    cotangent = jnp.linspace(0.5, 1.5, 4)
    partial_pullback = jax.vjp(only_a, x["a"])[1]
    expected_partial_pullback = (jnp.cos(x["a"]) + 2 * x["a"]) * cotangent[:, None]
    assert eqx.tree_equal(
        partial_pullback(cotangent)[0],
        expected_partial_pullback,
        rtol=1e-6,
        atol=1e-7,
    )
    out, pushforward = jax.linearize(wrapped, x)
    assert eqx.tree_equal(out, expected_out, rtol=1e-6, atol=1e-7)
    assert eqx.tree_equal(pushforward(tangent), expected_tangent, rtol=1e-6, atol=1e-7)
    assert eqx.tree_equal(
        pushforward(jax.tree.map(_double, tangent)),
        2 * expected_tangent,
        rtol=1e-6,
        atol=1e-7,
    )

    tangent_batch = jax.tree.map(_stack_with_double, tangent)
    assert eqx.tree_equal(
        jax.vmap(pushforward)(tangent_batch),
        jnp.stack((expected_tangent, 2 * expected_tangent)),
        rtol=1e-6,
        atol=1e-7,
    )


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_HIJAX, reason="HiJAX requires JAX 0.11 or newer")
def test_sparse_pullback_hijax_jacobians():
    """Jacobian APIs should compose the reusable pushforward and pullback."""
    x = jnp.arange(1.0, 5.0)
    wrapped = sparse_pullback_map(_cube, x)
    expected = jnp.diag(3 * x**2)

    np.testing.assert_allclose(jax.jacfwd(wrapped)(x), expected)
    np.testing.assert_allclose(jax.jacrev(wrapped)(x), expected)


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_HIJAX, reason="HiJAX requires JAX 0.11 or newer")
def test_sparse_pullback_hijax_higher_order():
    """Higher-order mode should rebind the primal and compose derivatives."""
    from adv_jax_math._sparse import _SparsePullbackPrimitive

    x = jnp.arange(1.0, 5.0)
    wrapped = sparse_pullback_map(_cube, x, higher_order=True)

    with patch.object(
        _SparsePullbackPrimitive,
        "expand",
        autospec=True,
        wraps=_SparsePullbackPrimitive.expand,
    ) as expand:
        out, pullback = jax.vjp(wrapped, x)

    np.testing.assert_allclose(out, x**3)
    np.testing.assert_allclose(pullback(jnp.ones_like(out))[0], 3 * x**2)
    assert expand.call_count == 1

    expected_hessian = jnp.diag(6 * x)
    scalar = partial(_sum_output, wrapped)
    np.testing.assert_allclose(
        jax.jacrev(jax.grad(scalar))(x),
        expected_hessian,
    )
    np.testing.assert_allclose(
        jax.hessian(scalar)(x),
        expected_hessian,
    )

    chunked = partial(
        sparse_pullback,
        _cube,
        batch_size=2,
        higher_order=True,
    )
    np.testing.assert_allclose(
        jax.hessian(partial(_sum_output, chunked))(x),
        expected_hessian,
    )


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_HIJAX, reason="HiJAX requires JAX 0.11 or newer")
def test_sparse_pullback_hijax_vmap():
    """The primitive should support mapped and unmapped batching inputs."""
    x = jnp.arange(1.0, 5.0)
    wrapped = sparse_pullback_map(_cube, x)
    xs = jnp.stack((x, 2 * x))
    tangents = jnp.ones_like(xs)

    np.testing.assert_allclose(jax.vmap(wrapped)(xs), xs**3)
    _, tangent_out = jax.vmap(partial(_jvp, wrapped))(xs, tangents)
    np.testing.assert_allclose(tangent_out, 3 * xs**2)
    np.testing.assert_allclose(
        jax.vmap(jax.grad(partial(_sum_output, wrapped)))(xs),
        3 * xs**2,
    )
    vmapped = partial(_vmap_call, wrapped)
    np.testing.assert_allclose(
        jax.grad(partial(_sum_output, vmapped))(xs),
        3 * xs**2,
    )
    np.testing.assert_allclose(
        jax.vmap(wrapped, in_axes=1, out_axes=1)(xs.T),
        (xs.T) ** 3,
    )
    np.testing.assert_allclose(
        jax.vmap(partial(wrapped, x), in_axes=(), out_axes=None, axis_size=2)(),
        x**3,
    )


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_HIJAX, reason="HiJAX requires JAX 0.11 or newer")
def test_sparse_pullback_hijax_reuses_diagonal(monkeypatch):
    """Linearized and transposed calls should not recompute the diagonal."""
    from adv_jax_math._sparse import _SparsePullbackPrimitive

    filter_vjp = Mock(wraps=eqx.filter_vjp)
    expand = Mock(wraps=_SparsePullbackPrimitive.expand)
    monkeypatch.setattr(eqx, "filter_vjp", filter_vjp)
    monkeypatch.setattr(_SparsePullbackPrimitive, "expand", expand)

    x = jnp.arange(1.0, 5.0)
    tangent = jnp.linspace(0.1, 0.4, 4)
    cotangent = jnp.linspace(0.5, 1.1, 4)
    wrapped = sparse_pullback_map(_cube, x)

    _, pushforward = jax.linearize(wrapped, x)
    assert filter_vjp.call_count == 1
    assert expand.call_count == 0
    np.testing.assert_allclose(pushforward(tangent), 3 * x**2 * tangent)
    np.testing.assert_allclose(pushforward(2 * tangent), 6 * x**2 * tangent)
    np.testing.assert_allclose(
        jax.vmap(pushforward)(jnp.stack((tangent, 2 * tangent))),
        jnp.stack((3 * x**2 * tangent, 6 * x**2 * tangent)),
    )
    assert filter_vjp.call_count == 1
    assert expand.call_count == 0

    filter_vjp.reset_mock()
    expand.reset_mock()
    _, pullback = jax.vjp(wrapped, x)
    assert filter_vjp.call_count == 1
    assert expand.call_count == 0
    np.testing.assert_allclose(pullback(cotangent)[0], 3 * x**2 * cotangent)
    np.testing.assert_allclose(pullback(2 * cotangent)[0], 6 * x**2 * cotangent)
    np.testing.assert_allclose(
        jax.vmap(pullback)(jnp.stack((cotangent, 2 * cotangent)))[0],
        jnp.stack((3 * x**2 * cotangent, 6 * x**2 * cotangent)),
    )
    assert filter_vjp.call_count == 1
    assert expand.call_count == 0


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_HIJAX, reason="HiJAX requires JAX 0.11 or newer")
def test_sparse_pullback_hijax_jit_with_dynamic_closure():
    """Array-valued function closures should remain dynamic under JIT."""
    x = jnp.arange(1.0, 5.0)
    tangent = jnp.linspace(0.1, 0.4, 4)
    cotangent = jnp.linspace(0.5, 1.1, 4)
    scale = jnp.array(1.7)

    out, out_tangent = jax.jit(_dynamic_closure_jvp)(scale, x, tangent)
    np.testing.assert_allclose(out, scale * x**3)
    np.testing.assert_allclose(out_tangent, scale * 3 * x**2 * tangent)

    np.testing.assert_allclose(
        jax.jit(_dynamic_closure_vjp)(scale, x, cotangent),
        scale * 3 * x**2 * cotangent,
    )

    closure_only = partial(_sparse_scaled_cube_at_fixed_input, y=x)
    out, out_tangent = jax.jvp(
        closure_only,
        (scale,),
        (jnp.ones_like(scale),),
    )
    np.testing.assert_allclose(out, scale * x**3)
    np.testing.assert_allclose(out_tangent, jnp.zeros_like(out))

    out, pushforward = jax.linearize(closure_only, scale)
    np.testing.assert_allclose(out, scale * x**3)
    np.testing.assert_allclose(
        pushforward(jnp.ones_like(scale)),
        jnp.zeros_like(out),
    )


@pytest.mark.unit
def test_sparse_pullback_legacy_backend():
    """JAX versions predating HiJAX should retain the custom-VJP backend."""
    _run_forced_cpu_devices(
        """
        import numpy as np

        import jax
        import jax.numpy as jnp

        jax.__version__ = "0.10.0"

        from adv_jax_math import sparse_pullback, sparse_pullback_map
        from adv_jax_math._derivatives import _USE_HIJAX

        assert not _USE_HIJAX
        x = jnp.arange(1.0, 5.0)
        wrapped = sparse_pullback_map(lambda y: y**3, x)
        out, pullback = jax.vjp(wrapped, x)
        np.testing.assert_allclose(out, x**3)
        np.testing.assert_allclose(
            pullback(jnp.ones_like(out))[0],
            3 * x**2,
        )

        try:
            jax.jvp(wrapped, (x,), (jnp.ones_like(x),))
        except TypeError:
            pass
        else:
            raise AssertionError("legacy custom VJP unexpectedly supported JVP")

        for fun in (
            lambda: sparse_pullback_map(lambda y: y**3, x, higher_order=True),
            lambda: sparse_pullback(lambda y: y**3, x, higher_order=True),
        ):
            try:
                fun()
            except NotImplementedError:
                pass
            else:
                raise AssertionError("legacy backend accepted higher_order=True")
        """,
        num_devices=1,
    )


@pytest.mark.unit
@pytest.mark.parametrize("case", ["unbatched", "chunked", "reduced", "strip_dim0"])
def test_sparse_pullback(case):
    """Test sparse pullback."""
    x = {
        "a": jnp.linspace(-1, 1, 10 * 7).reshape(10, 7),
        "b": {"c": jnp.linspace(0.1, 0.9, 10 * 5).reshape(10, 5)},
    }
    ct = jnp.linspace(0.5, 1.5, 10)

    if case == "unbatched":
        dense = _nested_batched_fun
        sparse = partial(sparse_pullback, _nested_batched_fun)
        cotangent = ct
    elif case == "chunked":
        dense = _nested_batched_fun
        sparse = partial(sparse_pullback, _nested_batched_fun, batch_size=4)
        cotangent = ct
    elif case == "reduced":
        dense = partial(_sum_output, _nested_batched_fun)
        sparse = partial(
            sparse_pullback,
            _nested_batched_fun,
            batch_size=4,
            reduction=jnp.add,
            chunk_reduction=jnp.sum,
        )
        cotangent = jnp.array(1.7)
    else:
        dense = partial(_vmap_call, _nested_single_fun)
        sparse = partial(
            sparse_pullback,
            _nested_single_fun,
            batch_size=1,
            strip_dim0=True,
        )
        cotangent = ct

    out, pullback = jax.vjp(dense, x)
    got_out, got_pullback = jax.vjp(sparse, x)
    np.testing.assert_allclose(got_out, out)
    _assert_tree_allclose(got_pullback(cotangent)[0], pullback(cotangent)[0])

    if _HAS_HIJAX:
        tangent = jax.tree.map(partial(_scaled_ones, 0.37), x)
        expected_out, expected_tangent = jax.jvp(dense, (x,), (tangent,))
        got_out, got_tangent = jax.jvp(sparse, (x,), (tangent,))
        np.testing.assert_allclose(got_out, expected_out)
        np.testing.assert_allclose(
            got_tangent,
            expected_tangent,
            rtol=1e-6,
            atol=1e-7,
        )
