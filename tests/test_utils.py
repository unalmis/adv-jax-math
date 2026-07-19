# Copyright (C) 2024 Kaya Unalmis
# SPDX-License-Identifier: LGPL-3.0

"""Tests for general utilities."""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from adv_jax_math._utils import (
    Index,
    asarray_inexact,
    ensure_tuple,
    errorif,
    identity,
    isnonnegint,
    isposint,
    setdefault,
    warnif,
)


@pytest.mark.unit
def test_index_builds_axis_selectors():
    """Index should preserve direct indexing and expand axis selections."""
    direct = Index[1:3]
    assert direct == slice(1, 3)
    assert Index.get(2, axis=-1, ndim=3) == (slice(None), slice(None), 2)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "nonnegative", "positive"),
    (
        pytest.param(-1, False, False, id="negative"),
        pytest.param(0, True, False, id="zero"),
        pytest.param(np.int64(2), True, True, id="numpy-integer"),
        pytest.param(1.5, False, False, id="non-integer"),
    ),
)
def test_integer_predicates(value, nonnegative, positive):
    """Integer predicates should reject negatives and non-indexable values."""
    assert isnonnegint(value) == nonnegative
    assert isposint(value) == positive


@pytest.mark.unit
@pytest.mark.parametrize("condition", (False, jnp.array(False)))
def test_errorif_ignores_false_conditions(condition):
    """False Python and scalar-array conditions should not raise."""
    assert errorif(condition) is None


@pytest.mark.unit
def test_errorif_raises_requested_exception():
    """The selected exception type and message should be preserved."""
    with pytest.raises(RuntimeError, match="failed"):
        errorif(jnp.array(True), RuntimeError, "failed")


@pytest.mark.unit
def test_warnif_respects_condition():
    """Warnings should only be emitted for true conditions."""
    with pytest.warns(RuntimeWarning, match="careful"):
        warnif(True, RuntimeWarning, "careful")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnif(False)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "expected_dtype", "weak_type"),
    (
        pytest.param(1, jnp.int32, True, id="weak-scalar"),
        pytest.param([1, 2], jnp.float32, False, id="integer-array"),
        pytest.param([1.0, 2.0], jnp.float32, False, id="floating-array"),
        pytest.param([1.0j], jnp.complex64, False, id="complex-array"),
    ),
)
def test_asarray_inexact(value, expected_dtype, weak_type):
    """Arrays should be inexact while weak scalar types remain weak."""
    result = asarray_inexact(value)
    assert result.dtype == expected_dtype
    assert result.weak_type is weak_type


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "default", "condition", "expected"),
    (
        pytest.param("value", "default", None, "value", id="implicit-value"),
        pytest.param(None, "default", None, "default", id="implicit-default"),
        pytest.param("value", "default", False, "default", id="explicit-default"),
        pytest.param(None, "default", True, None, id="explicit-value"),
    ),
)
def test_setdefault(value, default, condition, expected):
    """Explicit conditions should override the None-based default behavior."""
    assert setdefault(value, default, condition) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "expected"),
    (
        pytest.param((1, 2), (1, 2), id="tuple"),
        pytest.param([1, 2], (1, 2), id="list"),
        pytest.param(1, (1,), id="scalar"),
    ),
)
def test_ensure_tuple(value, expected):
    """Only non-container values should gain a new level of nesting."""
    assert ensure_tuple(value) == expected


@pytest.mark.unit
def test_identity_returns_same_object():
    """Identity should return its input without copying it."""
    value = object()
    assert identity(value) is value
