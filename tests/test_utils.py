# Copyright (C) 2024 Kaya Unalmis
# SPDX-License-Identifier: LGPL-3.0

"""Tests for general utilities."""

import warnings

import jax.numpy as jnp
import pytest

from adv_jax_math._utils import errorif, identity, warnif


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
def test_identity_returns_same_object():
    """Identity should return its input without copying it."""
    value = object()
    assert identity(value) is value
