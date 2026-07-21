# Copyright (C) 2024 Kaya Unalmis
# SPDX-License-Identifier: LGPL-3.0

"""General utilities."""

import warnings
from typing import Type


def errorif(cond: bool, err: Type[Exception] = ValueError, msg: str = ""):
    """Raise an error if condition is met."""
    if cond:
        raise err(msg)


def warnif(cond: bool, err: Type[Warning] = UserWarning, msg: str = ""):
    """Throw a warning if condition is met."""
    if cond:
        warnings.warn(msg, err)


def identity(x):
    """Return ``x`` unchanged."""
    return x
