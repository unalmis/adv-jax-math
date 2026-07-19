# Copyright (C) 2024 Kaya Unalmis
# SPDX-License-Identifier: LGPL-3.0

"""Differentiation and batching utilities for JAX."""

from . import _version
from ._batch import (
    batch_jacfwd,
    batch_jacrev,
    batch_map,
    batch_vectorize,
    batch_vmap,
)
from ._sparse import sparse_pullback, sparse_pullback_map

__all__ = [
    "batch_jacfwd",
    "batch_jacrev",
    "batch_map",
    "batch_vectorize",
    "batch_vmap",
    "sparse_pullback",
    "sparse_pullback_map",
]

__version__ = _version.get_versions()["version"]
