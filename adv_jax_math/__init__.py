# Copyright (C) 2024 Kaya Unalmis
# SPDX-License-Identifier: LGPL-3.0

"""Advanced automatic-differentiation and batching utilities for JAX."""

from . import _version
from ._batch import batch_map
from ._sparse import sparse_pullback, sparse_pullback_map

__all__ = ["batch_map", "sparse_pullback", "sparse_pullback_map"]

__version__ = _version.get_versions()["version"]
