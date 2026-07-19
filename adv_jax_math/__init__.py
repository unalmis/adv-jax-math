"""Advanced automatic-differentiation and batching utilities for JAX."""

from importlib.metadata import PackageNotFoundError, version

from ._batch import (
    batch_map,
    batched_vectorize,
    jacfwd_chunked,
    jacrev_chunked,
    make_shardable,
    vmap_chunked,
)
from ._sparse import sparse_pullback, sparse_pullback_map

__all__ = [
    "batch_map",
    "batched_vectorize",
    "jacfwd_chunked",
    "jacrev_chunked",
    "make_shardable",
    "sparse_pullback",
    "sparse_pullback_map",
    "vmap_chunked",
]

try:
    __version__ = version("adv-jax-math")
except PackageNotFoundError:
    __version__ = "0+unknown"
