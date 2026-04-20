from collections.abc import Mapping
from typing import Literal, Protocol

import numpy as np


class VariantFilter(Protocol):
    """Variant-filter interface consumed by
    :class:`vcztools.retrieval.VczReader`.

    Implementations need not inherit from this Protocol — any object
    that provides the three members below will satisfy it. The reader
    uses :attr:`referenced_fields` to decide which VCZ arrays to fetch
    and :attr:`scope` to decide how to combine the mask returned by
    :meth:`evaluate` with the per-variant region/target mask.

    A ``"variant"``-scope filter MUST return a 1-D bool array of length
    ``n_variants``. A ``"sample"``-scope filter MUST return a 2-D bool
    array of shape ``(n_variants, n_samples)``. When the reader is
    configured with ``filter_on_subset_samples=True``, a sample-scope
    filter sees (and must return a mask for) only the *subset* sample
    axis; otherwise it sees the full sample axis.
    """

    @property
    def referenced_fields(self) -> set[str]: ...

    @property
    def scope(self) -> Literal["variant", "sample"]: ...

    def evaluate(self, chunk_data: Mapping[str, np.ndarray]) -> np.ndarray: ...
