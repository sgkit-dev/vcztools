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
    array of shape ``(n_variants, n_samples)``. The sample axis the
    filter sees is controlled by
    :meth:`~vcztools.retrieval.VczReader.set_filter_samples`; by
    default it is the user's sample selection (``bcftools query``
    semantics). Callers that want ``bcftools view`` pre-subset
    semantics pass :attr:`~vcztools.retrieval.VczReader.non_null_sample_indices`.
    """

    @property
    def referenced_fields(self) -> set[str]: ...

    @property
    def scope(self) -> Literal["variant", "sample"]: ...

    def evaluate(self, chunk_data: Mapping[str, np.ndarray]) -> np.ndarray: ...
