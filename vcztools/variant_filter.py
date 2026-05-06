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


class AndFilter:
    """AND-compose multiple :class:`VariantFilter` objects, including
    across scopes.

    A variant-scope (1-D) result is broadcast along axis 1 against any
    sample-scope (2-D) result, so a synthetic ``-m/-M/-v/-V`` mask
    composes naturally with a user ``-i 'FMT/...'`` mask. The combined
    scope is ``sample`` when any input is sample-scope, else ``variant``.
    """

    def __init__(self, filters):
        self._filters = list(filters)
        self.scope = (
            "sample" if any(f.scope == "sample" for f in self._filters) else "variant"
        )
        self.referenced_fields = frozenset().union(
            *(f.referenced_fields for f in self._filters)
        )

    def evaluate(self, chunk_data):
        result = self._filters[0].evaluate(chunk_data)
        for f in self._filters[1:]:
            other = f.evaluate(chunk_data)
            if result.ndim == 1 and other.ndim == 2:
                result = np.expand_dims(result, axis=1) & other
            elif result.ndim == 2 and other.ndim == 1:
                result = result & np.expand_dims(other, axis=1)
            else:
                result = result & other
        return result


def compose(existing, new_filter):
    """AND-compose ``new_filter`` onto an existing filter, allowing
    mixed variant/sample scope via :class:`AndFilter`. Returns
    ``new_filter`` unchanged when there's nothing to compose with.
    """
    if existing is None:
        return new_filter
    return AndFilter([existing, new_filter])
