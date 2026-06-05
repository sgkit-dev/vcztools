"""Registry of *virtual* variant-axis fields — values produced per chunk
rather than read directly from Zarr. Most are derived from other store
fields (e.g. ``variant_AC`` from genotypes); others are structural,
derived from each variant's position in the store (``variant_index``).

A virtual field looks identical to a stored field at the
:class:`~vcztools.retrieval.VczReader` API: callers reference it by
its ``variant_*`` name in :meth:`~vcztools.retrieval.VczReader.variant_chunks`
or in a ``bcftools_filter`` expression. Resolution at chunk time:

- If the store has a real array under the same name and the caller did
  not ask the reader to recompute it, the stored value is returned.
- Otherwise the field's :attr:`VirtualField.compute` runs against the
  per-chunk materialisations of its declared dependencies.

The registry is the single source of truth for what's computable; the
reader consults it once at construction to drop any entry whose deps
aren't satisfied by the current store (with an optional ``degenerate``
form taking over when the primary's deps are absent — used for
``N_MISSING`` / ``F_MISSING`` on annotations-only stores that lack
``call_genotype``).

Compute functions take ``(deps, cache, context)``:

- ``deps`` — dict keyed by real-field name; values are the chunk-local
  arrays already materialised by the dispatcher.
- ``cache`` — per-chunk dict shared across virtual fields. Functions
  may read sibling values from it (e.g. AF reusing AC/AN) or publish
  results back into it (AC's compute publishes AN as a side effect so
  the AN entry can short-circuit).
- ``context`` — per-chunk :class:`ChunkContext` giving the chunk's plan
  entry and chunk size. Used by structural fields like ``variant_index``
  that derive from variant position; value fields ignore it.
"""

import dataclasses
from collections.abc import Callable

import numpy as np

from vcztools import calculate, constants, utils


@dataclasses.dataclass(frozen=True)
class VirtualField:
    name: str
    deps: tuple[str, ...]
    dtype: np.dtype
    dims: tuple[str, ...]
    description: str
    shape_from: str
    compute: Callable
    degenerate: "VirtualField | None" = None


@dataclasses.dataclass(frozen=True)
class ChunkContext:
    """Per-chunk plan state passed to :attr:`VirtualField.compute`.

    Structural virtual fields (``variant_index``) derive their value
    from where the chunk sits in the store rather than from any stored
    field; this carries the chunk's plan entry and the chunk size its
    index is measured in.
    """

    variant_chunk: utils.ChunkRead
    chunk_size: int


def _absolute_variant_indexes(entry: utils.ChunkRead, chunk_size: int) -> np.ndarray:
    """Global variant indexes contributed by ``entry``.

    Maps each variant the chunk read selects to its store-wide variant
    index. ``chunk_size`` is the unit ``entry.index`` is measured in —
    ``min_chunk`` for canonical plan entries, ``stream_chunk_size`` for
    stream-plan entries.
    """
    chunk_offset = entry.index * chunk_size
    sel = entry.selection
    if sel is None:
        local = np.arange(entry.num_selected, dtype=np.int64)
    elif isinstance(sel, slice):
        local = np.arange(*sel.indices(chunk_size), dtype=np.int64)
    else:
        local = np.asarray(sel, dtype=np.int64)
    return chunk_offset + local


def _count_missing_gt(gt: np.ndarray) -> np.ndarray:
    return np.sum(np.all(gt < 0, axis=-1), axis=1)


def _compute_variant_index(deps, cache, context):
    return _absolute_variant_indexes(context.variant_chunk, context.chunk_size)


def _compute_ac(deps, cache, context):
    gt = deps["call_genotype"]
    alt = deps["variant_allele"][:, 1:]
    ac, an = calculate.compute_ac_an(gt, alt)
    cache["variant_AN"] = an
    return ac


def _compute_an(deps, cache, context):
    cached = cache.get("variant_AN")
    if cached is not None:
        return cached
    gt = deps["call_genotype"]
    alt = deps["variant_allele"][:, 1:]
    _, an = calculate.compute_ac_an(gt, alt)
    return an


def _compute_af(deps, cache, context):
    ac = cache.get("variant_AC")
    an = cache.get("variant_AN")
    if ac is None:
        ac = _compute_ac(deps, cache, context)
    if an is None:
        an = cache["variant_AN"]
    with np.errstate(divide="ignore", invalid="ignore"):
        af = ac.astype(np.float32) / an[:, np.newaxis].astype(np.float32)
    af[an == 0] = constants.FLOAT32_MISSING
    af[ac == constants.INT_FILL] = constants.FLOAT32_FILL
    return af


def _compute_ns(deps, cache, context):
    gt = deps["call_genotype"]
    return np.sum(np.any(gt >= 0, axis=-1), axis=1).astype(np.int32)


def _compute_n_alt(deps, cache, context):
    alt = np.asarray(deps["variant_allele"])[:, 1:]
    return (alt != "").sum(axis=1).astype(np.int64)


def _compute_n_missing(deps, cache, context):
    return _count_missing_gt(np.asarray(deps["call_genotype"])).astype(np.int64)


def _compute_f_missing(deps, cache, context):
    gt = np.asarray(deps["call_genotype"])
    n_samples = gt.shape[1]
    if n_samples == 0:
        return np.zeros(gt.shape[0], dtype=np.float64)
    return _count_missing_gt(gt).astype(np.float64) / n_samples


def _compute_n_missing_zero(deps, cache, context):
    return np.zeros(deps["variant_position"].shape[0], dtype=np.int64)


def _compute_f_missing_zero(deps, cache, context):
    return np.zeros(deps["variant_position"].shape[0], dtype=np.float64)


REGISTRY: dict[str, VirtualField] = {
    "variant_index": VirtualField(
        name="variant_index",
        deps=(),
        dtype=np.dtype(np.int64),
        dims=("variants",),
        description="Global (store-wide) 0-based index of the variant",
        shape_from="variant_position",
        compute=_compute_variant_index,
    ),
    "variant_AC": VirtualField(
        name="variant_AC",
        deps=("call_genotype", "variant_allele"),
        dtype=np.dtype(np.int32),
        dims=("variants", "alt_alleles"),
        description="Allele count in genotypes",
        shape_from="variant_allele",
        compute=_compute_ac,
    ),
    "variant_AN": VirtualField(
        name="variant_AN",
        deps=("call_genotype", "variant_allele"),
        dtype=np.dtype(np.int32),
        dims=("variants",),
        description="Total number of alleles in called genotypes",
        shape_from="variant_position",
        compute=_compute_an,
    ),
    "variant_AF": VirtualField(
        name="variant_AF",
        deps=("call_genotype", "variant_allele"),
        dtype=np.dtype(np.float32),
        dims=("variants", "alt_alleles"),
        # Matches bcftools +fill-tags' header line so a recomputed AF
        # generates the same ``##INFO=<...>`` text as the oracle.
        description="Allele frequency",
        shape_from="variant_allele",
        compute=_compute_af,
    ),
    "variant_NS": VirtualField(
        name="variant_NS",
        deps=("call_genotype",),
        dtype=np.dtype(np.int32),
        dims=("variants",),
        description="Number of samples with data",
        shape_from="variant_position",
        compute=_compute_ns,
    ),
    "variant_N_ALT": VirtualField(
        name="variant_N_ALT",
        deps=("variant_allele",),
        dtype=np.dtype(np.int64),
        dims=("variants",),
        description="Number of non-empty ALT alleles",
        shape_from="variant_position",
        compute=_compute_n_alt,
    ),
    "variant_N_MISSING": VirtualField(
        name="variant_N_MISSING",
        deps=("call_genotype",),
        dtype=np.dtype(np.int64),
        dims=("variants",),
        description="Number of samples with all-missing genotypes",
        shape_from="variant_position",
        compute=_compute_n_missing,
        degenerate=VirtualField(
            name="variant_N_MISSING",
            deps=("variant_position",),
            dtype=np.dtype(np.int64),
            dims=("variants",),
            description="Number of samples with all-missing genotypes",
            shape_from="variant_position",
            compute=_compute_n_missing_zero,
        ),
    ),
    "variant_F_MISSING": VirtualField(
        name="variant_F_MISSING",
        deps=("call_genotype",),
        dtype=np.dtype(np.float64),
        dims=("variants",),
        description="Fraction of samples with all-missing genotypes",
        shape_from="variant_position",
        compute=_compute_f_missing,
        degenerate=VirtualField(
            name="variant_F_MISSING",
            deps=("variant_position",),
            dtype=np.dtype(np.float64),
            dims=("variants",),
            description="Fraction of samples with all-missing genotypes",
            shape_from="variant_position",
            compute=_compute_f_missing_zero,
        ),
    ),
}


def resolve_for_root(root) -> dict[str, VirtualField]:
    """Return the subset of :data:`REGISTRY` whose dependencies are
    satisfied by ``root``.

    An entry is included with its primary form if every dep in
    :attr:`VirtualField.deps` is present in ``root``; otherwise with
    its ``degenerate`` form if that form's deps are present; otherwise
    dropped.
    """
    resolved: dict[str, VirtualField] = {}
    for name, vf in REGISTRY.items():
        if all(d in root for d in vf.deps):
            resolved[name] = vf
        elif vf.degenerate is not None and all(d in root for d in vf.degenerate.deps):
            resolved[name] = vf.degenerate
    return resolved
