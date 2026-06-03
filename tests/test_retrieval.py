import concurrent.futures as cf
import logging
import threading
import time

import numpy as np
import numpy.testing as nt
import pandas as pd
import pytest

from tests import vcz_builder
from tests.utils import make_reader, to_vcz_icechunk
from vcztools import constants, utils
from vcztools import regions as regions_mod
from vcztools import retrieval as retrieval_mod
from vcztools import samples as samples_mod
from vcztools.bcftools_filter import BcftoolsFilter
from vcztools.retrieval import CachedLogicalVariantsChunk, VczReader


def test_variant_chunks(fx_sample_vcz):
    reader = make_reader(
        fx_sample_vcz.group,
        regions="20:1230236-",
        samples=["NA00002", "NA00003"],
        include="FMT/DP>3",
        view_semantics=True,
    )
    chunk_data = next(
        reader.variant_chunks(
            fields=["variant_contig", "variant_position", "call_DP", "call_GQ"],
        )
    )
    nt.assert_array_equal(chunk_data["variant_contig"], [1, 1])
    nt.assert_array_equal(chunk_data["variant_position"], [1230237, 1234567])
    nt.assert_array_equal(chunk_data["call_DP"], [[4, 2], [2, 3]])
    nt.assert_array_equal(chunk_data["call_GQ"], [[48, 61], [17, 40]])
    # note second site (at pos 1234567) is included even though both samples in mask
    # are False (NA00002 and NA00003), since sample NA00001 matched filter criteria,
    # but was then removed by samples_selection
    nt.assert_array_equal(
        chunk_data["sample_filter_pass"], [[True, False], [False, False]]
    )


def test_variant_chunks_empty_fields(fx_sample_vcz):
    reader = VczReader(fx_sample_vcz.group)
    with pytest.raises(StopIteration):
        next(reader.variant_chunks(fields=[]))


def test_variant_chunks_heterogeneous_chunk_sizes():
    # Variant-only fields with chunk sizes that are multiples of
    # variants_chunk_size (3) but not pairwise multiples of each other:
    # variant_position=6, variant_contig=9. compute_stream_chunk_size
    # must return their GCD (3) — using the minimum (6) would walk
    # variant_contig's blocks at the wrong offset (9 // 6 == 1 but
    # 1 * 6 != 9) and drift each iteration.
    num_variants = 18
    positions = np.arange(1, num_variants + 1, dtype=np.int32)
    contigs = np.arange(num_variants, dtype=np.int32)
    root = vcz_builder.make_vcz(
        variant_contig=contigs,
        variant_position=positions,
        alleles=[("A", "T")] * num_variants,
        num_samples=2,
        sample_id=["s0", "s1"],
        contigs=tuple(f"chr{i}" for i in range(num_variants)),
        variants_chunk_size=3,
        samples_chunk_size=2,
        call_genotype=np.zeros((num_variants, 2, 2), dtype=np.int8),
        field_chunk_overrides={"variant_position": 6, "variant_contig": 9},
    )
    reader = VczReader(root)
    seen_positions = []
    seen_contigs = []
    for chunk in reader.variant_chunks(fields=["variant_position", "variant_contig"]):
        seen_positions.append(chunk["variant_position"])
        seen_contigs.append(chunk["variant_contig"])
    nt.assert_array_equal(np.concatenate(seen_positions), positions)
    nt.assert_array_equal(np.concatenate(seen_contigs), contigs)


@pytest.mark.parametrize(
    ("regions", "samples"),
    [
        ("20:1230236-", ["NA00002", "NA00003"]),
        (["20:1230236-"], ["NA00002", "NA00003"]),
    ],
)
def test_variant_iter(fx_sample_vcz, regions, samples):
    reader = make_reader(
        fx_sample_vcz.group,
        regions=regions,
        samples=samples,
        include="FMT/DP>3",
        view_semantics=True,
    )
    it = reader.variants(
        fields=["variant_contig", "variant_position", "call_DP", "call_GQ"],
    )

    variant1 = next(it)
    assert variant1["variant_contig"] == 1
    assert variant1["variant_position"] == 1230237
    nt.assert_array_equal(variant1["call_DP"], [4, 2])
    nt.assert_array_equal(variant1["call_GQ"], [48, 61])
    nt.assert_array_equal(variant1["sample_filter_pass"], [True, False])

    variant2 = next(it)
    assert variant2["variant_contig"] == 1
    assert variant2["variant_position"] == 1234567
    nt.assert_array_equal(variant2["call_DP"], [2, 3])
    nt.assert_array_equal(variant2["call_GQ"], [17, 40])
    nt.assert_array_equal(variant2["sample_filter_pass"], [False, False])

    with pytest.raises(StopIteration):
        next(it)


def test_variant_iter_empty_fields(fx_sample_vcz):
    reader = VczReader(fx_sample_vcz.group)
    with pytest.raises(StopIteration):
        next(reader.variants(fields=[]))


def _make_filter_vcz(num_variants=9, variants_chunk_size=3):
    """Build a multi-chunk VCZ with two filters for FILTER expression tests."""
    # Alternate PASS-only and q10-only rows so every chunk has both kinds.
    variant_filter = np.zeros((num_variants, 2), dtype=bool)
    for i in range(num_variants):
        if i % 2 == 0:
            variant_filter[i, 0] = True  # PASS
        else:
            variant_filter[i, 1] = True  # q10
    return vcz_builder.make_vcz(
        variant_contig=[0] * num_variants,
        variant_position=list(range(100, 100 + num_variants)),
        alleles=[("A", "T")] * num_variants,
        filters=("PASS", "q10"),
        variant_filter=variant_filter,
        variants_chunk_size=variants_chunk_size,
        proportional_chunks=False,
    )


class TestFilterMultiChunk:
    """Regression tests for VariantChunkReader with mixed-axis fields.

    FILTER expressions reference both variant_filter (variant-chunked) and
    filter_id (single-chunk on the filters axis). VariantChunkReader must
    handle this without IndexError or silent data loss.
    """

    @staticmethod
    def _chunks(root, **kwargs):
        reader = make_reader(root, **kwargs)
        return reader.variant_chunks(fields=["variant_position"])

    def test_include_filter_eq_pass(self):
        results = list(self._chunks(_make_filter_vcz(), include='FILTER="PASS"'))
        positions = np.concatenate([chunk["variant_position"] for chunk in results])
        # Even-indexed variants (0, 2, 4, 6, 8) are PASS → positions 100, 102, ...
        nt.assert_array_equal(positions, [100, 102, 104, 106, 108])

    def test_include_filter_ne_pass(self):
        results = list(self._chunks(_make_filter_vcz(), include='FILTER!="PASS"'))
        positions = np.concatenate([chunk["variant_position"] for chunk in results])
        # Odd-indexed variants are q10 → positions 101, 103, ...
        nt.assert_array_equal(positions, [101, 103, 105, 107])

    def test_exclude_filter_pass(self):
        results = list(self._chunks(_make_filter_vcz(), exclude='FILTER="PASS"'))
        positions = np.concatenate([chunk["variant_position"] for chunk in results])
        nt.assert_array_equal(positions, [101, 103, 105, 107])

    def test_filter_subset_match(self):
        results = list(self._chunks(_make_filter_vcz(), include='FILTER~"q10"'))
        positions = np.concatenate([chunk["variant_position"] for chunk in results])
        nt.assert_array_equal(positions, [101, 103, 105, 107])

    def test_filter_with_regions(self):
        root = _make_filter_vcz()
        reader = make_reader(root, regions="chr1:104-107", include='FILTER="PASS"')
        results = list(reader.variant_chunks(fields=["variant_position"]))
        positions = np.concatenate([chunk["variant_position"] for chunk in results])
        # Positions 104-107 intersected with PASS (even) → 104, 106
        nt.assert_array_equal(positions, [104, 106])


class TestReadaheadStringField:
    """Multi-chunk pipeline must handle a variable-length string
    FORMAT field — the path where the prior static estimator
    undershot, and where the first-chunk measurement provides a
    realistic per-chunk budget.
    """

    def _make_vcz(self):
        vlen = np.array(
            [["a", "bb"], ["ccc", "dddd"], ["eeeee", "ffffff"]] * 3,
            dtype=np.dtypes.StringDType(),
        )
        return vcz_builder.make_vcz(
            variant_contig=[0] * 9,
            variant_position=list(range(100, 109)),
            alleles=[("A", "T")] * 9,
            num_samples=2,
            variants_chunk_size=3,
            call_fields={"AB": vlen},
        )

    def test_iterates_string_format_field(self):
        root = self._make_vcz()
        reader = VczReader(root)
        chunks = list(reader.variant_chunks(fields=["variant_position", "call_AB"]))
        assert len(chunks) == 3
        positions = np.concatenate([c["variant_position"] for c in chunks])
        ab = np.concatenate([c["call_AB"] for c in chunks], axis=0)
        nt.assert_array_equal(positions, list(range(100, 109)))
        nt.assert_array_equal(
            ab,
            [["a", "bb"], ["ccc", "dddd"], ["eeeee", "ffffff"]] * 3,
        )


def _make_string_field_vcz(num_variants=9, variants_chunk_size=3):
    """Multi-chunk VCZ with a variable-length string FORMAT field, used
    by the readahead-budget sweep to exercise the bootstrap measurement
    on a variable-width prefetch.
    """
    vlen = np.array(
        [["a", "bb"], ["ccc", "dddd"], ["eeeee", "ffffff"]] * (num_variants // 3),
        dtype=np.dtypes.StringDType(),
    )
    return vcz_builder.make_vcz(
        variant_contig=[0] * num_variants,
        variant_position=list(range(100, 100 + num_variants)),
        alleles=[("A", "T")] * num_variants,
        num_samples=2,
        variants_chunk_size=variants_chunk_size,
        proportional_chunks=False,
        call_fields={"AB": vlen},
    )


class TestReadaheadBudgetSweep:
    """``readahead_bytes`` controls the cross-chunk prefetch window.

    Sweeps across the budget range against a 9-variant / 3-chunk
    fixture: ``0`` (depth pinned at 1, the smallest legal budget),
    ``1`` (exercises the ``max(1, per_chunk_bytes)`` guard and the
    early-out branch of the budget loop), a budget that admits all
    remaining chunks once measured, an effectively unbounded budget,
    and the default (``None`` → :data:`DEFAULT_READAHEAD_BYTES`).

    Iteration must produce the full, correct sequence regardless of
    budget.
    """

    SWEEP = pytest.mark.parametrize(
        "readahead_bytes",
        [0, 1, 100, 10**9, None],
        ids=["zero", "one", "small", "large", "default"],
    )

    @SWEEP
    def test_single_field(self, readahead_bytes):
        root = _make_filter_vcz(num_variants=9, variants_chunk_size=3)
        reader = VczReader(root, readahead_bytes=readahead_bytes)
        chunks = list(reader.variant_chunks(fields=["variant_position"]))
        assert len(chunks) == 3
        positions = np.concatenate([c["variant_position"] for c in chunks])
        nt.assert_array_equal(positions, list(range(100, 109)))

    @SWEEP
    def test_string_field(self, readahead_bytes):
        # Variable-length string FORMAT field — bootstrap measurement
        # has to include the heap content via utils.array_memory_bytes, and
        # later-chunk drift can over- or under-shoot the budget.
        root = _make_string_field_vcz()
        reader = VczReader(root, readahead_bytes=readahead_bytes)
        chunks = list(reader.variant_chunks(fields=["variant_position", "call_AB"]))
        assert len(chunks) == 3
        positions = np.concatenate([c["variant_position"] for c in chunks])
        ab = np.concatenate([c["call_AB"] for c in chunks], axis=0)
        nt.assert_array_equal(positions, list(range(100, 109)))
        nt.assert_array_equal(
            ab,
            [["a", "bb"], ["ccc", "dddd"], ["eeeee", "ffffff"]] * 3,
        )


class TestMaterializeStringDTypeOwnData:
    """Regression for the SIGSEGV/SIGABRT in vcf_writer.write_vcf at
    sustained scale on long_bench.

    Under proportional chunking, a single variant-only StringDType
    block can back many stream chunks (``variant_allele`` chunks[0] =
    163,000 vs. min_chunk = 1,000 on long_bench). Each stream chunk
    pulls a basic-indexing slice of the shared block, which numpy 2
    returns as a *view* sharing the block's StringDType arena. The
    prefetch worker and the main thread then race on parallel
    StringDType operations against that arena — a numpy 2.4 hazard
    that surfaces as heap corruption after a few million records.
    :meth:`CachedLogicalVariantsChunk._materialize` is the chokepoint;
    it must hand back StringDType arrays that own their data.
    """

    def _make_vcz(self, *, num_variants=12, variants_chunk_size=3, info_chunk_size=12):
        """One info field with StringDType data, chunked larger than
        ``variants_chunk_size`` so the same block backs every stream
        chunk."""
        info = np.array(
            [f"impact_{i}" for i in range(num_variants)],
            dtype=np.dtypes.StringDType(),
        )
        return vcz_builder.make_vcz(
            variant_contig=[0] * num_variants,
            variant_position=list(range(100, 100 + num_variants)),
            alleles=[("A", "T")] * num_variants,
            variants_chunk_size=variants_chunk_size,
            info_fields={"IMPACT": info},
            field_chunk_overrides={"variant_IMPACT": info_chunk_size},
            proportional_chunks=False,
        )

    def test_yielded_string_arrays_own_their_data(self):
        root = self._make_vcz()
        reader = VczReader(root)
        chunks = list(
            reader.variant_chunks(fields=["variant_position", "variant_IMPACT"])
        )
        assert len(chunks) == 4
        for chunk in chunks:
            impact = chunk["variant_IMPACT"]
            assert impact.dtype.kind == "T"
            assert impact.flags.owndata, (
                "StringDType arrays from variant_chunks must own their "
                "data: a view that shares its arena with a multi-chunk "
                "shared block races with the prefetch worker."
            )


class TestVariantChunksReadOnly:
    """Arrays yielded by ``variant_chunks`` must be marked read-only.

    The reader hands back views into shared raw blocks, the cached
    static-field arrays, and StringDType buffers whose arena is shared
    with the prefetch worker. Mutating any of these would corrupt the
    next stream chunk or race the producer thread. The read-only flag
    is set at the yield chokepoint in ``_variant_chunks_gen``.
    """

    FIELDS = [
        "variant_position",
        "call_genotype",
        "variant_IMPACT",
        "variant_index",
    ]

    @staticmethod
    def _make_vcz(num_variants=9, variants_chunk_size=3):
        """Multi-chunk VCZ exercising a 1-D variant-axis field
        (``variant_position``), a 3-D call_* field
        (``call_genotype``), and a StringDType variant-axis info field
        (``variant_IMPACT``)."""
        num_samples = 2
        ploidy = 2
        impact = np.array(
            [f"impact_{i}" for i in range(num_variants)],
            dtype=np.dtypes.StringDType(),
        )
        call_genotype = np.zeros((num_variants, num_samples, ploidy), dtype=np.int8)
        return vcz_builder.make_vcz(
            variant_contig=[0] * num_variants,
            variant_position=list(range(100, 100 + num_variants)),
            alleles=[("A", "T")] * num_variants,
            num_samples=num_samples,
            variants_chunk_size=variants_chunk_size,
            proportional_chunks=False,
            call_genotype=call_genotype,
            info_fields={"IMPACT": impact},
        )

    def test_all_arrays_read_only(self):
        root = self._make_vcz()
        reader = VczReader(root)
        chunks = list(reader.variant_chunks(fields=self.FIELDS))
        assert len(chunks) == 3
        for chunk in chunks:
            for name in self.FIELDS:
                assert not chunk[name].flags.writeable, f"{name} must be read-only"

    @pytest.mark.parametrize("field", FIELDS)
    def test_write_raises(self, field):
        root = self._make_vcz()
        reader = VczReader(root)
        chunk = next(reader.variant_chunks(fields=self.FIELDS))
        arr = chunk[field]
        with pytest.raises(ValueError, match="read-only"):
            arr.flat[0] = arr.flat[0]

    def test_sample_filter_pass_read_only(self, fx_sample_vcz):
        reader = make_reader(
            fx_sample_vcz.group,
            regions="20:1230236-",
            samples=["NA00002", "NA00003"],
            include="FMT/DP>3",
            view_semantics=True,
        )
        chunk = next(
            reader.variant_chunks(
                fields=["variant_position", "call_DP"],
            )
        )
        mask = chunk["sample_filter_pass"]
        assert not mask.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            mask[0, 0] = not mask[0, 0]

    def test_static_field_read_only_across_yields(self):
        # ``filter_id`` is a static (no variants axis) field; the same
        # array object is yielded on every stream chunk. The read-only
        # flag must hold for every yield, not just the first.
        root = _make_filter_vcz(num_variants=9, variants_chunk_size=3)
        reader = VczReader(root)
        chunks = list(reader.variant_chunks(fields=["variant_position", "filter_id"]))
        assert len(chunks) == 3
        first = chunks[0]["filter_id"]
        for chunk in chunks:
            assert chunk["filter_id"] is first
            assert not chunk["filter_id"].flags.writeable

    def test_copy_is_writeable(self):
        # Documented escape hatch: callers that need to mutate must
        # ``.copy()`` first. The copy must be writeable — read-only is
        # a per-array flag, not a property of the underlying buffer.
        root = self._make_vcz()
        reader = VczReader(root)
        chunk = next(reader.variant_chunks(fields=["variant_position"]))
        writable = chunk["variant_position"].copy()
        assert writable.flags.writeable
        writable[0] = 0

    def test_variant_index_only_read_only(self):
        # Pseudo-field requested alone: ``real_query_fields`` is empty,
        # ``_absolute_variant_indexes`` is the only producer. The
        # emitted array must still be marked read-only at the yield
        # site even though no Zarr-backed field went through
        # ``output_view``.
        root = self._make_vcz()
        reader = VczReader(root)
        chunks = list(reader.variant_chunks(fields=["variant_index"]))
        assert len(chunks) == 3
        for chunk in chunks:
            idx = chunk["variant_index"]
            assert idx.dtype == np.int64
            assert not idx.flags.writeable
            with pytest.raises(ValueError, match="read-only"):
                idx[0] = 0
        nt.assert_array_equal(
            np.concatenate([c["variant_index"] for c in chunks]),
            np.arange(9),
        )

    def test_variant_index_after_filter_selection_read_only(self):
        # When a variant filter drops rows the per-chunk pseudo-field
        # array is fancy-indexed with the surviving-rows mask
        # (``value = value[variants_selection]``) before yield. That
        # post-selection result is a fresh owning array, and it must
        # also be flagged read-only.
        root = self._make_vcz()
        reader = VczReader(root)
        reader.set_variant_filter(
            BcftoolsFilter(field_names=reader.field_names, include="POS>=105")
        )
        emitted = []
        for chunk in reader.variant_chunks(fields=["variant_index"]):
            idx = chunk["variant_index"]
            assert not idx.flags.writeable
            with pytest.raises(ValueError, match="read-only"):
                idx[0] = 0
            emitted.append(idx)
        # variant_position starts at 100; POS>=105 keeps indexes 5..8.
        nt.assert_array_equal(np.concatenate(emitted), [5, 6, 7, 8])

    def test_variant_index_after_set_variants_read_only(self):
        # Sparse global indexes drive ``_absolute_variant_indexes`` down
        # its ndarray-selection branch (non-contiguous local indexes
        # within a chunk). The result must also be read-only.
        root = self._make_vcz()
        reader = VczReader(root)
        wanted = np.array([0, 2, 4, 7], dtype=np.int64)
        reader.set_variants(wanted)
        emitted = []
        for chunk in reader.variant_chunks(fields=["variant_index"]):
            idx = chunk["variant_index"]
            assert not idx.flags.writeable
            with pytest.raises(ValueError, match="read-only"):
                idx[0] = 0
            emitted.append(idx)
        nt.assert_array_equal(np.concatenate(emitted), wanted)

    @staticmethod
    def _make_sample_scope_vcz(num_variants=6, num_samples=3, variants_chunk_size=3):
        """Multi-chunk VCZ with a deterministic ``call_DP`` shaped so
        sample-scope filter outcomes are easy to compute by hand:
        ``DP[v, s] = v + s``, giving distinct sample masks per variant
        for filters like ``FMT/DP>=k``."""
        call_dp = np.add.outer(
            np.arange(num_variants, dtype=np.int32),
            np.arange(num_samples, dtype=np.int32),
        )
        return vcz_builder.make_vcz(
            variant_contig=[0] * num_variants,
            variant_position=list(range(100, 100 + num_variants)),
            alleles=[("A", "T")] * num_variants,
            num_samples=num_samples,
            call_genotype=np.zeros((num_variants, num_samples, 2), dtype=np.int8),
            call_fields={"DP": call_dp},
            variants_chunk_size=variants_chunk_size,
            proportional_chunks=False,
        )

    def test_sample_filter_pass_no_column_reindex_read_only(self):
        # Sample-scope filter where filter samples == output samples:
        # ``output_columns`` is None, so only the row slice
        # ``filter_result[variants_selection]`` fires. The slice
        # result must still be read-only.
        root = self._make_sample_scope_vcz()
        reader = VczReader(root)
        reader.set_variant_filter(
            BcftoolsFilter(field_names=reader.field_names, include="FMT/DP>=4")
        )
        chunks = list(reader.variant_chunks(fields=["variant_position"]))
        # DP[v, s] = v + s; DP>=4 drops v=0, v=1 (no sample passes);
        # v=2..5 each contribute one row.
        emitted_mask = np.concatenate([c["sample_filter_pass"] for c in chunks])
        nt.assert_array_equal(
            emitted_mask,
            [
                [False, False, True],  # v=2: 2,3,4
                [False, True, True],  # v=3: 3,4,5
                [True, True, True],  # v=4: 4,5,6
                [True, True, True],  # v=5: 5,6,7
            ],
        )
        for chunk in chunks:
            mask = chunk["sample_filter_pass"]
            assert not mask.flags.writeable
            with pytest.raises(ValueError, match="read-only"):
                mask[0, 0] = not mask[0, 0]

    def test_sample_filter_pass_all_rows_pass_read_only(self):
        # Filter where every row passes: ``variants_selection`` is
        # all-True, ``filter_result[variants_selection]`` is a fresh
        # owning array equal to ``filter_result``. Combined with
        # ``output_columns`` reindex (view_semantics + sample subset),
        # both branches fire on a full-survivor input.
        root = self._make_sample_scope_vcz()
        reader = make_reader(
            root,
            samples=[1, 2],
            include="FMT/DP>=2",
            view_semantics=True,
        )
        chunks = list(reader.variant_chunks(fields=["variant_position"]))
        emitted_mask = np.concatenate([c["sample_filter_pass"] for c in chunks])
        # All 6 variants survive (each has at least one DP>=2 sample);
        # output axis is samples [1, 2] only — column subset of the
        # 6x3 filter result above (drop column 0).
        nt.assert_array_equal(
            emitted_mask,
            [
                [False, True],  # v=0: DP=[1,2], samples [1,2]: [F,T]
                [True, True],  # v=1: DP=[2,3], samples [1,2]: [T,T]
                [True, True],  # v=2
                [True, True],  # v=3
                [True, True],  # v=4
                [True, True],  # v=5
            ],
        )
        for chunk in chunks:
            mask = chunk["sample_filter_pass"]
            assert not mask.flags.writeable
            with pytest.raises(ValueError, match="read-only"):
                mask[0, 0] = not mask[0, 0]

    def test_sample_filter_pass_absent_for_variant_scope_filter(self):
        # Variant-scope filters produce a 1-D mask and never publish
        # ``sample_filter_pass``. Regression guard against a future
        # change that always emits the key.
        root = self._make_sample_scope_vcz()
        reader = VczReader(root)
        reader.set_variant_filter(
            BcftoolsFilter(field_names=reader.field_names, include="POS>=103")
        )
        chunks = list(reader.variant_chunks(fields=["variant_position"]))
        assert len(chunks) > 0
        for chunk in chunks:
            assert "sample_filter_pass" not in chunk


class TestVczReaderArraysReadOnly:
    """Every numpy array ``VczReader`` exposes to a caller must be
    read-only.

    Many of these arrays are cached and reused: ``contig_ids``,
    ``filter_ids``, ``sample_ids``, ``raw_sample_ids``, etc. are
    ``@cached_property``; ``samples_selection`` / ``sample_ids`` are
    set once in ``set_samples`` (or the lazy default resolver) and
    then handed back on every subsequent property access. A mutation
    by a caller would silently change what every later caller sees.
    """

    # Array-valued attributes on VczReader. Each entry is
    # (attr_name, optional). "Optional" attrs may legally return None
    # for a store that lacks the underlying field; the test skips the
    # read-only assertion when the value is None.
    ATTRS = [
        ("contig_ids", False),
        ("filter_ids", False),
        ("contigs", False),
        ("filters", False),
        ("raw_sample_ids", False),
        ("non_null_sample_indices", False),
        ("sample_ids", False),
        ("samples_selection", False),
        ("contig_lengths", True),
        ("filter_descriptions", True),
        ("region_index", True),
    ]

    @pytest.mark.parametrize(("attr", "optional"), ATTRS)
    def test_property_read_only(self, fx_sample_vcz, attr, optional):
        reader = VczReader(fx_sample_vcz.group)
        value = getattr(reader, attr)
        if value is None:
            if not optional:
                pytest.fail(f"{attr} unexpectedly returned None")
            pytest.skip(f"{attr} not present in fixture")
        assert isinstance(value, np.ndarray), f"{attr} must be ndarray"
        assert not value.flags.writeable, f"{attr} must be read-only"
        with pytest.raises(ValueError, match="read-only"):
            value.flat[0] = value.flat[0]

    def test_variant_counts_per_chunk_read_only(self, fx_sample_vcz):
        reader = VczReader(fx_sample_vcz.group)
        counts = reader.variant_counts_per_chunk()
        assert not counts.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            counts[0] = 0

    def test_cached_property_returns_same_object(self, fx_sample_vcz):
        # Sanity: the read-only flag set on a ``cached_property`` array
        # is the same array on every subsequent access. Regression
        # guard against a future refactor that returns a fresh
        # (writeable) array each time.
        reader = VczReader(fx_sample_vcz.group)
        assert reader.contig_ids is reader.contig_ids
        assert reader.raw_sample_ids is reader.raw_sample_ids

    def test_set_samples_does_not_freeze_caller_input(self, fx_sample_vcz):
        # ``set_samples`` must own its internal copy: freezing
        # ``samples_selection`` for the reader's cache must not silently
        # lock the caller's input array. Regression guard for the
        # ``np.array`` (copy) vs. ``np.asarray`` (view) choice in
        # ``_normalize_sample_indexes``.
        reader = VczReader(fx_sample_vcz.group)
        caller_input = np.array([0, 2], dtype=np.int64)
        reader.set_samples(caller_input)
        assert not reader.samples_selection.flags.writeable
        # Caller's original input is still writeable.
        assert caller_input.flags.writeable
        caller_input[0] = 1  # must not raise

    def test_static_field_cache_returns_read_only(self, fx_sample_vcz):
        # ``_load_static_field`` results are also exposed indirectly
        # via ``variant_chunks``; the cache itself must hand back
        # read-only arrays so the first reader of the cache cannot
        # corrupt later readers.
        reader = VczReader(fx_sample_vcz.group)
        arr = reader._load_static_field("filter_id")
        assert not arr.flags.writeable
        # Repeated load returns the same (still-frozen) object.
        assert reader._load_static_field("filter_id") is arr


def _make_stream_reader(
    root,
    *,
    readahead_bytes=10**9,
    read_fields=None,
    n_chunks=None,
    executor=None,
    stream_chunk_size=None,
):
    """Construct a :class:`StreamReader` directly against ``root``,
    matching the wiring ``VczReader.variant_chunks`` does (default
    sample-chunk plan over non-null samples; one ``ChunkRead`` per
    stream chunk; no view-mode column remap).

    ``executor`` is the thread pool the reader submits reads to. The
    caller is responsible for shutting it down (e.g. via a ``with``
    block); ``None`` means "build a small pool here", which is the
    common case for tests that only need the reader to run once and
    don't share the executor with other readers.
    """
    if read_fields is None:
        read_fields = ["variant_position"]
    if executor is None:
        executor = cf.ThreadPoolExecutor(max_workers=2)
    samples_chunk_size = int(root["sample_id"].chunks[0])
    raw_sample_ids = root["sample_id"][:]
    samples_selection = np.flatnonzero(raw_sample_ids != "")
    sample_chunk_plan = samples_mod.build_chunk_plan(
        samples_selection, samples_chunk_size=samples_chunk_size
    )
    min_chunk = utils.compute_min_variants_chunk_size(root)
    if stream_chunk_size is None:
        stream_chunk_size = utils.compute_stream_chunk_size(
            root, read_fields, min_chunk
        )
    num_variants = int(root["variant_position"].shape[0])
    full_stream_chunks = (num_variants + stream_chunk_size - 1) // stream_chunk_size
    if n_chunks is None:
        n_chunks = full_stream_chunks
    n_chunks = min(n_chunks, full_stream_chunks)
    plan_length = min(n_chunks * stream_chunk_size, num_variants)
    stream_plan = utils.ChunkRead.simple_plan(plan_length, stream_chunk_size)
    return retrieval_mod.StreamReader(
        root,
        stream_plan,
        sample_chunk_plan,
        None,
        read_fields,
        readahead_bytes=readahead_bytes,
        executor=executor,
        stream_chunk_size=stream_chunk_size,
    )


class _DepthTrackingStreamReader(retrieval_mod.StreamReader):
    """StreamReader subclass that records ``len(_live)`` after each
    ``_submit_more`` call. Used to assert depth-control behaviour under
    different ``readahead_bytes`` values without observing the executor
    directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.depths: list[int] = []

    def _submit_more(self, must_advance_through):
        super()._submit_more(must_advance_through)
        self.depths.append(len(self._live))


def _vcz_for_template_tests():
    """Small VCZ exposing all four field shapes in the templates:

    - 1-D static (``sample_id``)
    - 1-D variant-axis non-call (``variant_position``)
    - 2-D variant-axis non-call (``variant_allele``)
    - 2-D ``call_*`` (``call_DP``)
    - 3-D ``call_*`` (``call_genotype``)

    Uses uniform chunks (no proportional recipe) because the consumers
    are unit tests that assert on chunk-derived quantities (multiplier,
    block_size_estimate, intra-slice math).
    """
    num_variants = 4
    num_samples = 4
    ploidy = 2
    call_dp = np.zeros((num_variants, num_samples), dtype=np.int32)
    call_genotype = np.zeros((num_variants, num_samples, ploidy), dtype=np.int8)
    return vcz_builder.make_vcz(
        variant_contig=[0] * num_variants,
        variant_position=list(range(1, num_variants + 1)),
        alleles=[("A", "T")] * num_variants,
        num_samples=num_samples,
        sample_id=[f"s{i}" for i in range(num_samples)],
        variants_chunk_size=2,
        samples_chunk_size=2,
        proportional_chunks=False,
        call_genotype=call_genotype,
        call_fields={"DP": call_dp},
    )


class TestMakeFieldSpecs:
    """``_make_field_specs`` records the per-field constants the
    stream reader needs to derive blocks on the fly: the field's array,
    its chunk multiplier over the stream chunk, whether it's a
    ``call_*`` field, an upper-bound block-size estimate, and the
    trailing block-index suffix tuple."""

    def test_static_field_rejected(self):
        root = _vcz_for_template_tests()
        with pytest.raises(AssertionError, match="non-variants-axis"):
            retrieval_mod._make_field_specs(root, ["sample_id"], stream_chunk_size=2)

    def test_variant_axis_1d_field(self):
        root = _vcz_for_template_tests()
        [spec] = retrieval_mod._make_field_specs(
            root, ["variant_position"], stream_chunk_size=2
        )
        assert spec.field == "variant_position"
        assert spec.arr == root["variant_position"]
        assert spec.is_call is False
        assert spec.multiplier == 1
        # 1-D variants axis → no extra dims after the variant chunk slot.
        assert spec.block_index_suffix == ()

    def test_variant_axis_2d_field(self):
        root = _vcz_for_template_tests()
        [spec] = retrieval_mod._make_field_specs(
            root, ["variant_allele"], stream_chunk_size=2
        )
        # 2-D (variants, alleles) → one trailing slice(None).
        assert spec.block_index_suffix == (slice(None),)

    def test_call_field_2d_suffix(self):
        root = _vcz_for_template_tests()
        [spec] = retrieval_mod._make_field_specs(root, ["call_DP"], stream_chunk_size=2)
        assert spec.is_call is True
        # 2-D (variants, samples) → suffix is empty: sample-chunk index
        # is the second slot, no trailing slice.
        assert spec.block_index_suffix == ()

    def test_call_field_3d_suffix(self):
        root = _vcz_for_template_tests()
        [spec] = retrieval_mod._make_field_specs(
            root, ["call_genotype"], stream_chunk_size=2
        )
        # 3-D (variants, samples, ploidy) → suffix carries one slice.
        assert spec.block_index_suffix == (slice(None),)

    def test_multiplier_for_scaled_up_variant_field(self):
        # variant_allele chunked at 2 * stream_chunk_size.
        root = vcz_builder.make_vcz(
            variant_contig=[0] * 4,
            variant_position=[1, 2, 3, 4],
            alleles=[("A", "T")] * 4,
            num_samples=2,
            sample_id=["s0", "s1"],
            variants_chunk_size=2,
            samples_chunk_size=2,
            proportional_chunks=False,
            call_genotype=np.zeros((4, 2, 2), dtype=np.int8),
            field_chunk_overrides={"variant_allele": 4},
        )
        specs = retrieval_mod._make_field_specs(
            root, ["variant_position", "variant_allele"], stream_chunk_size=2
        )
        assert [s.multiplier for s in specs] == [1, 2]

    def test_non_divisible_stream_chunk_size_rejected(self):
        # variant_allele chunked at 6 is not a multiple of stream_chunk_size=4.
        # _make_field_specs guards the invariant that
        # compute_stream_chunk_size must return a divisor of every read
        # field's chunks[0].
        root = vcz_builder.make_vcz(
            variant_contig=[0] * 6,
            variant_position=[1, 2, 3, 4, 5, 6],
            alleles=[("A", "T")] * 6,
            num_samples=2,
            sample_id=["s0", "s1"],
            variants_chunk_size=2,
            samples_chunk_size=2,
            call_genotype=np.zeros((6, 2, 2), dtype=np.int8),
            field_chunk_overrides={"variant_allele": 6},
        )
        with pytest.raises(ValueError, match="not a multiple of"):
            retrieval_mod._make_field_specs(
                root, ["variant_allele"], stream_chunk_size=4
            )

    def test_block_size_estimate_from_dtype_and_chunks(self):
        # variant_position: chunks=(2,), dtype int32 → 2 * 4 = 8 bytes/block.
        # call_genotype: chunks=(2, 2, 2), dtype int8 → 8 bytes/block.
        # call_DP: chunks=(2, 2), dtype int32 → 16 bytes/block.
        root = _vcz_for_template_tests()
        specs = retrieval_mod._make_field_specs(
            root,
            ["variant_position", "call_genotype", "call_DP"],
            stream_chunk_size=2,
        )
        assert specs[0].block_size_estimate == 8
        assert specs[1].block_size_estimate == 8
        assert specs[2].block_size_estimate == 16


class TestStreamReaderDeriveBlocks:
    """``StreamReader._derive_blocks`` builds the per-stream-chunk
    block records: one per non-``call_*`` field, one per
    ``(call_*, sample_chunk)`` pair."""

    def _make_reader(
        self,
        root,
        fields,
        *,
        stream_chunk_size=None,
        samples_selection=None,
    ):
        if stream_chunk_size is None:
            stream_chunk_size = int(root["variant_position"].chunks[0])
        samples_chunk_size = int(root["sample_id"].chunks[0])
        if samples_selection is None:
            non_null = np.flatnonzero(root["sample_id"][:] != "")
            samples_selection = non_null
        plan = samples_mod.build_chunk_plan(
            samples_selection, samples_chunk_size=samples_chunk_size
        )
        executor = cf.ThreadPoolExecutor(max_workers=2)
        return executor, retrieval_mod.StreamReader(
            root,
            utils.ChunkRead.simple_plan(
                int(root["variant_position"].shape[0]), stream_chunk_size
            ),
            plan,
            None,
            fields,
            readahead_bytes=0,
            executor=executor,
            stream_chunk_size=stream_chunk_size,
        )

    def test_non_call_field_one_record_per_chunk(self):
        root = _vcz_for_template_tests()
        executor, reader = self._make_reader(root, ["variant_position"])
        with executor:
            records = reader._derive_blocks(utils.ChunkRead(index=3, num_selected=2))
        assert len(records) == 1
        rec = records[0]
        assert rec.dest_key == ("variant_position",)
        assert rec.block_key == (("variant_position",), 3)
        assert rec.block_index == (3,)
        assert rec.intra_slice == slice(0, 2)

    def test_2d_non_call_field_keeps_trailing_slice(self):
        root = _vcz_for_template_tests()
        executor, reader = self._make_reader(root, ["variant_allele"])
        with executor:
            records = reader._derive_blocks(utils.ChunkRead(index=1, num_selected=2))
        rec = records[0]
        assert rec.dest_key == ("variant_allele",)
        assert rec.block_index == (1, slice(None))

    def test_call_field_fans_out_per_sample_chunk(self):
        root = _vcz_for_template_tests()
        # 4 samples, samples_chunk_size=2 → 2 sample chunks; 3-D call_genotype.
        executor, reader = self._make_reader(root, ["call_genotype"])
        with executor:
            records = reader._derive_blocks(utils.ChunkRead(index=0, num_selected=2))
        assert [r.dest_key for r in records] == [
            ("call_genotype", 0),
            ("call_genotype", 1),
        ]
        assert [r.block_index for r in records] == [
            (0, 0, slice(None)),
            (0, 1, slice(None)),
        ]
        for rec in records:
            assert rec.block_key == (rec.dest_key, 0)
            assert rec.intra_slice == slice(0, 2)

    def test_call_field_2d_no_trailing_slice(self):
        root = _vcz_for_template_tests()
        executor, reader = self._make_reader(root, ["call_DP"])
        with executor:
            records = reader._derive_blocks(utils.ChunkRead(index=0, num_selected=2))
        # 2-D call_DP → block_index slots are (variant, sample_chunk).
        assert [r.block_index for r in records] == [(0, 0), (0, 1)]

    def test_scaled_up_variant_field_translates_block_and_intra_slice(self):
        # variant_allele has chunks[0] = 6 = 3 * stream_chunk_size(2).
        # Stream chunk 5 → variant_block_idx = 5 // 3 = 1,
        # intra_start = (5 % 3) * 2 = 4 → intra_slice = slice(4, 6).
        root = vcz_builder.make_vcz(
            variant_contig=[0] * 18,
            variant_position=list(range(1, 19)),
            alleles=[("A", "T")] * 18,
            num_samples=2,
            sample_id=["s0", "s1"],
            variants_chunk_size=2,
            samples_chunk_size=2,
            call_genotype=np.zeros((18, 2, 2), dtype=np.int8),
            field_chunk_overrides={"variant_allele": 6},
        )
        executor, reader = self._make_reader(
            root, ["variant_allele"], stream_chunk_size=2
        )
        with executor:
            records = reader._derive_blocks(utils.ChunkRead(index=5, num_selected=2))
        assert records[0].block_index == (1, slice(None))
        assert records[0].intra_slice == slice(4, 6)
        # block_key is (dest_key, variant_block_idx).
        assert records[0].block_key == (("variant_allele",), 1)


class TestStreamReader:
    """Direct unit tests for ``retrieval.StreamReader``.

    The end-to-end suites cover correctness; this class targets the
    reader's own state machine — submission ordering, byte-budget
    depth control, single-future-per-block invariant, executor cleanup,
    and behaviour at the edges (empty plan, empty read columns).
    """

    @staticmethod
    def _vcz(num_variants=12, variants_chunk_size=3, num_samples=2):
        return vcz_builder.make_vcz(
            variant_contig=[0] * num_variants,
            variant_position=list(range(100, 100 + num_variants)),
            alleles=[("A", "T")] * num_variants,
            num_samples=num_samples,
            variants_chunk_size=variants_chunk_size,
            proportional_chunks=False,
        )

    def test_yields_one_chunk_per_plan_entry_in_order(self):
        root = self._vcz()
        reader = _make_stream_reader(root)
        indexes = [chunk.variant_chunk.index for chunk in reader]
        assert indexes == [0, 1, 2, 3]

    def test_empty_plan_yields_nothing(self):
        root = self._vcz()
        reader = _make_stream_reader(root, n_chunks=0)
        assert list(reader) == []
        # No live blocks after a clean drain.
        assert reader._live == {}

    def test_single_chunk_plan(self):
        root = self._vcz(num_variants=3, variants_chunk_size=3)
        reader = _make_stream_reader(root)
        chunks = list(reader)
        assert len(chunks) == 1
        assert chunks[0].variant_chunk.index == 0

    def test_readahead_bytes_zero_keeps_depth_one(self):
        # Budget=0 → after every submit, exactly one chunk's blocks are
        # live ahead of the consumer (and zero on the final, plan-exhausted
        # submit).
        root = self._vcz(num_variants=12, variants_chunk_size=3)
        with cf.ThreadPoolExecutor(max_workers=2) as executor:
            reader = _DepthTrackingStreamReader(
                root,
                [utils.ChunkRead(index=i, num_selected=3) for i in range(4)],
                samples_mod.build_chunk_plan(
                    np.arange(2, dtype=np.int64), samples_chunk_size=2
                ),
                None,
                ["variant_position"],
                readahead_bytes=0,
                executor=executor,
                stream_chunk_size=3,
            )
            list(reader)
        # Initial submit + post-yield submit per chunk = 5 entries.
        # Each leaves exactly one (or zero, at the end) block in flight.
        assert reader.depths == [1, 1, 1, 1, 0]

    def test_large_readahead_submits_all_remaining(self):
        # Budget of 10**9 dwarfs the per-block cost, so the initial
        # submit schedules every chunk's block in one go.
        root = self._vcz(num_variants=12, variants_chunk_size=3)
        with cf.ThreadPoolExecutor(max_workers=2) as executor:
            reader = _DepthTrackingStreamReader(
                root,
                [utils.ChunkRead(index=i, num_selected=3) for i in range(4)],
                samples_mod.build_chunk_plan(
                    np.arange(2, dtype=np.int64), samples_chunk_size=2
                ),
                None,
                ["variant_position"],
                readahead_bytes=10**9,
                executor=executor,
                stream_chunk_size=3,
            )
            list(reader)
        # 4 blocks submitted initially, then yielded and evicted one at a time.
        assert reader.depths == [4, 3, 2, 1, 0]

    def test_max_in_flight_tracks_peak_depth(self):
        root = self._vcz(num_variants=12, variants_chunk_size=3)
        reader = _make_stream_reader(root, readahead_bytes=10**9)
        assert reader.max_in_flight == 0
        list(reader)
        assert reader.max_in_flight == 4

    def test_max_in_flight_pinned_at_one_with_zero_budget(self):
        root = self._vcz(num_variants=12, variants_chunk_size=3)
        reader = _make_stream_reader(root, readahead_bytes=0)
        list(reader)
        assert reader.max_in_flight == 1

    def test_empty_read_fields_does_not_infinite_loop(self):
        # With no fields to prefetch each chunk derives zero records;
        # the reader still advances the cursor and yields empty
        # CachedLogicalVariantsChunks.
        root = self._vcz(num_variants=6, variants_chunk_size=3)
        reader = _make_stream_reader(root, read_fields=[], readahead_bytes=10**9)
        chunks = list(reader)
        assert len(chunks) == 2
        for chunk in chunks:
            assert chunk._blocks == {}

    def test_chunks_have_prefetched_blocks(self):
        # Every dest_key for the read fields lands in chunk._blocks
        # before the consumer receives it.
        root = self._vcz(num_variants=6, variants_chunk_size=3, num_samples=2)
        reader = _make_stream_reader(
            root,
            read_fields=["variant_position", "variant_contig"],
            readahead_bytes=0,
        )
        for chunk in reader:
            assert ("variant_position",) in chunk._blocks
            assert ("variant_contig",) in chunk._blocks

    def test_executor_outlives_full_iteration(self):
        # The reader does not own the executor; full drain leaves
        # the pool alive and ready to serve another reader.
        root = self._vcz()
        with cf.ThreadPoolExecutor(max_workers=2) as executor:
            reader = _make_stream_reader(root, executor=executor)
            list(reader)
            assert executor._shutdown is False
            # Pool is still usable for a second reader.
            second = _make_stream_reader(root, executor=executor)
            assert len(list(second)) > 0

    def test_pending_futures_cancelled_on_early_break(self):
        # Abandoning iteration cancels still-pending futures (those
        # that hadn't started); the executor itself stays alive. A
        # future that was already running when ``cancel()`` was called
        # cannot be cancelled — wait briefly for it to drain so the
        # final state is deterministic.
        root = self._vcz(num_variants=24, variants_chunk_size=3)
        with cf.ThreadPoolExecutor(max_workers=2) as executor:
            reader = _make_stream_reader(root, executor=executor, readahead_bytes=10**9)
            gen = iter(reader)
            next(gen)
            in_flight_snapshot = [fut for fut, _ in reader._live.values()]
            gen.close()
            assert executor._shutdown is False
            cf.wait(in_flight_snapshot, timeout=5.0)
            for fut in in_flight_snapshot:
                assert fut.cancelled() or fut.done()

    def test_shared_block_submitted_once_across_stream_chunks(self, monkeypatch):
        # variant_allele chunks span 4 stream chunks. The single block
        # must be submitted exactly once even though 4 stream chunks
        # reference it.
        num_variants = 8
        root = vcz_builder.make_vcz(
            variant_contig=[0] * num_variants,
            variant_position=list(range(1, num_variants + 1)),
            alleles=[("A", "T")] * num_variants,
            num_samples=2,
            sample_id=["s0", "s1"],
            variants_chunk_size=2,
            samples_chunk_size=2,
            call_genotype=np.zeros((num_variants, 2, 2), dtype=np.int8),
            field_chunk_overrides={"variant_allele": 8},
        )
        submitted: list[tuple] = []
        original = retrieval_mod._read_block

        def counting_read_block(arr, block_index):
            submitted.append((getattr(arr, "name", repr(arr)), block_index))
            return original(arr, block_index)

        monkeypatch.setattr(retrieval_mod, "_read_block", counting_read_block)
        reader = _make_stream_reader(
            root,
            read_fields=["variant_allele", "call_genotype"],
            readahead_bytes=10**9,
            stream_chunk_size=2,
        )
        list(reader)
        variant_allele_submits = [s for s in submitted if "variant_allele" in s[0]]
        # 4 stream chunks visit the same variant_allele block; the
        # single-future-per-block invariant means we submit it once.
        assert len(variant_allele_submits) == 1

    def test_shared_block_evicted_after_last_use(self):
        # Same layout as the dedup test; after each yield the next
        # stream chunk's keys are peeked and current-only keys are
        # dropped from _live. After the final stream chunk the live
        # set must be empty.
        num_variants = 8
        root = vcz_builder.make_vcz(
            variant_contig=[0] * num_variants,
            variant_position=list(range(1, num_variants + 1)),
            alleles=[("A", "T")] * num_variants,
            num_samples=2,
            sample_id=["s0", "s1"],
            variants_chunk_size=2,
            samples_chunk_size=2,
            call_genotype=np.zeros((num_variants, 2, 2), dtype=np.int8),
            field_chunk_overrides={"variant_allele": 8},
        )
        reader = _make_stream_reader(
            root,
            read_fields=["variant_allele", "call_genotype"],
            readahead_bytes=10**9,
            stream_chunk_size=2,
        )
        list(reader)
        assert reader._live == {}
        assert reader._in_flight_bytes == 0

    def test_shared_block_advances_when_budget_blocks_next_chunk(self):
        # Mixed-chunk-shape regression: a variant-only field with a
        # multiplier > 1 keeps one big block live across stream chunks,
        # while a call_* field fans out to many sample chunks per stream
        # chunk. After yielding stream chunk 0, eviction removes only
        # the call_* blocks; the shared variant-only block stays. If the
        # readahead budget is below
        # ``shared_block_bytes + next_chunk_call_bytes`` the budget gate
        # would refuse to submit chunk 1 even though the consumer is
        # about to demand it. The reader must override the budget in
        # that case — otherwise the iterator looks up an unsubmitted
        # block_key and raises KeyError.
        num_variants = 4
        num_samples = 8
        root = vcz_builder.make_vcz(
            variant_contig=[0] * num_variants,
            variant_position=list(range(1, num_variants + 1)),
            alleles=[("A", "T")] * num_variants,
            num_samples=num_samples,
            sample_id=[f"s{i}" for i in range(num_samples)],
            variants_chunk_size=2,
            samples_chunk_size=1,
            call_genotype=np.zeros((num_variants, num_samples, 2), dtype=np.int8),
            field_chunk_overrides={"variant_allele": 4},
            proportional_chunks=False,
        )
        reader = _make_stream_reader(
            root,
            read_fields=["variant_allele", "call_genotype"],
            readahead_bytes=0,
            stream_chunk_size=2,
        )
        chunks = list(reader)
        assert len(chunks) == 2
        # Every stream chunk got its call_genotype sample-chunk blocks
        # plus the shared variant_allele block.
        for chunk in chunks:
            assert ("variant_allele",) in chunk._blocks
            for sci in range(num_samples):
                assert ("call_genotype", sci) in chunk._blocks


class TestVczReaderBackendsEndToEnd:
    """All four storage backends read the same local-directory VCZ identically.

    Single canonical place to confirm that VczReader works end-to-end
    over local / fsspec / obstore / icechunk. Uses the v3 fixture
    because icechunk requires Zarr v3; the other three backends read
    v2/v3 transparently.
    """

    @pytest.fixture
    def fx_root(self, request, fx_sample_vcz3, tmp_path):
        backend = request.param
        directory = fx_sample_vcz3.directory_path
        if backend == "icechunk":
            ic_path = to_vcz_icechunk(directory, tmp_path)
            return utils.open_zarr(ic_path, backend_storage="icechunk")
        return utils.open_zarr(directory, backend_storage=backend)

    @pytest.mark.parametrize(
        "fx_root", [None, "fsspec", "obstore", "icechunk"], indirect=True
    )
    def test_variants_iteration(self, fx_root):
        reader = VczReader(fx_root)
        it = reader.variants(
            fields=["variant_contig", "variant_position", "call_genotype"]
        )

        v1 = next(it)
        assert v1["variant_contig"] == 0  # first contig in the table is "19"
        assert v1["variant_position"] == 111
        nt.assert_array_equal(v1["call_genotype"], [[0, 0], [0, 0], [0, 1]])

        v2 = next(it)
        assert v2["variant_contig"] == 0
        assert v2["variant_position"] == 112


class TestVczReaderRegions:
    """Cover the three accepted region/target input shapes plus error paths.

    Region/target parsing now lives in
    :func:`vcztools.regions.build_chunk_plan`; these tests exercise it
    directly for validation and round-trip the happy paths through
    :class:`VczReader` via the ``make_reader`` helper.
    """

    @staticmethod
    def _vcz():
        # 10 variants on chr1 at positions 1..10.
        return vcz_builder.make_vcz(
            variant_contig=[0] * 10,
            variant_position=list(range(1, 11)),
            alleles=[("A", "T")] * 10,
            contigs=("chr1",),
            variants_chunk_size=3,
        )

    def _positions(self, reader):
        chunks = list(reader.variant_chunks(fields=["variant_position"]))
        if not chunks:
            return np.array([], dtype=np.int32)
        return np.concatenate([c["variant_position"] for c in chunks])

    def _build_chunk_plan(self, **kwargs):
        """Run :func:`regions.build_chunk_plan` against a fresh reader
        built from :meth:`_vcz`. Used by the rejection-path tests that
        only need to reach the input validator."""
        with VczReader(self._vcz()) as reader:
            return regions_mod.build_chunk_plan(reader, **kwargs)

    def test_regions_string(self):
        reader = make_reader(self._vcz(), regions="chr1:3-5")
        nt.assert_array_equal(self._positions(reader), [3, 4, 5])

    def test_regions_list_of_strings(self):
        reader = make_reader(self._vcz(), regions=["chr1:3-5", "chr1:8-9"])
        nt.assert_array_equal(self._positions(reader), [3, 4, 5, 8, 9])

    def test_regions_dataframe(self):
        df = pd.DataFrame(
            {
                "contig": ["chr1"],
                "start": pd.array([3], dtype="Int64"),
                "end": pd.array([5], dtype="Int64"),
            }
        )
        reader = make_reader(self._vcz(), regions=df)
        nt.assert_array_equal(self._positions(reader), [3, 4, 5])

    def test_regions_dataframe_with_na_end(self):
        df = pd.DataFrame(
            {
                "contig": ["chr1"],
                "start": pd.array([8], dtype="Int64"),
                "end": pd.array([pd.NA], dtype="Int64"),
            }
        )
        reader = make_reader(self._vcz(), regions=df)
        nt.assert_array_equal(self._positions(reader), [8, 9, 10])

    def test_targets_complement_flag(self):
        reader = make_reader(self._vcz(), targets="chr1:3-5", targets_complement=True)
        nt.assert_array_equal(self._positions(reader), [1, 2, 6, 7, 8, 9, 10])

    def test_regions_rejects_caret_prefix(self):
        with pytest.raises(ValueError, match="targets_complement=True"):
            self._build_chunk_plan(regions="^chr1:1-3")

    def test_targets_rejects_caret_prefix(self):
        with pytest.raises(ValueError, match="targets_complement=True"):
            self._build_chunk_plan(targets="^chr1:1-3")

    def test_regions_rejects_comma(self):
        with pytest.raises(ValueError, match=r"regions string .* contains ','"):
            self._build_chunk_plan(regions="chr1:1-3,chr1:5-7")

    def test_targets_rejects_comma(self):
        with pytest.raises(ValueError, match=r"targets string .* contains ','"):
            self._build_chunk_plan(targets="chr1:1-3,chr1:5-7")

    def test_regions_invalid_type(self):
        with pytest.raises(TypeError, match="regions must be"):
            self._build_chunk_plan(regions=42)

    def test_targets_invalid_type(self):
        with pytest.raises(TypeError, match="targets must be"):
            self._build_chunk_plan(targets=42)

    def test_regions_dataframe_missing_columns(self):
        df = pd.DataFrame({"contig": ["chr1"], "start": pd.array([1], dtype="Int64")})
        with pytest.raises(ValueError, match="missing required columns.*end"):
            self._build_chunk_plan(regions=df)

    def test_flat_index_array_accepted(self):
        """``set_variants(np.ndarray)`` buckets indexes into a plan."""
        reader = VczReader(self._vcz())
        reader.set_variants(np.array([2, 4, 7], dtype=np.int64))
        nt.assert_array_equal(self._positions(reader), [3, 5, 8])


class TestVczReaderSamples:
    """Cover VczReader sample input: default, integer-index list, error cases."""

    def test_samples_default_selects_all(self, fx_sample_vcz):
        reader = VczReader(fx_sample_vcz.group)
        nt.assert_array_equal(reader.sample_ids, ["NA00001", "NA00002", "NA00003"])

    def test_samples_list(self, fx_sample_vcz):
        reader = VczReader(fx_sample_vcz.group)
        reader.set_samples([0, 2])
        nt.assert_array_equal(reader.sample_ids, ["NA00001", "NA00003"])
        nt.assert_array_equal(reader.samples_selection, [0, 2])

    def test_samples_preserves_input_order(self, fx_sample_vcz):
        reader = VczReader(fx_sample_vcz.group)
        reader.set_samples([2, 0])
        nt.assert_array_equal(reader.sample_ids, ["NA00003", "NA00001"])
        nt.assert_array_equal(reader.samples_selection, [2, 0])

    def test_samples_rejects_string_input(self, fx_sample_vcz):
        reader = VczReader(fx_sample_vcz.group)
        with pytest.raises(TypeError, match="integer indexes"):
            reader.set_samples(["NA00001"])

    def test_samples_rejects_string_scalar(self, fx_sample_vcz):
        reader = VczReader(fx_sample_vcz.group)
        with pytest.raises(TypeError, match="integer indexes"):
            reader.set_samples("NA00001")

    def test_samples_rejects_non_integer_numpy_array(self, fx_sample_vcz):
        reader = VczReader(fx_sample_vcz.group)
        with pytest.raises(TypeError, match="integer indexes"):
            reader.set_samples(np.array([0.0, 2.0]))

    def test_samples_accepts_numpy_int_array(self, fx_sample_vcz):
        reader = VczReader(fx_sample_vcz.group)
        reader.set_samples(np.array([0, 2], dtype=np.int64))
        nt.assert_array_equal(reader.sample_ids, ["NA00001", "NA00003"])

    def test_samples_out_of_range_raises(self, fx_sample_vcz):
        reader = VczReader(fx_sample_vcz.group)
        with pytest.raises(ValueError, match="sample index out of range"):
            reader.set_samples([0, 99])

    def test_samples_negative_raises(self, fx_sample_vcz):
        reader = VczReader(fx_sample_vcz.group)
        with pytest.raises(ValueError, match="sample index out of range"):
            reader.set_samples([-1])

    def test_samples_empty_list(self, fx_sample_vcz):
        # Post-resolve, an empty list means "no samples" (e.g. all
        # requested names were dropped by ignore_missing_samples).
        reader = VczReader(fx_sample_vcz.group)
        reader.set_samples([])
        nt.assert_array_equal(reader.sample_ids, [])
        assert reader.sample_chunk_plan.chunk_reads == []
        assert reader.sample_chunk_plan.permutation is None


class TestVczReaderSampleChunks:
    """End-to-end sample-chunk pruning.

    Builds a VCZ whose sample axis spans multiple chunks and verifies
    that selecting subsets yields correct ``call_*`` data regardless of
    whether the selection hits one chunk, several chunks, or the
    last (possibly partial) chunk.
    """

    @staticmethod
    def _vcz(num_samples=6, samples_chunk_size=2):
        sample_ids = [f"s{i}" for i in range(num_samples)]
        # call_DP[i, j] = i * 10 + j — unique per (variant, sample) so
        # mis-indexing is caught immediately.
        call_dp = np.array(
            [[i * 10 + j for j in range(num_samples)] for i in range(3)],
            dtype=np.int32,
        )
        return vcz_builder.make_vcz(
            variant_contig=[0, 0, 0],
            variant_position=[1, 2, 3],
            alleles=[("A", "T")] * 3,
            num_samples=num_samples,
            sample_id=sample_ids,
            samples_chunk_size=samples_chunk_size,
            call_fields={"DP": call_dp},
        )

    def _call_dp(self, reader):
        chunks = list(reader.variant_chunks(fields=["call_DP"]))
        return np.concatenate([c["call_DP"] for c in chunks], axis=0)

    @staticmethod
    def _plan_chunk_indexes(plan):
        return [cr.index for cr in plan.chunk_reads]

    def test_single_chunk_selection(self):
        # s2, s3 both live in sample chunk 1 (indexes 2, 3) and
        # cover the full chunk → selection collapses to None to skip
        # a no-op fancy-index gather in CachedVariantChunk.
        reader = VczReader(self._vcz())
        reader.set_samples([2, 3])
        plan = reader.sample_chunk_plan
        assert self._plan_chunk_indexes(plan) == [1]
        assert plan.chunk_reads[0].selection is None
        assert plan.permutation is None
        dp = self._call_dp(reader)
        nt.assert_array_equal(dp, [[2, 3], [12, 13], [22, 23]])

    def test_multi_chunk_selection(self):
        # s1 is in chunk 0; s4 is in chunk 2; chunk 1 is skipped.
        reader = VczReader(self._vcz())
        reader.set_samples([1, 4])
        plan = reader.sample_chunk_plan
        assert self._plan_chunk_indexes(plan) == [0, 2]
        assert plan.permutation is None
        dp = self._call_dp(reader)
        nt.assert_array_equal(dp, [[1, 4], [11, 14], [21, 24]])

    def test_preserves_user_order(self):
        # Same chunks as the multi-chunk test, but the user order is
        # reversed — the output must follow the input list via the
        # plan's permutation.
        reader = VczReader(self._vcz())
        reader.set_samples([4, 1])
        plan = reader.sample_chunk_plan
        assert self._plan_chunk_indexes(plan) == [0, 2]
        nt.assert_array_equal(plan.permutation, [1, 0])
        dp = self._call_dp(reader)
        nt.assert_array_equal(dp, [[4, 1], [14, 11], [24, 21]])

    def test_partial_final_chunk(self):
        # 5 samples with chunk size 2 → chunks sized [2, 2, 1]. s4 sits
        # alone in the final chunk; the contiguous local index
        # collapses to slice(0, 1) (basic indexing → view).
        reader = VczReader(self._vcz(num_samples=5, samples_chunk_size=2))
        reader.set_samples([4])
        plan = reader.sample_chunk_plan
        assert self._plan_chunk_indexes(plan) == [2]
        assert plan.chunk_reads[0].selection == slice(0, 1)
        assert plan.permutation is None
        dp = self._call_dp(reader)
        nt.assert_array_equal(dp, [[4], [14], [24]])

    def test_default_plan_is_identity(self):
        # Default (no set_samples call) plans a read over every sample
        # chunk with each full-chunk selection.
        reader = VczReader(self._vcz())
        plan = reader.sample_chunk_plan
        assert self._plan_chunk_indexes(plan) == [0, 1, 2]
        assert plan.permutation is None
        dp = self._call_dp(reader)
        nt.assert_array_equal(dp.shape, (3, 6))
        nt.assert_array_equal(dp[0], [0, 1, 2, 3, 4, 5])

    def test_default_masked_header_reduces(self):
        # Default with masked "" entries in sample_id: the plan drops
        # the masked indices, matching the old post-filter path.
        root = self._vcz()
        root["sample_id"][:2] = ""
        reader = VczReader(root)
        nt.assert_array_equal(reader.sample_ids, ["s2", "s3", "s4", "s5"])
        dp = self._call_dp(reader)
        nt.assert_array_equal(dp, [[2, 3, 4, 5], [12, 13, 14, 15], [22, 23, 24, 25]])

    def test_empty_samples_list_produces_empty_plan(self):
        # An empty samples list (via --force-samples dropping every
        # requested unknown, or --drop-genotypes) produces an empty
        # sample chunk plan — no chunks to read, no sample columns
        # in output.
        reader = VczReader(self._vcz())
        reader.set_samples([])
        assert reader.sample_chunk_plan.chunk_reads == []
        assert reader.sample_chunk_plan.permutation is None


def _vcz_for_cache_tests(
    num_samples=4, samples_chunk_size=2, variants_chunk_size=2, num_variants=4
):
    call_dp = np.array(
        [[v * 10 + s for s in range(num_samples)] for v in range(num_variants)],
        dtype=np.int32,
    )
    return vcz_builder.make_vcz(
        variant_contig=[0] * num_variants,
        variant_position=list(range(1, num_variants + 1)),
        alleles=[("A", "T")] * num_variants,
        num_samples=num_samples,
        sample_id=[f"s{i}" for i in range(num_samples)],
        variants_chunk_size=variants_chunk_size,
        samples_chunk_size=samples_chunk_size,
        proportional_chunks=False,
        call_fields={"DP": call_dp},
    )


def _make_cached_chunk(
    root,
    *,
    fields=(),
    variant_chunk_idx=0,
    variant_selection=None,
    samples_selection=None,
    view_semantics=False,
):
    raw_sample_ids = root["sample_id"][:]
    non_null_indices = np.flatnonzero(raw_sample_ids != "")
    samples_chunk_size = int(root["sample_id"].chunks[0])
    if samples_selection is None:
        samples_selection = non_null_indices
    samples_selection = np.asarray(samples_selection, dtype=np.int64)
    subset_plan = samples_mod.build_chunk_plan(
        samples_selection, samples_chunk_size=samples_chunk_size
    )
    non_null_plan = samples_mod.build_chunk_plan(
        non_null_indices, samples_chunk_size=samples_chunk_size
    )
    if view_semantics:
        sample_chunk_plan = non_null_plan
        output_columns = np.searchsorted(non_null_indices, samples_selection)
    else:
        sample_chunk_plan = subset_plan
        output_columns = None
    stream_chunk_size = int(root["variant_position"].chunks[0])
    num_variants = int(root["variant_position"].shape[0])
    if variant_selection is None:
        base = variant_chunk_idx * stream_chunk_size
        variant_num_selected = min(stream_chunk_size, num_variants - base)
    elif isinstance(variant_selection, slice):
        variant_num_selected = variant_selection.stop - variant_selection.start
    else:
        variant_num_selected = len(variant_selection)
    variant_chunk = utils.ChunkRead(
        index=variant_chunk_idx,
        num_selected=variant_num_selected,
        selection=variant_selection,
    )
    # Build the same block records the StreamReader would, then read
    # synchronously and intra-slice — mirrors the StreamReader's
    # CachedLogicalVariantsChunk handoff but without the executor.
    executor = cf.ThreadPoolExecutor(max_workers=1)
    with executor:
        reader = retrieval_mod.StreamReader(
            root,
            [variant_chunk],
            sample_chunk_plan,
            output_columns,
            list(fields),
            readahead_bytes=0,
            executor=executor,
            stream_chunk_size=stream_chunk_size,
        )
        records = reader._derive_blocks(variant_chunk)
        blocks = {
            rec.dest_key: retrieval_mod._read_block(rec.arr, rec.block_index)[
                rec.intra_slice
            ]
            for rec in records
        }
    return CachedLogicalVariantsChunk(
        root,
        variant_chunk,
        sample_chunk_plan=sample_chunk_plan,
        output_columns=output_columns,
        blocks=blocks,
        stream_chunk_size=stream_chunk_size,
    )


class TestCachedLogicalVariantsChunkCache:
    """CachedLogicalVariantsChunk consumes prefetched, intra-sliced blocks and
    caches assembled views so a field reused across filter_view /
    output_view is materialized once."""

    def test_view_mode_prefetch_covers_every_real_sample_chunk(self):
        # View-mode prefetch covers the whole real sample axis (so a
        # sample-scope filter evaluating on the real axis has every
        # block it needs). output_view's column slice never reaches
        # back into ``_blocks`` — it slices the filter_view.
        root = _vcz_for_cache_tests(num_samples=6, samples_chunk_size=2)
        chunk = _make_cached_chunk(
            root,
            fields=["call_DP"],
            samples_selection=np.array([1, 5]),
            view_semantics=True,
        )
        prefetched = {k[1] for k in chunk._blocks if k[0] == "call_DP"}
        assert prefetched == {0, 1, 2}

    def test_subset_mode_filter_and_output_view_share_assembled_array(self):
        # Subset-mode: filter_view and output_view use the same plan
        # object → they share a single assembled array in the view
        # cache, not merely shared prefetched blocks.
        root = _vcz_for_cache_tests()
        chunk = _make_cached_chunk(root, fields=["call_DP"])
        fv = chunk.filter_view("call_DP")
        ov = chunk.output_view("call_DP")
        assert fv is ov

    def test_variant_field_cached_within_chunk(self):
        chunk = _make_cached_chunk(_vcz_for_cache_tests(), fields=["variant_position"])
        first = chunk.filter_view("variant_position")
        second = chunk.output_view("variant_position")
        assert first is second


class TestCachedLogicalVariantsChunkAxes:
    """filter_view and output_view return data in the right sample axis."""

    @staticmethod
    def _vcz():
        # 4 samples, all real; variants_chunk covers both variants.
        num_samples = 4
        num_variants = 2
        call_dp = np.array(
            [[v * 10 + s for s in range(num_samples)] for v in range(num_variants)],
            dtype=np.int32,
        )
        return vcz_builder.make_vcz(
            variant_contig=[0] * num_variants,
            variant_position=list(range(1, num_variants + 1)),
            alleles=[("A", "T")] * num_variants,
            num_samples=num_samples,
            sample_id=[f"s{i}" for i in range(num_samples)],
            variants_chunk_size=num_variants,
            samples_chunk_size=2,
            proportional_chunks=False,
            call_fields={"DP": call_dp},
        )

    def test_subset_mode_filter_view_uses_subset_axis(self):
        chunk = _make_cached_chunk(
            self._vcz(),
            fields=["call_DP"],
            samples_selection=np.array([0, 2]),
        )
        dp = chunk.filter_view("call_DP")
        assert dp.shape == (2, 2)
        nt.assert_array_equal(dp, [[0, 2], [10, 12]])

    def test_view_mode_filter_view_uses_real_axis(self):
        chunk = _make_cached_chunk(
            self._vcz(),
            fields=["call_DP"],
            samples_selection=np.array([0, 2]),
            view_semantics=True,
        )
        dp = chunk.filter_view("call_DP")
        assert dp.shape == (2, 4)
        nt.assert_array_equal(dp, [[0, 1, 2, 3], [10, 11, 12, 13]])

    @pytest.mark.parametrize("view_semantics", [False, True])
    def test_output_view_always_subset_axis(self, view_semantics):
        chunk = _make_cached_chunk(
            self._vcz(),
            fields=["call_DP"],
            samples_selection=np.array([0, 2]),
            view_semantics=view_semantics,
        )
        dp = chunk.output_view("call_DP")
        assert dp.shape == (2, 2)
        nt.assert_array_equal(dp, [[0, 2], [10, 12]])


class TestVariantChunksFilterPlusSamples:
    """End-to-end: a call_* field referenced by both filter and query,
    with sample subsetting active, preserves bcftools semantics."""

    def test_custom_variant_filter_no_bcftools(self, fx_sample_vcz):
        # Hand-rolled filter implementing the VariantFilter protocol —
        # proves VczReader has no bcftools-specific coupling.
        class PositionGt:
            referenced_fields = frozenset({"variant_position"})
            scope = "variant"

            def __init__(self, threshold):
                self._threshold = threshold

            def evaluate(self, chunk_data):
                return chunk_data["variant_position"] > self._threshold

        reader = VczReader(fx_sample_vcz.group)
        reader.set_variant_filter(PositionGt(1000000))
        chunks = list(reader.variant_chunks(fields=["variant_position"]))
        positions = np.concatenate([c["variant_position"] for c in chunks])
        nt.assert_array_equal(positions, [1110696, 1230237, 1234567, 1235237])

    def test_default_filter_samples_sample_scope(self, fx_sample_vcz):
        # Default filter axis (no full_sample_filter) is the sample
        # subset — bcftools-query-style post-subset evaluation.
        # Position 1234567 (where NA00001 but not NA00002/NA00003
        # matched FMT/DP>3) is DROPPED.
        reader = VczReader(fx_sample_vcz.group)
        reader.set_samples([1, 2])
        reader.set_variants(regions_mod.build_chunk_plan(reader, regions="20:1230236-"))
        reader.set_variant_filter(
            BcftoolsFilter(field_names=reader.field_names, include="FMT/DP>3"),
        )
        chunks = list(reader.variant_chunks(fields=["variant_position", "call_DP"]))
        chunk = chunks[0]
        assert chunk["call_DP"].shape[1] == 2
        nt.assert_array_equal(chunk["variant_position"], [1230237])

    def test_explicit_filter_samples_no_subset_is_noop(self, fx_sample_vcz):
        # With no sample subset, view semantics must return identical
        # results to the default.
        root = fx_sample_vcz.group

        def build(view_semantics):
            reader = VczReader(root)
            reader.set_variants(
                regions_mod.build_chunk_plan(reader, regions="20:1230236-")
            )
            if view_semantics:
                reader.set_bcftools_semantics(full_sample_filter=True)
            reader.set_variant_filter(
                BcftoolsFilter(field_names=reader.field_names, include="FMT/DP>3"),
            )
            return reader

        pre = list(build(True).variant_chunks(fields=["variant_position"]))
        post = list(build(False).variant_chunks(fields=["variant_position"]))
        nt.assert_array_equal(
            np.concatenate([c["variant_position"] for c in pre]),
            np.concatenate([c["variant_position"] for c in post]),
        )

    def test_explicit_filter_samples_variant_scope_unchanged(self, fx_sample_vcz):
        # Variant-scope filters touch no sample axis; the filter-samples
        # setter is a no-op regardless of subset.
        root = fx_sample_vcz.group

        def build(view_semantics):
            reader = VczReader(root)
            reader.set_samples([0])
            if view_semantics:
                reader.set_bcftools_semantics(full_sample_filter=True)
            reader.set_variant_filter(
                BcftoolsFilter(field_names=reader.field_names, include="POS > 1000000"),
            )
            return reader

        pre = list(build(True).variant_chunks(fields=["variant_position"]))
        post = list(build(False).variant_chunks(fields=["variant_position"]))
        nt.assert_array_equal(
            np.concatenate([c["variant_position"] for c in pre]),
            np.concatenate([c["variant_position"] for c in post]),
        )

    def test_filter_sees_full_samples_output_is_pruned(self, fx_sample_vcz):
        # Locks in that variants can be included because non-selected
        # samples matched the filter, while the returned call_* arrays
        # are sample-pruned — exercising the single-reader path where
        # both views coexist on one field.
        reader = make_reader(
            fx_sample_vcz.group,
            regions="20:1230236-",
            samples=["NA00002", "NA00003"],
            include="FMT/DP>3",
            view_semantics=True,
        )
        chunks = list(reader.variant_chunks(fields=["variant_position", "call_DP"]))
        chunk = chunks[0]
        # Sample-pruned shape: 2 samples (NA00002, NA00003).
        assert chunk["call_DP"].shape[1] == 2
        # Position 1234567 survives only because NA00001 (not in the
        # selection) matched the filter — same lock-in as test_variant_chunks
        # but via the unified reader.
        nt.assert_array_equal(chunk["variant_position"], [1230237, 1234567])


class TestVczReaderMissingSamplesMultiChunk:
    """VczReader with a many-chunk sample axis and a large fraction of
    masked (``sample_id == ""``) samples.

    The store has 50 samples across 10 sample chunks with 30 masked
    indices arranged to exercise a fully-masked chunk, a fully-real
    chunk, mixed chunks, and masked indices on chunk boundaries.
    ``call_DP[v, s] = v*100 + s`` gives every cell a unique value so
    assertions are arithmetic; ``call_GQ`` is crafted per-row so
    sample-scope filter tests have predictable match sets.

    Bcftools has no notion of a missing sample — masked slots are a VCZ
    extension. Correct behaviour is therefore that masked data never
    influences filter evaluation, regardless of the filter axis.
    Tests that exercise that invariant today fail and are marked xfail;
    their xfail markers come off when the production path is fixed.
    """

    NUM_SAMPLES = 50
    NUM_VARIANTS = 15
    SAMPLES_CHUNK = 5
    VARIANTS_CHUNK = 5

    MASKED = np.array(
        [
            6,
            7,
            8,  # chunk 1: keep {5, 9}
            11,
            12,
            13,
            14,  # chunk 2: keep {10}
            15,
            16,
            17,
            18,
            19,  # chunk 3: fully masked
            21,
            22,
            23,  # chunk 4: keep {20, 24}
            26,
            27,
            28,
            29,  # chunk 5: keep {25}
            34,  # chunk 6: boundary (last), keep 30-33
            35,
            36,  # chunk 7: boundary (first), keep 37-39
            41,
            42,
            43,
            44,  # chunk 8: keep {40}
            45,
            46,
            47,
            48,  # chunk 9: keep {49}
        ]
    )
    REAL = np.setdiff1d(np.arange(NUM_SAMPLES), MASKED)

    @classmethod
    def _vcz(cls):
        sample_ids = np.array([f"s{i}" for i in range(cls.NUM_SAMPLES)], dtype="<U16")
        sample_ids[cls.MASKED] = ""

        v_idx = np.arange(cls.NUM_VARIANTS, dtype=np.int32)[:, None]
        s_idx = np.arange(cls.NUM_SAMPLES, dtype=np.int32)[None, :]
        call_dp = v_idx * 100 + s_idx

        call_gq = np.zeros((cls.NUM_VARIANTS, cls.NUM_SAMPLES), dtype=np.int32)
        # Row 10 — only masked samples cross "FMT/GQ > 50".
        call_gq[10, cls.MASKED] = 100
        # Row 11 — every real sample matches.
        call_gq[11, cls.REAL] = 100
        # Row 12 — only real sample s25 matches (outside the test subset).
        call_gq[12, 25] = 100
        # Row 13 — only real sample s37 matches (inside the test subset).
        call_gq[13, 37] = 100

        return vcz_builder.make_vcz(
            variant_contig=[0] * cls.NUM_VARIANTS,
            variant_position=list(range(1000, 1000 + cls.NUM_VARIANTS)),
            alleles=[("A", "T")] * cls.NUM_VARIANTS,
            num_samples=cls.NUM_SAMPLES,
            sample_id=sample_ids,
            samples_chunk_size=cls.SAMPLES_CHUNK,
            variants_chunk_size=cls.VARIANTS_CHUNK,
            call_fields={"DP": call_dp, "GQ": call_gq},
        )

    @staticmethod
    def _plan_indexes(plan):
        return [cr.index for cr in plan.chunk_reads]

    def test_default_drops_all_masked_samples(self):
        root = self._vcz()
        reader = VczReader(root)
        expected_ids = [f"s{i}" for i in self.REAL.tolist()]
        nt.assert_array_equal(reader.sample_ids, expected_ids)
        chunks = list(reader.variant_chunks(fields=["call_DP"]))
        dp = np.concatenate([c["call_DP"] for c in chunks], axis=0)
        assert dp.shape == (self.NUM_VARIANTS, self.REAL.size)
        v_idx = np.arange(self.NUM_VARIANTS)[:, None]
        expected_dp = v_idx * 100 + self.REAL[None, :]
        nt.assert_array_equal(dp, expected_dp)
        # Fully-masked chunk 3 is absent from the plan; every other chunk
        # appears exactly once.
        assert self._plan_indexes(reader.sample_chunk_plan) == [
            0,
            1,
            2,
            4,
            5,
            6,
            7,
            8,
            9,
        ]

    def test_subset_spanning_many_chunks_with_missing(self):
        reader = VczReader(self._vcz())
        subset = [0, 10, 20, 37, 49]
        reader.set_samples(subset)
        plan = reader.sample_chunk_plan
        assert self._plan_indexes(plan) == [0, 2, 4, 7, 9]
        assert plan.permutation is None
        # Each per-chunk selection is a single contiguous index, so it
        # collapses to a slice (basic indexing → view).
        expected_slices = [
            slice(0, 1),
            slice(0, 1),
            slice(0, 1),
            slice(2, 3),
            slice(4, 5),
        ]
        for cr, expected in zip(plan.chunk_reads, expected_slices):
            assert cr.selection == expected
        chunks = list(reader.variant_chunks(fields=["call_DP"]))
        dp = np.concatenate([c["call_DP"] for c in chunks], axis=0)
        v_idx = np.arange(self.NUM_VARIANTS)[:, None]
        expected_dp = v_idx * 100 + np.array(subset)[None, :]
        nt.assert_array_equal(dp, expected_dp)

    def test_subset_with_user_order_permutation(self):
        reader = VczReader(self._vcz())
        subset = [49, 37, 20, 10, 0]
        reader.set_samples(subset)
        plan = reader.sample_chunk_plan
        assert self._plan_indexes(plan) == [0, 2, 4, 7, 9]
        nt.assert_array_equal(plan.permutation, [4, 3, 2, 1, 0])
        chunks = list(reader.variant_chunks(fields=["call_DP"]))
        dp = np.concatenate([c["call_DP"] for c in chunks], axis=0)
        v_idx = np.arange(self.NUM_VARIANTS)[:, None]
        expected_dp = v_idx * 100 + np.array(subset)[None, :]
        nt.assert_array_equal(dp, expected_dp)

    def test_subset_inside_partial_chunk(self):
        reader = VczReader(self._vcz())
        subset = [5, 9]
        reader.set_samples(subset)
        plan = reader.sample_chunk_plan
        assert self._plan_indexes(plan) == [1]
        nt.assert_array_equal(plan.chunk_reads[0].selection, [0, 4])
        assert plan.permutation is None
        chunks = list(reader.variant_chunks(fields=["call_DP"]))
        dp = np.concatenate([c["call_DP"] for c in chunks], axis=0)
        v_idx = np.arange(self.NUM_VARIANTS)[:, None]
        expected_dp = v_idx * 100 + np.array(subset)[None, :]
        nt.assert_array_equal(dp, expected_dp)

    def test_variant_scope_filter_with_subset_and_missing(self):
        root = self._vcz()
        subset_names = ["s0", "s5", "s20", "s40"]
        subset_indexes = [0, 5, 20, 40]
        reader = make_reader(
            root,
            samples=subset_names,
            include="POS > 1007",
        )
        chunks = list(reader.variant_chunks(fields=["variant_position", "call_DP"]))
        positions = np.concatenate([c["variant_position"] for c in chunks])
        nt.assert_array_equal(positions, list(range(1008, 1015)))
        dp = np.concatenate([c["call_DP"] for c in chunks], axis=0)
        assert dp.shape == (7, 4)
        v_range = np.arange(8, 15)[:, None]
        expected_dp = v_range * 100 + np.array(subset_indexes)[None, :]
        nt.assert_array_equal(dp, expected_dp)

    def test_sample_scope_filter_pre_subset_ignores_masked(self):
        root = self._vcz()
        reader = make_reader(
            root,
            samples=["s0", "s10", "s20", "s37", "s49"],
            include="FMT/GQ > 50",
            view_semantics=True,
        )
        chunks = list(reader.variant_chunks(fields=["variant_position"]))
        positions = np.concatenate([c["variant_position"] for c in chunks])
        assert 1010 not in positions.tolist()

    def test_sample_scope_filter_pre_subset_keeps_non_subset_real_match(self):
        root = self._vcz()
        subset_indexes = [0, 10, 20, 37, 49]
        reader = make_reader(
            root,
            samples=[f"s{i}" for i in subset_indexes],
            include="FMT/GQ > 50",
            view_semantics=True,
        )
        chunks = list(reader.variant_chunks(fields=["variant_position", "call_DP"]))
        positions = np.concatenate([c["variant_position"] for c in chunks])
        # Row 12's only match is real sample s25 (not in the subset); the
        # filter's variant-inclusion decision sees every real sample, so
        # the variant survives. call_* output is subset-pruned.
        assert 1012 in positions.tolist()
        idx = int(np.flatnonzero(positions == 1012)[0])
        dp = np.concatenate([c["call_DP"] for c in chunks], axis=0)
        expected_row = 12 * 100 + np.array(subset_indexes)
        nt.assert_array_equal(dp[idx], expected_row)
        sample_filter_pass = np.concatenate(
            [c["sample_filter_pass"] for c in chunks], axis=0
        )
        nt.assert_array_equal(sample_filter_pass[idx], [False] * 5)

    def test_sample_scope_filter_post_subset_sees_only_subset(self):
        root = self._vcz()
        subset_indexes = [0, 10, 20, 37, 49]
        reader = make_reader(
            root,
            samples=[f"s{i}" for i in subset_indexes],
            include="FMT/GQ > 50",
        )
        chunks = list(reader.variant_chunks(fields=["variant_position", "call_DP"]))
        positions = np.concatenate([c["variant_position"] for c in chunks])
        # Row 12 (only s25 matches — not in subset) is dropped; row 13
        # (s37 matches — in subset) is kept; row 11 (all real match) is
        # kept because subset samples are real. Row 10 (masked-only) is
        # dropped in this mode already.
        assert 1010 not in positions.tolist()
        assert 1011 in positions.tolist()
        assert 1012 not in positions.tolist()
        assert 1013 in positions.tolist()
        idx_13 = int(np.flatnonzero(positions == 1013)[0])
        dp = np.concatenate([c["call_DP"] for c in chunks], axis=0)
        nt.assert_array_equal(dp[idx_13], 13 * 100 + np.array(subset_indexes))
        sample_filter_pass = np.concatenate(
            [c["sample_filter_pass"] for c in chunks], axis=0
        )
        nt.assert_array_equal(
            sample_filter_pass[idx_13], [False, False, False, True, False]
        )

    def test_default_masking_sample_scope_filter_ignores_masked(self):
        root = self._vcz()
        variant_filter = BcftoolsFilter(
            field_names=frozenset(root.keys()), include="FMT/GQ > 50"
        )
        reader = VczReader(root)
        reader.set_variant_filter(variant_filter)
        chunks = list(reader.variant_chunks(fields=["variant_position"]))
        positions = np.concatenate([c["variant_position"] for c in chunks])
        # Correct behaviour: row 10 (masked-only match) is dropped; row
        # 11 (all-real match) is kept.
        assert 1010 not in positions.tolist()
        assert 1011 in positions.tolist()

    def test_both_modes_agree_when_filter_matches_only_subset(self):
        root = self._vcz()
        subset_indexes = [0, 10, 20, 37, 49]
        subset_names = [f"s{i}" for i in subset_indexes]

        reader_pre = make_reader(
            root,
            samples=subset_names,
            include="FMT/DP>1400",
            view_semantics=True,
        )
        reader_post = make_reader(
            root,
            samples=subset_names,
            include="FMT/DP>1400",
        )
        pre = list(reader_pre.variant_chunks(fields=["variant_position", "call_DP"]))
        post = list(reader_post.variant_chunks(fields=["variant_position", "call_DP"]))
        assert len(pre) == len(post)
        for p, q in zip(pre, post):
            nt.assert_array_equal(p["variant_position"], q["variant_position"])
            nt.assert_array_equal(p["call_DP"], q["call_DP"])
            nt.assert_array_equal(p["sample_filter_pass"], q["sample_filter_pass"])
        positions = np.concatenate([c["variant_position"] for c in pre])
        nt.assert_array_equal(positions, [1014])
        dp = np.concatenate([c["call_DP"] for c in pre], axis=0)
        nt.assert_array_equal(dp, [[1400, 1410, 1420, 1437, 1449]])
        sample_filter_pass = np.concatenate(
            [c["sample_filter_pass"] for c in pre], axis=0
        )
        nt.assert_array_equal(sample_filter_pass, [[False, True, True, True, True]])


def _basic_vcz(num_variants=3, num_samples=2, **kwargs):
    return vcz_builder.make_vcz(
        variant_contig=[0] * num_variants,
        variant_position=list(range(1, num_variants + 1)),
        alleles=[("A", "T")] * num_variants,
        num_samples=num_samples,
        **kwargs,
    )


class TestEmptyCallArray:
    """``CachedLogicalVariantsChunk._empty_call_array`` produces a zero-column
    output for ``call_*`` fields when the sample chunk plan is empty
    (e.g. ``set_samples([])``). Both the explicit-selection branch and
    the full-chunk fallback need direct coverage.
    """

    @staticmethod
    def _vcz():
        # 5 variants over chunk size 3 → chunks 0 (full, 3 rows) and
        # 1 (partial, 2 rows). Used to exercise the partial-chunk
        # arithmetic in the ``selection is None`` branch.
        num_samples = 2
        call_dp = np.zeros((5, num_samples), dtype=np.int32)
        return vcz_builder.make_vcz(
            variant_contig=[0] * 5,
            variant_position=list(range(1, 6)),
            alleles=[("A", "T")] * 5,
            num_samples=num_samples,
            samples_chunk_size=num_samples,
            variants_chunk_size=3,
            proportional_chunks=False,
            call_fields={"DP": call_dp},
        )

    def _empty_plan_chunk(self, root, *, variant_chunk_idx, selection):
        empty_plan = samples_mod.SampleChunkPlan(chunk_reads=[], permutation=None)
        chunk_size = int(root["variant_position"].chunks[0])
        num_variants = int(root["variant_position"].shape[0])
        if selection is None:
            num_selected = min(
                chunk_size, num_variants - variant_chunk_idx * chunk_size
            )
        elif isinstance(selection, slice):
            num_selected = selection.stop - selection.start
        else:
            num_selected = len(selection)
        return CachedLogicalVariantsChunk(
            root,
            utils.ChunkRead(
                index=variant_chunk_idx,
                num_selected=num_selected,
                selection=selection,
            ),
            sample_chunk_plan=empty_plan,
            output_columns=None,
            blocks={},
            stream_chunk_size=chunk_size,
        )

    def test_explicit_selection_uses_sel_size(self):
        chunk = self._empty_plan_chunk(
            self._vcz(), variant_chunk_idx=0, selection=np.array([0, 2])
        )
        out = chunk.output_view("call_DP")
        assert out.shape == (2, 0)
        assert out.dtype == np.int32

    def test_full_chunk_uses_chunk_size(self):
        chunk = self._empty_plan_chunk(self._vcz(), variant_chunk_idx=0, selection=None)
        out = chunk.output_view("call_DP")
        assert out.shape == (3, 0)

    def test_partial_final_chunk(self):
        # 5 variants, chunks of 3 → final chunk has 2 rows.
        chunk = self._empty_plan_chunk(self._vcz(), variant_chunk_idx=1, selection=None)
        out = chunk.output_view("call_DP")
        assert out.shape == (2, 0)


class TestGetFilterIdsFallback:
    """``_get_filter_ids`` falls back to a single ``b"PASS"`` entry when
    the store has no ``filter_id`` array. ``VczReader.filters``
    surfaces that fallback through its ``cached_property``."""

    def test_no_filter_id_returns_pass(self):
        root = _basic_vcz()
        del root["filter_id"]
        result = retrieval_mod._get_filter_ids(root)
        nt.assert_array_equal(result, np.array(["PASS"], dtype="S"))

    def test_reader_filters_uses_fallback(self):
        root = _basic_vcz()
        del root["filter_id"]
        reader = VczReader(root)
        nt.assert_array_equal(reader.filters, np.array(["PASS"], dtype="S"))

    def test_filter_id_present_returns_store_array(self, fx_sample_vcz):
        # Counterpart to the fallback test: when ``filter_id`` is
        # present, ``_get_filter_ids`` returns the fixed-length-bytes
        # encoding of the store array.
        result = retrieval_mod._get_filter_ids(fx_sample_vcz.group)
        assert result.dtype.kind == "S"
        assert b"PASS" in result.tolist()


def test_validate_samples_input_none():
    # ``None`` passes through every other validator; make sure the
    # helper accepts it without raising.
    assert retrieval_mod._validate_samples_input(None) is None


class TestSettersOneShot:
    """``set_samples`` and ``set_bcftools_semantics`` are one-shot — a
    second call raises ``RuntimeError``. ``set_variants`` and
    ``set_variant_filter`` are re-callable (covered by
    :class:`TestSettersReplace`)."""

    def test_set_samples_twice_raises(self, fx_sample_vcz):
        reader = VczReader(fx_sample_vcz.group)
        reader.set_samples([0])
        with pytest.raises(RuntimeError, match="samples already configured"):
            reader.set_samples([1])

    def test_set_bcftools_semantics_twice_raises(self, fx_sample_vcz):
        reader = VczReader(fx_sample_vcz.group)
        reader.set_bcftools_semantics()
        with pytest.raises(RuntimeError, match="bcftools semantics already configured"):
            reader.set_bcftools_semantics()


class TestSettersReplace:
    """``set_variants`` and ``set_variant_filter`` may be called
    multiple times; each call replaces the prior value."""

    def test_set_variants_replaces(self, fx_sample_vcz):
        reader = VczReader(fx_sample_vcz.group)
        reader.set_variants(np.array([0], dtype=np.int64))
        reader.set_variants(np.array([1, 2], dtype=np.int64))
        positions = []
        for chunk in reader.variant_chunks(fields=["variant_position"]):
            positions.extend(chunk["variant_position"].tolist())
        full = reader.root["variant_position"][:]
        assert positions == [int(full[1]), int(full[2])]

    def test_set_variant_filter_replaces(self, fx_sample_vcz):
        reader = VczReader(fx_sample_vcz.group)
        reader.set_variant_filter(
            BcftoolsFilter(field_names=reader.field_names, include="POS<0")
        )
        reader.set_variant_filter(
            BcftoolsFilter(field_names=reader.field_names, include="POS>0")
        )
        n = sum(
            len(chunk["variant_position"])
            for chunk in reader.variant_chunks(fields=["variant_position"])
        )
        assert n == reader.num_variants

    def test_set_variant_filter_none_clears(self, fx_sample_vcz):
        reader = VczReader(fx_sample_vcz.group)
        reader.set_variant_filter(
            BcftoolsFilter(field_names=reader.field_names, include="POS<0")
        )
        reader.set_variant_filter(None)
        assert reader.variant_filter is None
        n = sum(
            len(chunk["variant_position"])
            for chunk in reader.variant_chunks(fields=["variant_position"])
        )
        assert n == reader.num_variants

    def test_in_flight_generator_uses_filter_snapshot(self):
        # Two chunks of two variants each (POS = 100, 101, 102, 103).
        root = vcz_builder.make_vcz(
            variant_contig=[0] * 4,
            variant_position=[100, 101, 102, 103],
            alleles=[("A", "T")] * 4,
            variants_chunk_size=2,
        )
        reader = VczReader(root)
        reader.set_variant_filter(
            BcftoolsFilter(field_names=reader.field_names, include="POS<102")
        )
        gen = reader.variant_chunks(fields=["variant_position"])
        first = next(gen)
        # Swap to a permissive filter mid-iteration; the in-flight
        # generator should keep using the snapshot taken at start.
        reader.set_variant_filter(
            BcftoolsFilter(field_names=reader.field_names, include="POS>=0")
        )
        rest = [chunk["variant_position"].tolist() for chunk in gen]
        gen.close()
        # The snapshot filter (POS<102) keeps the first chunk in full
        # (100, 101) and excludes the second chunk entirely.
        assert first["variant_position"].tolist() == [100, 101]
        assert rest == []


class TestMaterialiseVariantFilter:
    """``VczReader.materialise_variant_filter`` resolves a variant-scope
    filter into a fixed selection: clears ``variant_filter`` and
    replaces ``variant_chunk_plan`` with one over the surviving
    indexes."""

    @staticmethod
    def _build(num_variants=5):
        # DP = [0, 10, 20, 30, 40]; INFO field, variant-scope.
        return vcz_builder.make_vcz(
            variant_contig=[0] * num_variants,
            variant_position=list(range(100, 100 + num_variants)),
            alleles=[("A", "T")] * num_variants,
            num_samples=1,
            call_genotype=np.zeros((num_variants, 1, 2), dtype=np.int8),
            info_fields={"DP": np.arange(num_variants, dtype=np.int32) * 10},
            variants_chunk_size=2,
        )

    def test_no_filter_is_noop(self):
        reader = VczReader(self._build())
        reader.materialise_variant_filter()
        assert reader.variant_filter is None

    def test_filter_is_cleared(self):
        reader = VczReader(self._build())
        reader.set_variant_filter(
            BcftoolsFilter(field_names=reader.field_names, include="DP>=20")
        )
        reader.materialise_variant_filter()
        assert reader.variant_filter is None

    def test_surviving_indexes_match_independent_eval(self):
        # DP = [0, 10, 20, 30, 40]; include "DP>=20" keeps indexes 2,3,4.
        reader = VczReader(self._build())
        reader.set_variant_filter(
            BcftoolsFilter(field_names=reader.field_names, include="DP>=20")
        )
        reader.materialise_variant_filter()
        positions = []
        for chunk in reader.variant_chunks(fields=["variant_position"]):
            positions.extend(chunk["variant_position"].tolist())
        assert positions == [102, 103, 104]

    def test_filter_excludes_all(self):
        reader = VczReader(self._build())
        reader.set_variant_filter(
            BcftoolsFilter(field_names=reader.field_names, include="DP>1000")
        )
        reader.materialise_variant_filter()
        assert reader.variant_filter is None
        assert len(reader.variant_chunk_plan) == 0

    def test_composes_with_pre_existing_set_variants(self):
        # Region/index selection already applied via set_variants;
        # filter further narrows the surviving set. Indexes 1, 3 survive
        # set_variants; of those, DP>=30 keeps only index 3.
        reader = VczReader(self._build())
        reader.set_variants(np.array([1, 3], dtype=np.int64))
        reader.set_variant_filter(
            BcftoolsFilter(field_names=reader.field_names, include="DP>=30")
        )
        reader.materialise_variant_filter()
        positions = []
        for chunk in reader.variant_chunks(fields=["variant_position"]):
            positions.extend(chunk["variant_position"].tolist())
        assert positions == [103]

    def test_sample_scope_filter_rejected(self):
        reader = VczReader(
            vcz_builder.make_vcz(
                variant_contig=[0, 0],
                variant_position=[100, 200],
                alleles=[("A", "T"), ("A", "T")],
                num_samples=2,
                call_genotype=np.zeros((2, 2, 2), dtype=np.int8),
            )
        )

        class _SampleScope:
            scope = "sample"
            referenced_fields = frozenset({"call_genotype"})

            def evaluate(self, chunk_data):
                raise NotImplementedError

        reader.set_variant_filter(_SampleScope())
        with pytest.raises(ValueError, match="Sample-scope"):
            reader.materialise_variant_filter()


class TestAbsoluteVariantIndexes:
    """``retrieval._absolute_variant_indexes`` maps a
    :class:`~vcztools.utils.ChunkRead` to the global variant indexes
    it contributes. Three branches by ``ChunkRead.selection`` shape:
    ``None`` (full chunk, length from ``num_selected``), ``slice``
    (basic-indexing range), ``ndarray`` (fancy indexing). All branches
    add ``index * variants_chunk_size`` and return ``int64``."""

    def test_selection_none_full_chunk_at_origin(self):
        entry = utils.ChunkRead(index=0, num_selected=4, selection=None)
        out = retrieval_mod._absolute_variant_indexes(entry, chunk_size=4)
        nt.assert_array_equal(out, [0, 1, 2, 3])
        assert out.dtype == np.int64

    def test_selection_none_offset_by_chunk_index(self):
        # index=3, chunk_size=4 → offset 12 over a full chunk.
        entry = utils.ChunkRead(index=3, num_selected=4, selection=None)
        out = retrieval_mod._absolute_variant_indexes(entry, chunk_size=4)
        nt.assert_array_equal(out, [12, 13, 14, 15])

    def test_selection_none_partial_last_chunk(self):
        # Partial last chunk: num_selected < chunk_size. Length must
        # come from num_selected, not chunk_size.
        entry = utils.ChunkRead(index=2, num_selected=2, selection=None)
        out = retrieval_mod._absolute_variant_indexes(entry, chunk_size=4)
        nt.assert_array_equal(out, [8, 9])

    def test_selection_slice_full_chunk(self):
        entry = utils.ChunkRead(index=1, num_selected=3, selection=slice(0, 3))
        out = retrieval_mod._absolute_variant_indexes(entry, chunk_size=3)
        nt.assert_array_equal(out, [3, 4, 5])
        assert out.dtype == np.int64

    def test_selection_slice_partial(self):
        # slice(1, 3) inside chunk index 4 with chunk_size 4 → [17, 18].
        entry = utils.ChunkRead(index=4, num_selected=2, selection=slice(1, 3))
        out = retrieval_mod._absolute_variant_indexes(entry, chunk_size=4)
        nt.assert_array_equal(out, [17, 18])

    def test_selection_slice_open_ended(self):
        # slice(2, None) is normalised against chunk_size by sel.indices,
        # producing [2, chunk_size).
        entry = utils.ChunkRead(index=0, num_selected=2, selection=slice(2, None))
        out = retrieval_mod._absolute_variant_indexes(entry, chunk_size=4)
        nt.assert_array_equal(out, [2, 3])

    def test_selection_ndarray(self):
        # Non-contiguous local indexes inside chunk index 2 (offset 6).
        sel = np.array([0, 2, 3], dtype=np.int64)
        entry = utils.ChunkRead(index=2, num_selected=3, selection=sel)
        out = retrieval_mod._absolute_variant_indexes(entry, chunk_size=3)
        nt.assert_array_equal(out, [6, 8, 9])
        assert out.dtype == np.int64

    def test_selection_ndarray_promotes_to_int64(self):
        # Helper coerces to int64 even when the input dtype is narrower.
        sel = np.array([0, 2], dtype=np.int32)
        entry = utils.ChunkRead(index=1, num_selected=2, selection=sel)
        out = retrieval_mod._absolute_variant_indexes(entry, chunk_size=4)
        nt.assert_array_equal(out, [4, 6])
        assert out.dtype == np.int64

    def test_selection_ndarray_empty(self):
        sel = np.array([], dtype=np.int64)
        entry = utils.ChunkRead(index=5, num_selected=0, selection=sel)
        out = retrieval_mod._absolute_variant_indexes(entry, chunk_size=4)
        assert out.shape == (0,)
        assert out.dtype == np.int64

    def test_selection_none_empty(self):
        # num_selected=0 with selection=None: empty result, no offset
        # added (np.arange(0) is empty so chunk_offset never broadcasts).
        entry = utils.ChunkRead(index=7, num_selected=0, selection=None)
        out = retrieval_mod._absolute_variant_indexes(entry, chunk_size=4)
        assert out.shape == (0,)
        assert out.dtype == np.int64


class TestVariantIndexPseudoField:
    """``VczReader.variant_chunks(fields=["variant_index"])`` emits the
    global (store-wide) variant index of each surviving variant in
    each chunk, without reading any variants-axis array. Used by
    ``materialise_variant_filter`` to build a chunk plan from a
    filter; available to any caller that needs to round-trip the
    selection back to global indexes."""

    @staticmethod
    def _make(num_variants=10, variants_chunk_size=3):
        return vcz_builder.make_vcz(
            variant_contig=[0] * num_variants,
            variant_position=list(range(100, 100 + num_variants)),
            alleles=[("A", "T")] * num_variants,
            num_samples=1,
            call_genotype=np.zeros((num_variants, 1, 2), dtype=np.int8),
            info_fields={"DP": np.arange(num_variants, dtype=np.int32) * 10},
            variants_chunk_size=variants_chunk_size,
        )

    def test_default_plan_yields_full_axis(self):
        # No selection, no filter: concatenated indexes cover [0, N).
        reader = VczReader(self._make(num_variants=10, variants_chunk_size=3))
        per_chunk = [
            chunk["variant_index"]
            for chunk in reader.variant_chunks(fields=["variant_index"])
        ]
        assert len(per_chunk) == 4  # 3, 3, 3, 1
        assert per_chunk[0].dtype == np.int64
        nt.assert_array_equal(np.concatenate(per_chunk), np.arange(10))

    def test_default_plan_per_chunk_layout(self):
        # Each chunk's array spans [k*chunk_size, (k+1)*chunk_size) up
        # to the partial last chunk.
        reader = VczReader(self._make(num_variants=10, variants_chunk_size=3))
        per_chunk = [
            chunk["variant_index"]
            for chunk in reader.variant_chunks(fields=["variant_index"])
        ]
        nt.assert_array_equal(per_chunk[0], [0, 1, 2])
        nt.assert_array_equal(per_chunk[1], [3, 4, 5])
        nt.assert_array_equal(per_chunk[2], [6, 7, 8])
        nt.assert_array_equal(per_chunk[3], [9])

    def test_set_variants_array_indexes_match(self):
        # Sparse global indexes: each surviving variant's position in
        # the original axis is recovered in the per-chunk arrays.
        reader = VczReader(self._make(num_variants=10, variants_chunk_size=3))
        wanted = np.array([0, 1, 5, 7, 9], dtype=np.int64)
        reader.set_variants(wanted)
        emitted = np.concatenate(
            [
                chunk["variant_index"]
                for chunk in reader.variant_chunks(fields=["variant_index"])
            ]
        )
        nt.assert_array_equal(emitted, wanted)

    def test_slice_selection_is_handled(self):
        # A single contiguous range becomes a slice selection on the
        # ChunkRead; the pseudo-field must produce the same global
        # indexes.
        reader = VczReader(self._make(num_variants=10, variants_chunk_size=3))
        reader.set_variants(np.array([3, 4], dtype=np.int64))
        plan = reader.variant_chunk_plan
        assert isinstance(plan[0].selection, slice)
        emitted = np.concatenate(
            [
                chunk["variant_index"]
                for chunk in reader.variant_chunks(fields=["variant_index"])
            ]
        )
        nt.assert_array_equal(emitted, [3, 4])

    def test_ndarray_selection_is_handled(self):
        # Non-contiguous local indexes within a chunk keep the
        # selection as an ndarray (no collapse to slice). The
        # pseudo-field must use the ndarray branch of the helper.
        reader = VczReader(self._make(num_variants=10, variants_chunk_size=3))
        reader.set_variants(np.array([0, 2, 4, 6, 7], dtype=np.int64))
        plan = reader.variant_chunk_plan
        # Chunk 0 has local selection [0, 2] — non-contiguous → ndarray.
        assert isinstance(plan[0].selection, np.ndarray)
        emitted = np.concatenate(
            [
                chunk["variant_index"]
                for chunk in reader.variant_chunks(fields=["variant_index"])
            ]
        )
        nt.assert_array_equal(emitted, [0, 2, 4, 6, 7])

    def test_variant_filter_yields_surviving_indexes(self):
        # DP = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]; include "DP>=40"
        # keeps indexes 4..9.
        reader = VczReader(self._make(num_variants=10, variants_chunk_size=3))
        reader.set_variant_filter(
            BcftoolsFilter(field_names=reader.field_names, include="DP>=40")
        )
        emitted = np.concatenate(
            [
                chunk["variant_index"]
                for chunk in reader.variant_chunks(fields=["variant_index"])
            ]
        )
        nt.assert_array_equal(emitted, [4, 5, 6, 7, 8, 9])

    def test_filter_excludes_chunk_skips_emit(self):
        # When a whole chunk is filtered out, no entry is yielded for
        # it (variant_chunks already skips empty chunks).
        reader = VczReader(self._make(num_variants=10, variants_chunk_size=3))
        reader.set_variant_filter(
            BcftoolsFilter(field_names=reader.field_names, include="DP>=70")
        )
        per_chunk = [
            chunk["variant_index"]
            for chunk in reader.variant_chunks(fields=["variant_index"])
        ]
        assert len(per_chunk) == 2
        nt.assert_array_equal(per_chunk[0], [7, 8])
        nt.assert_array_equal(per_chunk[1], [9])

    def test_combined_with_real_field_lengths_match(self):
        # Mixing variant_index with a Zarr-backed variants-axis field
        # yields aligned arrays (same length per chunk).
        reader = VczReader(self._make(num_variants=10, variants_chunk_size=3))
        reader.set_variant_filter(
            BcftoolsFilter(field_names=reader.field_names, include="DP>=40")
        )
        for chunk in reader.variant_chunks(
            fields=["variant_index", "variant_position"]
        ):
            indexes = chunk["variant_index"]
            positions = chunk["variant_position"]
            assert indexes.shape == positions.shape
            nt.assert_array_equal(positions, indexes + 100)

    def test_empty_fields_still_short_circuits(self):
        reader = VczReader(self._make(num_variants=3, variants_chunk_size=3))
        with pytest.raises(StopIteration):
            next(reader.variant_chunks(fields=[]))


class TestVariantCountsPerChunk:
    """``VczReader.variant_counts_per_chunk`` derives per-plan-entry
    variant counts from the chunk plan structure alone — no Zarr
    access. Used by ``BedEncoder`` to size byte offsets without an
    upfront scan."""

    @staticmethod
    def _make(num_variants, variants_chunk_size=3):
        return vcz_builder.make_vcz(
            variant_contig=[0] * num_variants,
            variant_position=list(range(100, 100 + num_variants)),
            alleles=[("A", "T")] * num_variants,
            variants_chunk_size=variants_chunk_size,
            proportional_chunks=False,
        )

    def test_default_plan_full_chunks(self):
        # 9 variants with chunk_size 3 → three full chunks of 3.
        reader = VczReader(self._make(num_variants=9))
        nt.assert_array_equal(reader.variant_counts_per_chunk(), np.array([3, 3, 3]))

    def test_default_plan_partial_last_chunk(self):
        # 7 variants with chunk_size 3 → 3, 3, 1.
        reader = VczReader(self._make(num_variants=7))
        nt.assert_array_equal(reader.variant_counts_per_chunk(), np.array([3, 3, 1]))

    def test_set_variants_array(self):
        # Indexes [0, 1, 5, 7] over 9 variants, chunk_size 3:
        #   chunk 0 selection [0, 1] (size 2)
        #   chunk 1 selection [2]    (size 1)
        #   chunk 2 selection [1]    (size 1)
        reader = VczReader(self._make(num_variants=9))
        reader.set_variants(np.array([0, 1, 5, 7], dtype=np.int64))
        nt.assert_array_equal(reader.variant_counts_per_chunk(), np.array([2, 1, 1]))

    def test_set_variants_contiguous_range_collapses_to_slice(self):
        # Indexes [3, 4, 5] = chunk 1 fully → selection collapses to
        # None (full chunk) per normalise_local_selection.
        reader = VczReader(self._make(num_variants=9))
        reader.set_variants(np.array([3, 4, 5], dtype=np.int64))
        nt.assert_array_equal(reader.variant_counts_per_chunk(), np.array([3]))

    def test_set_variants_slice_within_chunk(self):
        # Indexes [3, 4] = chunk 1 partial → selection is slice(0, 2).
        reader = VczReader(self._make(num_variants=9))
        reader.set_variants(np.array([3, 4], dtype=np.int64))
        plan = reader.variant_chunk_plan
        assert isinstance(plan[0].selection, slice)
        nt.assert_array_equal(reader.variant_counts_per_chunk(), np.array([2]))

    def test_empty_plan(self):
        reader = VczReader(self._make(num_variants=9))
        reader.set_variants(np.empty(0, dtype=np.int64))
        nt.assert_array_equal(
            reader.variant_counts_per_chunk(), np.array([], dtype=np.int64)
        )

    def test_after_materialise_variant_filter(self):
        # 10 variants with chunk_size 4 → chunks of 4, 4, 2.
        # POS = 100..109; filter POS<105 → indexes 0..4 survive
        # (chunk 0 full, chunk 1 first variant only).
        reader = VczReader(self._make(num_variants=10, variants_chunk_size=4))
        reader.set_variant_filter(
            BcftoolsFilter(field_names=reader.field_names, include="POS<105")
        )
        reader.materialise_variant_filter()
        nt.assert_array_equal(reader.variant_counts_per_chunk(), np.array([4, 1]))

    def test_num_selected_matches_emitted_chunk_size(self):
        # Cross-check: num_selected stored on each ChunkRead must equal
        # the number of variants the reader actually emits for that
        # chunk. Covers default plan, sparse index selection, and
        # post-filter plan.
        reader = VczReader(self._make(num_variants=10, variants_chunk_size=4))
        reader.set_variants(np.array([0, 1, 5, 8, 9], dtype=np.int64))
        plan = reader.variant_chunk_plan
        emitted = [
            len(chunk["variant_position"])
            for chunk in reader.variant_chunks(fields=["variant_position"])
        ]
        assert emitted == [cr.num_selected for cr in plan]


class TestVariantChunksStart:
    """``VczReader.variant_chunks(start=k)`` begins iteration at the
    k-th entry of the persistent ``variant_chunk_plan``. Used by
    consumers that maintain their own iterator state across calls
    (e.g. ``BedEncoder``)."""

    @staticmethod
    def _make_vcz(num_variants=12, variants_chunk_size=3):
        return vcz_builder.make_vcz(
            variant_contig=[0] * num_variants,
            variant_position=list(range(100, 100 + num_variants)),
            alleles=[("A", "T")] * num_variants,
            variants_chunk_size=variants_chunk_size,
            proportional_chunks=False,
        )

    def test_default_kwarg_omitted_iterates_full_plan(self):
        reader = VczReader(self._make_vcz())
        chunks = list(reader.variant_chunks(fields=["variant_position"]))
        assert len(chunks) == 4
        positions = np.concatenate([c["variant_position"] for c in chunks])
        nt.assert_array_equal(positions, list(range(100, 112)))

    def test_start_zero_equals_omitted(self):
        reader = VczReader(self._make_vcz())
        baseline = list(reader.variant_chunks(fields=["variant_position"]))
        explicit = list(reader.variant_chunks(fields=["variant_position"], start=0))
        assert len(baseline) == len(explicit)
        for a, b in zip(baseline, explicit):
            nt.assert_array_equal(a["variant_position"], b["variant_position"])

    def test_start_k_yields_plan_suffix(self):
        reader = VczReader(self._make_vcz())
        chunks = list(reader.variant_chunks(fields=["variant_position"], start=2))
        assert len(chunks) == 2
        positions = np.concatenate([c["variant_position"] for c in chunks])
        nt.assert_array_equal(positions, list(range(106, 112)))

    def test_start_at_last_chunk(self):
        reader = VczReader(self._make_vcz())
        chunks = list(reader.variant_chunks(fields=["variant_position"], start=3))
        assert len(chunks) == 1
        nt.assert_array_equal(chunks[0]["variant_position"], list(range(109, 112)))

    def test_start_at_plan_length_is_empty(self):
        reader = VczReader(self._make_vcz())
        chunks = list(reader.variant_chunks(fields=["variant_position"], start=4))
        assert chunks == []

    def test_start_past_plan_length_is_empty(self):
        reader = VczReader(self._make_vcz())
        chunks = list(reader.variant_chunks(fields=["variant_position"], start=99))
        assert chunks == []

    def test_negative_start_raises(self):
        reader = VczReader(self._make_vcz())
        with pytest.raises(ValueError, match="start must be >= 0"):
            list(reader.variant_chunks(fields=["variant_position"], start=-1))

    def test_start_does_not_mutate_persistent_plan(self):
        reader = VczReader(self._make_vcz())
        before = list(reader.variant_chunk_plan)
        list(reader.variant_chunks(fields=["variant_position"], start=2))
        after = list(reader.variant_chunk_plan)
        assert before == after
        assert len(after) == 4

    def test_start_after_set_variants_slices_persistent_subset(self):
        reader = VczReader(self._make_vcz())
        reader.set_variants(np.array([1, 4, 7, 10], dtype=np.int64))
        full = list(reader.variant_chunks(fields=["variant_position"]))
        assert len(full) > 1
        sliced = list(reader.variant_chunks(fields=["variant_position"], start=1))
        assert len(sliced) == len(full) - 1
        expected = np.concatenate([c["variant_position"] for c in full[1:]])
        actual = np.concatenate([c["variant_position"] for c in sliced])
        nt.assert_array_equal(actual, expected)

    def test_concurrent_threads_with_different_starts(self):
        reader = VczReader(self._make_vcz())

        def drain(start):
            chunks = list(
                reader.variant_chunks(fields=["variant_position"], start=start)
            )
            return np.concatenate([c["variant_position"] for c in chunks])

        with cf.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(drain, [0, 1, 2, 3]))

        nt.assert_array_equal(results[0], list(range(100, 112)))
        nt.assert_array_equal(results[1], list(range(103, 112)))
        nt.assert_array_equal(results[2], list(range(106, 112)))
        nt.assert_array_equal(results[3], list(range(109, 112)))


class TestFilterSampleChunkPlanDefault:
    """``filter_sample_chunk_plan`` returns the same plan as
    ``sample_chunk_plan`` in the default (non-view-semantics) mode."""

    def test_default_returns_sample_chunk_plan(self, fx_sample_vcz):
        reader = VczReader(fx_sample_vcz.group)
        assert reader.filter_sample_chunk_plan is reader.sample_chunk_plan


class TestVczReaderCachedProperties:
    """Direct exercise of every ``VczReader`` cached property — they're
    thin Zarr-metadata accessors, but that means they aren't covered
    by the iteration paths."""

    def test_contig_ids(self, fx_sample_vcz):
        ids = VczReader(fx_sample_vcz.group).contig_ids
        assert "19" in ids.tolist()

    def test_filter_ids(self, fx_sample_vcz):
        ids = VczReader(fx_sample_vcz.group).filter_ids
        assert "PASS" in ids.tolist()

    def test_contigs_returns_fixed_length_bytes(self, fx_sample_vcz):
        contigs = VczReader(fx_sample_vcz.group).contigs
        assert contigs.dtype.kind == "S"

    def test_num_variants(self, fx_sample_vcz):
        n = VczReader(fx_sample_vcz.group).num_variants
        assert n == int(fx_sample_vcz.group["variant_position"].shape[0])

    def test_num_samples(self, fx_sample_vcz):
        n = VczReader(fx_sample_vcz.group).num_samples
        assert n == int(fx_sample_vcz.group["sample_id"].shape[0])

    def test_source_present(self):
        root = _basic_vcz()
        root.attrs["source"] = "vcztools-test"
        assert VczReader(root).source == "vcztools-test"

    def test_source_absent(self):
        assert VczReader(_basic_vcz()).source is None

    def test_vcf_meta_information_present(self):
        root = _basic_vcz()
        root.attrs["vcf_meta_information"] = ["##contig=<ID=chr1>"]
        assert VczReader(root).vcf_meta_information == ["##contig=<ID=chr1>"]

    def test_vcf_meta_information_absent(self):
        assert VczReader(_basic_vcz()).vcf_meta_information is None

    def test_field_info_cache_starts_empty(self):
        # _field_info_cache is itself a cached_property; first access
        # materialises the empty dict.
        reader = VczReader(_basic_vcz())
        assert reader._field_info_cache == {}

    def test_contig_lengths_absent(self):
        assert VczReader(_basic_vcz()).contig_lengths is None

    def test_contig_lengths_present(self):
        root = _basic_vcz()
        root.create_array("contig_length", shape=(1,), dtype="i4")
        root["contig_length"][:] = [42]
        nt.assert_array_equal(VczReader(root).contig_lengths, [42])

    def test_region_index_absent_raises(self):
        root = _basic_vcz()
        del root["region_index"]
        reader = VczReader(root)
        with pytest.raises(ValueError, match="Could not load 'region_index'"):
            _ = reader.region_index

    def test_region_index_present(self):
        root = _basic_vcz()
        idx = VczReader(root).region_index
        # Shape varies with chunking; just confirm it's a 2-D table.
        assert idx.ndim == 2

    def test_filter_descriptions_absent(self):
        assert VczReader(_basic_vcz()).filter_descriptions is None

    def test_filter_descriptions_present(self):
        root = vcz_builder.make_vcz(
            variant_contig=[0, 0],
            variant_position=[1, 2],
            alleles=[("A", "T")] * 2,
            num_samples=1,
            filters=("PASS", "q10"),
            filter_descriptions=("All filters passed", "Quality below 10"),
        )
        descriptions = VczReader(root).filter_descriptions
        assert "All filters passed" in descriptions.tolist()


class TestGetFieldInfo:
    """``VczReader.get_field_info`` returns a memoised
    :class:`FieldInfo`; second access of the same field returns the
    same instance from the cache."""

    def test_returns_fieldinfo_with_metadata(self, fx_sample_vcz):
        reader = VczReader(fx_sample_vcz.group)
        info = reader.get_field_info("variant_position")
        arr = fx_sample_vcz.group["variant_position"]
        assert info.name == "variant_position"
        assert info.dtype == arr.dtype
        assert info.shape == tuple(arr.shape)
        assert "variants" in info.dims

    def test_second_access_returns_cached_instance(self, fx_sample_vcz):
        reader = VczReader(fx_sample_vcz.group)
        first = reader.get_field_info("variant_position")
        second = reader.get_field_info("variant_position")
        assert first is second


class TestResolveQueryFieldsAutoDiscover:
    """``variant_chunks(fields=None)`` auto-discovers every
    ``variant_*`` and ``call_*`` field in the store."""

    def test_auto_discover_includes_variant_and_call_fields(self):
        # Use a VCZ with a known call_* field so the test pins both
        # branches of the comprehension.
        num_samples = 2
        call_dp = np.zeros((3, num_samples), dtype=np.int32)
        root = vcz_builder.make_vcz(
            variant_contig=[0, 0, 0],
            variant_position=[1, 2, 3],
            alleles=[("A", "T")] * 3,
            num_samples=num_samples,
            call_fields={"DP": call_dp},
        )
        reader = VczReader(root)
        chunk = next(reader.variant_chunks())
        assert "variant_position" in chunk
        assert "call_DP" in chunk


class TestStaticFieldCache:
    """``VczReader._load_static_field`` caches each static field once;
    ``variant_chunks`` seeds chunk_data and filter_data from that cache
    rather than going through the readahead pipeline."""

    def test_load_static_field_caches_array(self, fx_sample_vcz):
        reader = VczReader(fx_sample_vcz.group)
        first = reader._load_static_field("filter_id")
        second = reader._load_static_field("filter_id")
        # Same object on repeat call → cached, not re-read.
        assert first is second
        assert list(reader._static_field_cache) == ["filter_id"]

    def test_variant_chunks_with_static_query_field(self):
        # A static field requested as a query field is supplied from
        # the reader cache and is the same object across chunks.
        root = _make_filter_vcz(num_variants=9, variants_chunk_size=3)
        reader = VczReader(root)
        chunks = list(reader.variant_chunks(fields=["variant_position", "filter_id"]))
        assert len(chunks) == 3
        first_filter_id = chunks[0]["filter_id"]
        nt.assert_array_equal(first_filter_id, ["PASS", "q10"])
        for chunk in chunks[1:]:
            assert chunk["filter_id"] is first_filter_id

    def test_filter_referenced_static_field_not_in_pipeline(self, monkeypatch):
        # When a FILTER expression references filter_id the readahead
        # pipeline must NOT submit a (filter_id,) read — the value
        # comes from the reader's static cache.
        seen_fields: list[str] = []
        original = retrieval_mod._read_block

        def capturing_read_block(arr, block_index):
            seen_fields.append(arr.path.rsplit("/", 1)[-1])
            return original(arr, block_index)

        monkeypatch.setattr(retrieval_mod, "_read_block", capturing_read_block)

        root = _make_filter_vcz(num_variants=9, variants_chunk_size=3)
        reader = make_reader(root, include='FILTER="PASS"')
        list(reader.variant_chunks(fields=["variant_position"]))
        # filter_id is referenced by the FILTER expression but is read
        # from the reader cache, never via _read_block.
        assert "filter_id" not in seen_fields
        assert "variant_position" in seen_fields
        assert "variant_filter" in seen_fields


class TestFmtBytes:
    @pytest.mark.parametrize(
        ("n", "expected"),
        [
            (0, "0 bytes"),
            (1, "1 bytes"),
            (1023, "1023 bytes"),
        ],
    )
    def test_bytes(self, n, expected):
        assert retrieval_mod._fmt_bytes(n) == expected

    @pytest.mark.parametrize(
        ("n", "expected"),
        [
            (1024, "1.0 KiB"),
            (1536, "1.5 KiB"),
            (1024**2 - 1, "1024.0 KiB"),
        ],
    )
    def test_kib(self, n, expected):
        assert retrieval_mod._fmt_bytes(n) == expected

    @pytest.mark.parametrize(
        ("n", "expected"),
        [
            (1024**2, "1.0 MiB"),
            (1024**2 + 512 * 1024, "1.5 MiB"),
            (256 * 1024**2, "256.0 MiB"),
            (1024**3 - 1, "1024.0 MiB"),
        ],
    )
    def test_mib(self, n, expected):
        assert retrieval_mod._fmt_bytes(n) == expected

    @pytest.mark.parametrize(
        ("n", "expected"),
        [
            (1024**3, "1.0 GiB"),
            (1024**3 + 512 * 1024**2, "1.5 GiB"),
            (10 * 1024**3, "10.0 GiB"),
        ],
    )
    def test_gib(self, n, expected):
        assert retrieval_mod._fmt_bytes(n) == expected


class TestOneLineRepr:
    def test_collapses_multiline(self):
        class Store:
            def __repr__(self):
                return "<FakeStore>\nread_only: True\nsnapshot_id: ABC\n"

        out = retrieval_mod._one_line_repr(Store())
        assert "\n" not in out
        assert out == "<FakeStore> read_only: True snapshot_id: ABC"

    def test_collapses_runs_of_whitespace(self):
        class Store:
            def __repr__(self):
                return "<FakeStore>   spaced     out"

        out = retrieval_mod._one_line_repr(Store())
        assert out == "<FakeStore> spaced out"

    def test_none(self):
        assert retrieval_mod._one_line_repr(None) == "None"

    def test_already_single_line_unchanged(self):
        class Store:
            def __repr__(self):
                return "<LocalStore('/path/to/dir')>"

        assert retrieval_mod._one_line_repr(Store()) == "<LocalStore('/path/to/dir')>"


class TestLogging:
    """Lock in the contract that the read-path emits the expected
    INFO / DEBUG log events. See vcztools/retrieval.py for the
    full set of messages."""

    def test_info_iteration_start_and_end(self, fx_sample_vcz, caplog):
        reader = VczReader(fx_sample_vcz.group)
        with caplog.at_level(logging.INFO, logger="vcztools.retrieval"):
            list(reader.variant_chunks(fields=["variant_position"]))
        assert "variant_chunks: starting iteration" in caplog.text
        assert "variant_chunks: iteration done" in caplog.text
        assert "stream_chunk_size=" in caplog.text

    def test_info_summary_includes_chunk_and_variant_counts(
        self, fx_sample_vcz, caplog
    ):
        reader = VczReader(fx_sample_vcz.group)
        with caplog.at_level(logging.INFO, logger="vcztools.retrieval"):
            list(reader.variant_chunks(fields=["variant_position"]))
        done_lines = [r for r in caplog.records if "iteration done" in r.getMessage()]
        assert len(done_lines) == 1
        msg = done_lines[0].getMessage()
        assert "chunks visited" in msg
        assert "yielded" in msg
        assert "variants" in msg
        assert "max readahead depth" in msg

    def test_debug_per_chunk_lines(self, fx_sample_vcz, caplog):
        reader = VczReader(fx_sample_vcz.group)
        with caplog.at_level(logging.DEBUG, logger="vcztools.retrieval"):
            list(reader.variant_chunks(fields=["variant_position"]))
        assert "StreamReader init:" in caplog.text
        assert "read complete in" in caplog.text
        assert "yielded" in caplog.text

    def test_trace_schedule_chunk(self, fx_sample_vcz, caplog):
        # Per-chunk submission lines fire once per chunk and are too noisy
        # for DEBUG; they live at the sub-DEBUG TRACE level.
        reader = VczReader(fx_sample_vcz.group)
        with caplog.at_level(retrieval_mod.TRACE, logger="vcztools.retrieval"):
            list(reader.variant_chunks(fields=["variant_position"]))
        assert "submitted stream chunk 0:" in caplog.text

    def test_debug_static_field_load(self, fx_sample_vcz, caplog):
        # filter_id is a static (no variants axis) field; referencing it
        # via a FILTER expression triggers _load_static_field.
        reader = make_reader(fx_sample_vcz.group, include='FILTER="PASS"')
        with caplog.at_level(logging.DEBUG, logger="vcztools.retrieval"):
            list(reader.variant_chunks(fields=["variant_position"]))
        assert "Loaded static field: filter_id" in caplog.text

    def test_debug_filter_pass_count(self, fx_sample_vcz, caplog):
        reader = make_reader(fx_sample_vcz.group, include='FILTER="PASS"')
        with caplog.at_level(logging.DEBUG, logger="vcztools.retrieval"):
            list(reader.variant_chunks(fields=["variant_position"]))
        assert "filter pass" in caplog.text

    def test_debug_setters(self, fx_sample_vcz, caplog):
        with caplog.at_level(logging.DEBUG, logger="vcztools.retrieval"):
            reader = VczReader(fx_sample_vcz.group)
            reader.set_samples([0, 1])
        assert "VczReader init:" in caplog.text
        assert "set_samples:" in caplog.text

    def test_init_log_is_single_line_with_multiline_store_repr(
        self, fx_sample_vcz, caplog, monkeypatch
    ):
        # Wrap the fixture root with a proxy that exposes a multi-line
        # repr from .store, mimicking IcechunkStore's behaviour.
        class MultiLineStore:
            def __repr__(self):
                return "<FakeStore>\nread_only: True\nsnapshot_id: ABC\nbranch: None"

        real_group = fx_sample_vcz.group

        class Proxy:
            store = MultiLineStore()

            def __getattr__(self, name):
                return getattr(real_group, name)

            def __getitem__(self, key):
                return real_group[key]

        with caplog.at_level(logging.DEBUG, logger="vcztools.retrieval"):
            VczReader(Proxy())
        init_records = [
            r for r in caplog.records if "VczReader init:" in r.getMessage()
        ]
        assert len(init_records) == 1
        msg = init_records[0].getMessage()
        assert "\n" not in msg
        assert "<FakeStore> read_only: True snapshot_id: ABC branch: None" in msg

    def test_no_logging_at_warning_level(self, fx_sample_vcz, caplog):
        # The read path is informational, not a warning source: nothing
        # should fire above INFO during a normal iteration.
        reader = VczReader(fx_sample_vcz.group)
        with caplog.at_level(logging.WARNING, logger="vcztools.retrieval"):
            list(reader.variant_chunks(fields=["variant_position"]))
        retrieval_records = [
            r for r in caplog.records if r.name == "vcztools.retrieval"
        ]
        assert retrieval_records == []

    def test_no_warning_for_undersized_budget(self, fx_sample_vcz, caplog):
        # A readahead budget smaller than a single block silently pins
        # depth at 1 (always submits at least one block ahead of the
        # consumer); the iteration completes without any WARNING.
        reader = VczReader(fx_sample_vcz.group, readahead_bytes=1)
        with caplog.at_level(logging.WARNING, logger="vcztools.retrieval"):
            list(reader.variant_chunks(fields=["variant_position"]))
        assert caplog.text == ""


class TestVariantChunksPrefetch:
    """``variant_chunks`` returns a one-deep prefetch iterator that
    drives the inner generator in a background thread so the
    consumer's per-chunk work overlaps with the producer's
    per-chunk assembly. These cases lock in the wrapper's contract:
    eager validation, empty-iterator short-circuit, exception
    propagation, and clean worker-thread teardown."""

    def _prefetch_threads(self):
        return [t for t in threading.enumerate() if "vcztools-prefetch" in t.name]

    def test_eager_negative_start_validation(self, fx_sample_vcz):
        # Was previously raised on first next() (lazy generator);
        # the wrapper validates eagerly on the call itself.
        reader = VczReader(fx_sample_vcz.group)
        with pytest.raises(ValueError, match="start must be >= 0"):
            reader.variant_chunks(start=-1)

    def test_empty_fields_starts_no_worker(self, fx_sample_vcz):
        # fields=[] short-circuits to iter(()) without spinning up
        # the prefetch worker — exhausting the iterator must not
        # leave a vcztools-prefetch thread alive.
        reader = VczReader(fx_sample_vcz.group)
        before = len(self._prefetch_threads())
        result = list(reader.variant_chunks(fields=[]))
        assert result == []
        assert len(self._prefetch_threads()) == before

    def test_exception_in_inner_gen_surfaces_to_consumer(
        self, fx_sample_vcz, monkeypatch
    ):
        # The wrapper retrieves each item from the worker future via
        # result(); an exception raised by the inner generator must
        # re-raise on the consumer's next() call rather than be
        # swallowed.
        sentinel = RuntimeError("boom from inner generator")

        def faulty_gen(self, *, fields=None, start=0, force_recompute=False):
            yield {"variant_position": np.array([0])}
            raise sentinel

        monkeypatch.setattr(VczReader, "_variant_chunks_gen", faulty_gen)
        reader = VczReader(fx_sample_vcz.group)
        it = reader.variant_chunks(fields=["variant_position"])
        # First chunk arrives normally.
        next(it)
        with pytest.raises(RuntimeError, match="boom from inner generator"):
            next(it)
        it.close()

    def test_close_terminates_worker_thread(self, fx_sample_vcz):
        # After close(), no vcztools-prefetch thread should remain
        # running — confirms the wrapper joins its worker pool.
        reader = VczReader(fx_sample_vcz.group)
        before = self._prefetch_threads()
        it = reader.variant_chunks(fields=["variant_position"])
        # Pull one chunk so the worker is definitely live.
        next(it)
        it.close()
        # Pools shut down asynchronously; allow the worker a brief
        # window to exit before asserting absence.
        deadline = time.time() + 1.0
        while time.time() < deadline:
            after = self._prefetch_threads()
            if len(after) <= len(before):
                break
            time.sleep(0.01)
        assert len(self._prefetch_threads()) <= len(before)


class TestProportionalChunkSizes:
    """End-to-end correctness when variant-only fields use a chunk
    size that is a multiple of the call_* (= min_chunk) chunk size."""

    @staticmethod
    def _build(*, num_variants, min_chunk, allele_chunk, num_samples=2):
        rng = np.random.default_rng(0)
        alleles = [
            (chr(ord("A") + i % 4), chr(ord("T") + i % 4)) for i in range(num_variants)
        ]
        call_genotype = rng.integers(
            0, 2, size=(num_variants, num_samples, 2), dtype=np.int8
        )
        call_dp = rng.integers(0, 100, size=(num_variants, num_samples), dtype=np.int32)
        return vcz_builder.make_vcz(
            variant_contig=[0] * num_variants,
            variant_position=list(range(1, num_variants + 1)),
            alleles=alleles,
            num_samples=num_samples,
            sample_id=[f"s{i}" for i in range(num_samples)],
            variants_chunk_size=min_chunk,
            samples_chunk_size=num_samples,
            call_genotype=call_genotype,
            call_fields={"DP": call_dp},
            field_chunk_overrides={"variant_allele": allele_chunk},
        )

    def test_block_cache_dedups_reads_within_one_field_block(self, monkeypatch):
        # variant_allele chunked at 4 * min_chunk; one field block spans
        # 4 stream chunks. Iterating all 4 must issue the variant_allele
        # block read exactly once (single-future-per-block invariant).
        num_variants = 8
        min_chunk = 2
        store = self._build(
            num_variants=num_variants, min_chunk=min_chunk, allele_chunk=min_chunk * 4
        )
        # variant_allele has only one field block; call_genotype has 4
        # min_chunk-sized blocks.
        assert store["variant_allele"].cdata_shape[0] == 1
        assert store["call_genotype"].cdata_shape[0] == 4

        read_log: list[tuple] = []
        original_read_block = retrieval_mod._read_block

        def counting_read_block(arr, block_index):
            arr_path = getattr(arr, "name", None) or repr(arr)
            read_log.append((arr_path, block_index))
            return original_read_block(arr, block_index)

        monkeypatch.setattr(retrieval_mod, "_read_block", counting_read_block)

        with retrieval_mod.VczReader(store) as r:
            list(r.variant_chunks(fields=["variant_allele", "call_genotype"]))

        variant_allele_reads = [k for k in read_log if "variant_allele" in k[0]]
        # Exactly one read for the single variant_allele field block,
        # despite 4 stream chunks consuming it.
        assert len(variant_allele_reads) == 1

    @pytest.mark.parametrize("multiplier", [1, 2, 3, 5])
    def test_variant_allele_multiplier_matches_baseline(self, multiplier):
        # Build a baseline (multiplier=1) and a scaled-up store with the
        # same logical content; assert variant_chunks() yields the same
        # arrays after concatenation.
        num_variants = 17
        min_chunk = 3
        baseline = self._build(
            num_variants=num_variants, min_chunk=min_chunk, allele_chunk=min_chunk
        )
        scaled = self._build(
            num_variants=num_variants,
            min_chunk=min_chunk,
            allele_chunk=min_chunk * multiplier,
        )
        assert baseline["variant_allele"].chunks[0] == min_chunk
        assert scaled["variant_allele"].chunks[0] == min_chunk * multiplier

        with retrieval_mod.VczReader(baseline) as r:
            base_chunks = list(
                r.variant_chunks(fields=["variant_allele", "call_genotype", "call_DP"])
            )
        with retrieval_mod.VczReader(scaled) as r:
            scaled_chunks = list(
                r.variant_chunks(fields=["variant_allele", "call_genotype", "call_DP"])
            )

        assert len(base_chunks) == len(scaled_chunks)
        for base_c, scaled_c in zip(base_chunks, scaled_chunks):
            assert sorted(base_c.keys()) == sorted(scaled_c.keys())
            for key in base_c:
                nt.assert_array_equal(base_c[key], scaled_c[key])

    @pytest.mark.parametrize(
        ("num_variants", "min_chunk", "multiplier", "region"),
        [
            (17, 5, 2, None),
            (17, 5, 2, "chr1:7-13"),
            (20, 4, 5, None),
            (20, 4, 5, "chr1:9-14"),
            (1, 3, 2, None),
            (6, 3, 2, "chr1:2-2"),
        ],
    )
    def test_region_query_parity_with_partial_last_chunk(
        self, num_variants, min_chunk, multiplier, region
    ):
        # Same logical content stored two ways; same region query.
        # Covers partial last chunks (num_variants not a multiple of
        # min_chunk * multiplier) and the no-region "iterate all" path.
        baseline = self._build(
            num_variants=num_variants, min_chunk=min_chunk, allele_chunk=min_chunk
        )
        scaled = self._build(
            num_variants=num_variants,
            min_chunk=min_chunk,
            allele_chunk=min_chunk * multiplier,
        )

        def _read_with_region(root):
            with retrieval_mod.VczReader(root) as r:
                if region is not None:
                    plan = regions_mod.build_chunk_plan(r, regions=region)
                    r.set_variants(plan)
                return list(
                    r.variant_chunks(fields=["variant_allele", "call_genotype"])
                )

        base_chunks = _read_with_region(baseline)
        scaled_chunks = _read_with_region(scaled)
        assert len(base_chunks) == len(scaled_chunks)
        for base_c, scaled_c in zip(base_chunks, scaled_chunks):
            for key in base_c:
                nt.assert_array_equal(base_c[key], scaled_c[key])


def _make_virtual_field_vcz():
    """5-variant, 3-sample diploid VCZ mirroring the C-kernel basic-test
    inputs so per-variant AC/AN/AF are easy to hand-compute. Variants 3
    and 4 carry a haploid call in sample 2 (encoded with INT_FILL in the
    unused ploidy slot) to exercise mixed ploidy."""
    return vcz_builder.make_vcz(
        variant_contig=[0] * 5,
        variant_position=[100, 200, 300, 400, 500],
        alleles=[
            ("A", "T", ""),
            ("A", "T", "G"),
            ("A", "T", "G"),
            ("A", "T", ""),
            ("A", "T", "G"),
        ],
        num_samples=3,
        ploidy=2,
        call_genotype=[
            [[0, 0], [0, 1], [1, 1]],
            [[0, 0], [0, 2], [1, 2]],
            [[0, 1], [1, 2], [2, 2]],
            [
                [constants.INT_MISSING, constants.INT_MISSING],
                [constants.INT_MISSING, constants.INT_FILL],
                [1, constants.INT_FILL],
            ],
            [
                [0, 2],
                [constants.INT_MISSING, constants.INT_FILL],
                [1, constants.INT_FILL],
            ],
        ],
        variants_chunk_size=5,
    )


_EXPECTED_AC = np.array(
    [
        [3, constants.INT_FILL],
        [1, 2],
        [2, 3],
        [1, constants.INT_FILL],
        [1, 1],
    ],
    dtype=np.int32,
)
_EXPECTED_AN = np.array([6, 6, 6, 1, 3], dtype=np.int32)


def _store_variant_array(root, name, data, dimension_names):
    """Store a variants-axis array carrying dimension metadata. A bare
    ``root[name] = data`` leaves no dims, so the reader cannot tell the
    array has a variants axis and misreads it as a static field."""
    arr = root.create_array(
        name=name,
        shape=data.shape,
        chunks=data.shape,
        dtype=data.dtype,
        dimension_names=dimension_names,
        compressors=None,
        filters=None,
    )
    arr[...] = data
    arr.attrs["_ARRAY_DIMENSIONS"] = list(dimension_names)


# AC / AN recomputed over the sample subset [0, 1] (sample 2 dropped).
_SUBSET01_AC = np.array(
    [
        [1, constants.INT_FILL],
        [0, 1],
        [2, 1],
        [0, constants.INT_FILL],
        [0, 1],
    ],
    dtype=np.int32,
)
_SUBSET01_AN = np.array([4, 4, 4, 0, 2], dtype=np.int32)


class TestVirtualFields:
    """Per-variant AC/AN/AF/NS/N_ALT/N_MISSING/F_MISSING computed on
    the fly by ``VczReader`` via the virtual-fields registry."""

    def test_field_names_is_stored_only(self):
        root = _make_virtual_field_vcz()
        reader = VczReader(root)
        # field_names is the set of Zarr-backed arrays; virtual fields
        # are NOT in it even though they are addressable by name.
        for name in ("variant_AC", "variant_AN", "variant_AF"):
            assert name not in reader.field_names

    def test_virtual_field_names_lists_available_virtuals(self):
        root = _make_virtual_field_vcz()
        reader = VczReader(root)
        # Every registry entry whose deps are satisfied turns up here.
        expected = {
            "variant_AC",
            "variant_AN",
            "variant_AF",
            "variant_NS",
            "variant_N_ALT",
            "variant_N_MISSING",
            "variant_F_MISSING",
        }
        assert expected <= reader.virtual_field_names

    def test_virtual_field_names_omits_call_geno_dependent_without_gt(self):
        # Annotations-only VCZ: only fields whose deps are present (or
        # whose degenerate form is present) appear.
        root = vcz_builder.make_vcz(
            variant_contig=[0] * 3,
            variant_position=[1, 2, 3],
            alleles=[("A", "T")] * 3,
        )
        reader = VczReader(root)
        assert "variant_AC" not in reader.virtual_field_names
        assert "variant_AN" not in reader.virtual_field_names
        assert "variant_NS" not in reader.virtual_field_names
        # N_ALT needs only variant_allele.
        assert "variant_N_ALT" in reader.virtual_field_names
        # N_MISSING / F_MISSING fall back to the all-zero degenerate.
        assert "variant_N_MISSING" in reader.virtual_field_names
        assert "variant_F_MISSING" in reader.virtual_field_names

    def test_get_field_info_ac(self):
        root = _make_virtual_field_vcz()
        reader = VczReader(root)
        info = reader.get_field_info("variant_AC")
        assert info.name == "variant_AC"
        assert info.dtype == np.int32
        assert info.shape == (5, 2)
        assert info.dims == ("variants", "alt_alleles")
        assert info.attrs["description"].startswith("Allele count")

    def test_get_field_info_an(self):
        root = _make_virtual_field_vcz()
        reader = VczReader(root)
        info = reader.get_field_info("variant_AN")
        assert info.dtype == np.int32
        assert info.shape == (5,)
        assert info.dims == ("variants",)

    def test_chunk_values_match_kernel(self):
        root = _make_virtual_field_vcz()
        reader = VczReader(root)
        chunk = next(reader.variant_chunks(fields=["variant_AC", "variant_AN"]))
        nt.assert_array_equal(chunk["variant_AC"], _EXPECTED_AC)
        nt.assert_array_equal(chunk["variant_AN"], _EXPECTED_AN)

    def test_stored_field_wins_by_default(self):
        # Stored variant_AC (zeros) is returned by default, not the
        # registry-computed value.
        root = _make_virtual_field_vcz()
        root["variant_AC"] = np.zeros((5, 2), dtype=np.int32)
        reader = VczReader(root)
        chunk = next(reader.variant_chunks(fields=["variant_AC"]))
        nt.assert_array_equal(chunk["variant_AC"], np.zeros((5, 2)))

    def test_force_recompute_true_recomputes_all_virtuals(self):
        root = _make_virtual_field_vcz()
        root["variant_AC"] = np.zeros((5, 2), dtype=np.int32)
        reader = VczReader(root)
        chunk = next(reader.variant_chunks(fields=["variant_AC"], force_recompute=True))
        nt.assert_array_equal(chunk["variant_AC"], _EXPECTED_AC)

    def test_force_recompute_iterable_targets_named_fields(self):
        root = _make_virtual_field_vcz()
        root["variant_AC"] = np.zeros((5, 2), dtype=np.int32)
        root["variant_AN"] = np.zeros(5, dtype=np.int32)
        reader = VczReader(root)
        chunk = next(
            reader.variant_chunks(
                fields=["variant_AC", "variant_AN"],
                force_recompute=["variant_AC"],
            )
        )
        # AC recomputed; AN still the stored sentinel zeros.
        nt.assert_array_equal(chunk["variant_AC"], _EXPECTED_AC)
        nt.assert_array_equal(chunk["variant_AN"], np.zeros(5, dtype=np.int32))

    def test_virtual_field_without_stored_always_computed(self):
        # No stored variant_AN, no stored variant_AC — the field is
        # always computed even without force_recompute.
        root = _make_virtual_field_vcz()
        reader = VczReader(root)
        chunk = next(reader.variant_chunks(fields=["variant_AN"]))
        nt.assert_array_equal(chunk["variant_AN"], _EXPECTED_AN)

    def test_filter_via_virtual_ac_without_stored(self):
        # No stored variant_AC; ``AC>1`` resolves through the virtual
        # field uniformly.
        root = _make_virtual_field_vcz()
        reader = VczReader(root)
        bf = BcftoolsFilter(
            field_names=reader.field_names | reader.virtual_field_names,
            include="AC>1",
        )
        reader.set_variant_filter(bf)
        chunks = list(reader.variant_chunks(fields=["variant_position"]))
        nt.assert_array_equal(
            np.concatenate([c["variant_position"] for c in chunks]),
            [100, 200, 300],
        )

    def test_default_filter_and_output_agree_under_force_recompute(self):
        # Default mode: the filter and the emitted column resolve a
        # field identically. With a stale stored AC and force_recompute,
        # the ``AC>1`` filter sees the recomputed values (the old code
        # filtered on the stored zeros, disagreeing with the output).
        root = _make_virtual_field_vcz()
        _store_variant_array(
            root,
            "variant_AC",
            np.zeros((5, 2), dtype=np.int32),
            ("variants", "alt_alleles"),
        )
        reader = VczReader(root)
        bf = BcftoolsFilter(
            field_names=reader.field_names | reader.virtual_field_names,
            include="AC>1",
        )
        reader.set_variant_filter(bf)
        chunks = list(
            reader.variant_chunks(
                fields=["variant_position", "variant_AC"],
                force_recompute=["variant_AC"],
            )
        )
        pos = np.concatenate([c["variant_position"] for c in chunks])
        ac = np.concatenate([c["variant_AC"] for c in chunks])
        nt.assert_array_equal(pos, [100, 200, 300])
        nt.assert_array_equal(ac, _EXPECTED_AC[[0, 1, 2]])

    def test_default_subset_recomputes_sample_dependent(self):
        # Default mode: a sample subset makes AC/AN reflect the subset
        # even with a stale stored array and no force_recompute.
        root = _make_virtual_field_vcz()
        _store_variant_array(
            root,
            "variant_AC",
            np.zeros((5, 2), dtype=np.int32),
            ("variants", "alt_alleles"),
        )
        _store_variant_array(
            root, "variant_AN", np.zeros(5, dtype=np.int32), ("variants",)
        )
        reader = VczReader(root)
        reader.set_samples([0, 1])
        chunk = next(reader.variant_chunks(fields=["variant_AC", "variant_AN"]))
        nt.assert_array_equal(chunk["variant_AC"], _SUBSET01_AC)
        nt.assert_array_equal(chunk["variant_AN"], _SUBSET01_AN)

    def test_default_subset_keeps_sample_independent_stored(self):
        # N_ALT depends only on variant_allele, so a subset must not
        # trigger recompute — a stored value survives unchanged.
        root = _make_virtual_field_vcz()
        _store_variant_array(
            root, "variant_N_ALT", np.full(5, 99, dtype=np.int64), ("variants",)
        )
        reader = VczReader(root)
        reader.set_samples([0, 1])
        chunk = next(reader.variant_chunks(fields=["variant_N_ALT"]))
        nt.assert_array_equal(chunk["variant_N_ALT"], np.full(5, 99))

    def test_default_empty_subset_keeps_stored(self):
        # An empty subset has no genotypes to recompute over, so the
        # subset-recompute rule is suppressed and the stored array wins.
        root = _make_virtual_field_vcz()
        _store_variant_array(
            root,
            "variant_AC",
            np.full((5, 2), 7, dtype=np.int32),
            ("variants", "alt_alleles"),
        )
        reader = VczReader(root)
        reader.set_samples([])
        chunk = next(reader.variant_chunks(fields=["variant_AC"]))
        nt.assert_array_equal(chunk["variant_AC"], np.full((5, 2), 7))

    def test_bcftools_semantics_filter_reads_stored_output_recomputes(self):
        # bcftools mode: the filter reads the stored INFO/AC while the
        # output recomputes (the confirmed bcftools quirk). Stored AC is
        # 5 everywhere so ``AC>1`` keeps every row; if the filter used
        # the recomputed AC, row 400 (AC=[1, FILL]) would be dropped.
        root = _make_virtual_field_vcz()
        _store_variant_array(
            root,
            "variant_AC",
            np.full((5, 2), 5, dtype=np.int32),
            ("variants", "alt_alleles"),
        )
        reader = VczReader(root)
        reader.set_bcftools_semantics()
        bf = BcftoolsFilter(
            field_names=reader.field_names | reader.virtual_field_names,
            include="AC>1",
        )
        reader.set_variant_filter(bf)
        chunks = list(
            reader.variant_chunks(
                fields=["variant_position", "variant_AC"],
                force_recompute=["variant_AC"],
            )
        )
        pos = np.concatenate([c["variant_position"] for c in chunks])
        ac = np.concatenate([c["variant_AC"] for c in chunks])
        nt.assert_array_equal(pos, [100, 200, 300, 400, 500])
        nt.assert_array_equal(ac, _EXPECTED_AC)

    def test_variant_af_computed(self):
        root = _make_virtual_field_vcz()
        reader = VczReader(root)
        chunk = next(reader.variant_chunks(fields=["variant_AF"]))
        # Row 0: AC=[3, FILL], AN=6 -> AF=[0.5, FILL]
        fill_bits = constants.FLOAT32_FILL.view(np.int32)
        missing_bits = constants.FLOAT32_MISSING.view(np.int32)
        assert chunk["variant_AF"][0][0] == np.float32(0.5)
        assert chunk["variant_AF"][0].view(np.int32)[1] == fill_bits
        # Row 3 has one ALT (num_alleles=2) -> AF=[1.0, FILL].
        af3 = chunk["variant_AF"][3]
        assert af3[0] == np.float32(1.0)
        assert af3.view(np.int32)[1] == fill_bits
        # Row 4: AC=[1, 1], AN=3 -> AF=[1/3, 1/3].
        nt.assert_allclose(chunk["variant_AF"][4], [1 / 3, 1 / 3])
        # Spot-check the masking helpers stay in sync.
        assert missing_bits != fill_bits

    def test_variant_ns_computed(self):
        root = _make_virtual_field_vcz()
        reader = VczReader(root)
        chunk = next(reader.variant_chunks(fields=["variant_NS"]))
        # Rows 0-2: every sample has at least one called slot -> NS=3
        # Row 3: sample 2 is haploid (1, FILL) so NS=1
        # Row 4: samples 0 and 2 have called slots -> NS=2
        nt.assert_array_equal(chunk["variant_NS"], [3, 3, 3, 1, 2])

    def test_variant_n_alt_no_call_geno_required(self):
        # N_ALT only needs variant_allele, so it works on annotations-only.
        root = vcz_builder.make_vcz(
            variant_contig=[0] * 3,
            variant_position=[1, 2, 3],
            alleles=[("A", "T"), ("A", "T", "G"), ("A", "")],
        )
        reader = VczReader(root)
        chunk = next(reader.variant_chunks(fields=["variant_N_ALT"]))
        nt.assert_array_equal(chunk["variant_N_ALT"], [1, 2, 0])
