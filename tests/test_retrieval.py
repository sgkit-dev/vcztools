import concurrent.futures as cf
import logging

import numpy as np
import numpy.testing as nt
import pandas as pd
import pytest

from tests import vcz_builder
from tests.utils import make_reader, to_vcz_icechunk
from vcztools import regions as regions_mod
from vcztools import retrieval as retrieval_mod
from vcztools import samples as samples_mod
from vcztools import utils
from vcztools.bcftools_filter import BcftoolsFilter
from vcztools.retrieval import CachedVariantChunk, VczReader


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


def _make_pipeline(
    root,
    *,
    readahead_bytes=10**9,
    read_fields=None,
    n_chunks=None,
    executor=None,
):
    """Construct a ``ReadaheadPipeline`` directly against ``root``,
    matching the wiring ``VczReader.variant_chunks`` does (default
    sample-chunk plan over non-null samples; one ``ChunkRead`` per
    variant chunk; no view-mode column remap).

    ``executor`` is the thread pool the pipeline submits reads to. The
    caller is responsible for shutting it down (e.g. via a ``with``
    block); ``None`` means "build a small pool here", which is the
    common case for tests that only need the pipeline to run once and
    don't share the executor with other pipelines.
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
    if n_chunks is None:
        n_chunks = int(root["variant_position"].cdata_shape[0])
    variant_chunk_plan = [utils.ChunkRead(index=i) for i in range(n_chunks)]
    return retrieval_mod.ReadaheadPipeline(
        root,
        variant_chunk_plan,
        sample_chunk_plan,
        None,
        read_fields,
        readahead_bytes=readahead_bytes,
        executor=executor,
    )


class _DepthTrackingPipeline(retrieval_mod.ReadaheadPipeline):
    """Pipeline subclass that records ``len(_in_flight)`` after each
    ``_refill`` call. Used to assert depth-control behaviour under
    different ``readahead_bytes`` values without observing the executor
    directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.depths = []

    def _refill(self):
        super()._refill()
        self.depths.append(len(self._in_flight))


def _vcz_for_template_tests():
    """Small VCZ exposing all four field shapes in the templates:

    - 1-D static (``sample_id``)
    - 1-D variant-axis non-call (``variant_position``)
    - 2-D variant-axis non-call (``variant_allele``)
    - 2-D ``call_*`` (``call_DP``)
    - 3-D ``call_*`` (``call_genotype``)
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
        call_genotype=call_genotype,
        call_fields={"DP": call_dp},
    )


class TestCreateChunkReadList:
    """``create_chunk_read_list`` resolves each requested field to a
    template once: one template per non-``call_*`` field, one per
    ``(field, sample_chunk)`` pair for ``call_*``.
    """

    def _sample_plan(self, root):
        samples_chunk_size = int(root["sample_id"].chunks[0])
        non_null = np.flatnonzero(root["sample_id"][:] != "")
        return samples_mod.build_chunk_plan(
            non_null, samples_chunk_size=samples_chunk_size
        )

    def test_static_field_rejected(self):
        root = _vcz_for_template_tests()
        with pytest.raises(AssertionError, match="non-variants-axis"):
            retrieval_mod.create_chunk_read_list(
                root, self._sample_plan(root), ["sample_id"]
            )

    def test_variant_axis_1d_field(self):
        root = _vcz_for_template_tests()
        templates = retrieval_mod.create_chunk_read_list(
            root, self._sample_plan(root), ["variant_position"]
        )
        assert len(templates) == 1
        t = templates[0]
        assert t.key == ("variant_position",)
        assert t.arr == root["variant_position"]
        # 1-D variants axis → no extra dims after the variant chunk slot.
        assert t.block_index_suffix == ()

    def test_variant_axis_2d_field(self):
        root = _vcz_for_template_tests()
        templates = retrieval_mod.create_chunk_read_list(
            root, self._sample_plan(root), ["variant_allele"]
        )
        assert len(templates) == 1
        t = templates[0]
        assert t.key == ("variant_allele",)
        # 2-D (variants, alleles) → one trailing slice(None).
        assert t.block_index_suffix == (slice(None),)

    def test_call_field_2d_fans_out_per_sample_chunk(self):
        root = _vcz_for_template_tests()
        plan = self._sample_plan(root)
        # 4 samples, samples_chunk_size=2 → 2 sample chunks.
        assert len(plan.chunk_reads) == 2
        templates = retrieval_mod.create_chunk_read_list(root, plan, ["call_DP"])
        assert len(templates) == 2
        assert [t.key for t in templates] == [("call_DP", 0), ("call_DP", 1)]
        call_dp = root["call_DP"]
        for t, cr in zip(templates, plan.chunk_reads):
            assert t.arr == call_dp
            # 2-D (variants, samples) → suffix is (sci,), no trailing slices.
            assert t.block_index_suffix == (cr.index,)

    def test_call_field_3d_keeps_trailing_slice(self):
        root = _vcz_for_template_tests()
        plan = self._sample_plan(root)
        templates = retrieval_mod.create_chunk_read_list(root, plan, ["call_genotype"])
        assert len(templates) == len(plan.chunk_reads)
        for t, cr in zip(templates, plan.chunk_reads):
            assert t.key == ("call_genotype", cr.index)
            # 3-D (variants, samples, ploidy) → suffix carries (sci, slice).
            assert t.block_index_suffix == (cr.index, slice(None))

    def test_multiple_fields_in_input_order(self):
        # call_DP fans out into 2 entries between the two scalar
        # templates; ordering is "fields in input order, then sample
        # chunks within a call_*".
        root = _vcz_for_template_tests()
        plan = self._sample_plan(root)
        templates = retrieval_mod.create_chunk_read_list(
            root, plan, ["variant_position", "call_DP", "variant_allele"]
        )
        assert [t.key for t in templates] == [
            ("variant_position",),
            ("call_DP", 0),
            ("call_DP", 1),
            ("variant_allele",),
        ]


class TestUpdateChunkReadList:
    """``update_chunk_read_list`` substitutes the variant chunk index
    into each template, leaving the templates unchanged so the same
    list is reusable across every chunk in a query.
    """

    def test_variant_non_call_template_prepends_variant_chunk_index(self):
        root = _vcz_for_template_tests()
        plan = samples_mod.SampleChunkPlan(
            chunk_reads=[utils.ChunkRead(index=0)], permutation=None
        )
        templates = retrieval_mod.create_chunk_read_list(
            root, plan, ["variant_position", "variant_allele"]
        )
        reads = retrieval_mod.update_chunk_read_list(templates, 3)
        keys = [r[0] for r in reads]
        block_indexes = [r[2] for r in reads]
        assert keys == [("variant_position",), ("variant_allele",)]
        assert block_indexes == [(3,), (3, slice(None))]

    def test_call_template_keeps_sample_chunk_index_after_variant(self):
        root = _vcz_for_template_tests()
        plan = samples_mod.build_chunk_plan(
            np.array([0, 1, 2, 3], dtype=np.int64), samples_chunk_size=2
        )
        templates = retrieval_mod.create_chunk_read_list(root, plan, ["call_DP"])
        reads = retrieval_mod.update_chunk_read_list(templates, 5)
        assert [r[0] for r in reads] == [("call_DP", 0), ("call_DP", 1)]
        assert [r[2] for r in reads] == [(5, 0), (5, 1)]

    def test_two_calls_yield_independent_lists(self):
        # Reusing a template list across chunks must not have any
        # cross-talk: each call returns a fresh list of fresh tuples
        # against the supplied variant chunk index.
        root = _vcz_for_template_tests()
        plan = samples_mod.SampleChunkPlan(
            chunk_reads=[utils.ChunkRead(index=0)], permutation=None
        )
        templates = retrieval_mod.create_chunk_read_list(
            root, plan, ["variant_position"]
        )
        reads_a = retrieval_mod.update_chunk_read_list(templates, 0)
        reads_b = retrieval_mod.update_chunk_read_list(templates, 4)
        assert reads_a is not reads_b
        assert reads_a[0][2] == (0,)
        assert reads_b[0][2] == (4,)


class TestReadaheadPipeline:
    """Direct unit tests for ``retrieval.ReadaheadPipeline``.

    The end-to-end suites cover correctness; this class targets the
    pipeline's own state machine — bootstrap, budget-driven scheduling
    depth, executor cleanup, and behaviour at the edges (empty plan,
    empty read columns).
    """

    @staticmethod
    def _vcz(num_variants=12, variants_chunk_size=3, num_samples=2):
        return vcz_builder.make_vcz(
            variant_contig=[0] * num_variants,
            variant_position=list(range(100, 100 + num_variants)),
            alleles=[("A", "T")] * num_variants,
            num_samples=num_samples,
            variants_chunk_size=variants_chunk_size,
        )

    def test_yields_one_chunk_per_plan_entry_in_order(self):
        root = self._vcz()
        pipeline = _make_pipeline(root)
        indexes = [chunk.variant_chunk.index for chunk in pipeline]
        assert indexes == [0, 1, 2, 3]

    def test_empty_plan_yields_nothing(self):
        root = self._vcz()
        pipeline = _make_pipeline(root, n_chunks=0)
        assert list(pipeline) == []
        # No pending futures left after a clean drain.
        assert pipeline._in_flight == []

    def test_single_chunk_plan(self):
        root = self._vcz(num_variants=3, variants_chunk_size=3)
        pipeline = _make_pipeline(root)
        chunks = list(pipeline)
        assert len(chunks) == 1
        assert chunks[0].variant_chunk.index == 0

    def test_bootstrap_runs_first_chunk_solo(self):
        # Until the first chunk's prefetch lands the pipeline can't
        # measure per-chunk bytes, so _refill must schedule exactly one
        # chunk on the bootstrap path.
        root = self._vcz()
        pipeline = _make_pipeline(root, readahead_bytes=10**9)
        gen = iter(pipeline)
        # Generator hasn't run yet — no scheduling, no measurement.
        assert pipeline._per_chunk_bytes is None
        chunk = next(gen)
        # After bootstrap the measurement is recorded and matches the
        # prefetched blocks' content.
        assert isinstance(pipeline._per_chunk_bytes, int)
        assert pipeline._per_chunk_bytes > 0
        expected = sum(utils.array_memory_bytes(v) for v in chunk._blocks.values())
        assert pipeline._per_chunk_bytes == expected
        gen.close()

    def test_readahead_bytes_zero_keeps_depth_one(self):
        # Budget=0 → after every refill, exactly one chunk is queued
        # ahead of the consumer (and zero on the final, plan-exhausted
        # refill).
        root = self._vcz(num_variants=12, variants_chunk_size=3)
        with cf.ThreadPoolExecutor(max_workers=2) as executor:
            pipeline = _DepthTrackingPipeline(
                root,
                [utils.ChunkRead(index=i) for i in range(4)],
                samples_mod.build_chunk_plan(
                    np.arange(2, dtype=np.int64), samples_chunk_size=2
                ),
                None,
                ["variant_position"],
                readahead_bytes=0,
                executor=executor,
            )
            list(pipeline)
        # 4 chunks → 5 refills (one per consume + the post-final empty refill).
        assert pipeline.depths == [1, 1, 1, 1, 0]

    def test_large_readahead_schedules_all_remaining_after_bootstrap(self):
        # Budget of 10**9 dwarfs the per-chunk cost, so the second
        # refill fills with every remaining chunk in one go.
        root = self._vcz(num_variants=12, variants_chunk_size=3)
        with cf.ThreadPoolExecutor(max_workers=2) as executor:
            pipeline = _DepthTrackingPipeline(
                root,
                [utils.ChunkRead(index=i) for i in range(4)],
                samples_mod.build_chunk_plan(
                    np.arange(2, dtype=np.int64), samples_chunk_size=2
                ),
                None,
                ["variant_position"],
                readahead_bytes=10**9,
                executor=executor,
            )
            list(pipeline)
        # Bootstrap depth=1, then post-yield-1 schedules the remaining
        # 3, then drains.
        assert pipeline.depths == [1, 3, 2, 1, 0]

    def test_empty_read_fields_does_not_infinite_loop(self):
        # With no fields to prefetch the bootstrap measurement is 0
        # bytes; without the ``max(1, per_chunk_bytes)`` guard the
        # budget loop would never exit. List materialises the full
        # sequence.
        root = self._vcz(num_variants=6, variants_chunk_size=3)
        pipeline = _make_pipeline(root, read_fields=[], readahead_bytes=10**9)
        chunks = list(pipeline)
        assert len(chunks) == 2
        for chunk in chunks:
            assert chunk._blocks == {}

    def test_chunks_have_prefetched_blocks(self):
        # Every (key, future) submitted lands in chunk._blocks before
        # the consumer receives the chunk.
        root = self._vcz(num_variants=6, variants_chunk_size=3, num_samples=2)
        pipeline = _make_pipeline(
            root,
            read_fields=["variant_position", "variant_contig"],
            readahead_bytes=0,
        )
        for chunk in pipeline:
            assert ("variant_position",) in chunk._blocks
            assert ("variant_contig",) in chunk._blocks

    def test_executor_outlives_full_iteration(self):
        # The pipeline does not own the executor; full drain leaves
        # the pool alive and ready to serve another pipeline.
        root = self._vcz()
        with cf.ThreadPoolExecutor(max_workers=2) as executor:
            pipeline = _make_pipeline(root, executor=executor)
            list(pipeline)
            assert executor._shutdown is False
            # Pool is still usable for a second pipeline.
            second = _make_pipeline(root, executor=executor)
            assert len(list(second)) > 0

    def test_pending_futures_cancelled_on_early_break(self):
        # Abandoning iteration cancels still-pending futures (those
        # that hadn't started); the executor itself stays alive.
        root = self._vcz(num_variants=24, variants_chunk_size=3)
        with cf.ThreadPoolExecutor(max_workers=2) as executor:
            pipeline = _make_pipeline(root, executor=executor, readahead_bytes=10**9)
            gen = iter(pipeline)
            next(gen)
            in_flight_snapshot = [
                fut for _, futures in pipeline._in_flight for _, fut in futures
            ]
            gen.close()
            assert executor._shutdown is False
            for fut in in_flight_snapshot:
                assert fut.cancelled() or fut.done()


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
            regions_mod.build_chunk_plan(self._vcz(), regions="^chr1:1-3")

    def test_targets_rejects_caret_prefix(self):
        with pytest.raises(ValueError, match="targets_complement=True"):
            regions_mod.build_chunk_plan(self._vcz(), targets="^chr1:1-3")

    def test_regions_rejects_comma(self):
        with pytest.raises(ValueError, match=r"regions string .* contains ','"):
            regions_mod.build_chunk_plan(self._vcz(), regions="chr1:1-3,chr1:5-7")

    def test_targets_rejects_comma(self):
        with pytest.raises(ValueError, match=r"targets string .* contains ','"):
            regions_mod.build_chunk_plan(self._vcz(), targets="chr1:1-3,chr1:5-7")

    def test_regions_invalid_type(self):
        with pytest.raises(TypeError, match="regions must be"):
            regions_mod.build_chunk_plan(self._vcz(), regions=42)

    def test_targets_invalid_type(self):
        with pytest.raises(TypeError, match="targets must be"):
            regions_mod.build_chunk_plan(self._vcz(), targets=42)

    def test_regions_dataframe_missing_columns(self):
        df = pd.DataFrame({"contig": ["chr1"], "start": pd.array([1], dtype="Int64")})
        with pytest.raises(ValueError, match="missing required columns.*end"):
            regions_mod.build_chunk_plan(self._vcz(), regions=df)

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
    variant_chunk = utils.ChunkRead(
        index=variant_chunk_idx, selection=variant_selection
    )
    templates = retrieval_mod.create_chunk_read_list(root, sample_chunk_plan, fields)
    reads = retrieval_mod.update_chunk_read_list(templates, variant_chunk.index)
    blocks = {key: retrieval_mod._read_block(arr, idx) for key, arr, idx in reads}
    return CachedVariantChunk(
        root,
        variant_chunk,
        sample_chunk_plan=sample_chunk_plan,
        output_columns=output_columns,
        blocks=blocks,
    )


class TestCachedVariantChunkCache:
    """CachedVariantChunk consumes prefetched blocks and caches
    assembled views so a field reused across filter_view / output_view
    is materialized once."""

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


class TestCachedVariantChunkAxes:
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
        # Default filter axis (no set_filter_samples call) is the
        # sample subset — bcftools-query-style post-subset evaluation.
        # Position 1234567 (where NA00001 but not NA00002/NA00003
        # matched FMT/DP>3) is DROPPED.
        reader = VczReader(fx_sample_vcz.group)
        reader.set_variants(
            regions_mod.build_chunk_plan(fx_sample_vcz.group, regions="20:1230236-")
        )
        reader.set_samples([1, 2])
        reader.set_variant_filter(
            BcftoolsFilter(field_names=reader.field_names, include="FMT/DP>3"),
        )
        chunks = list(reader.variant_chunks(fields=["variant_position", "call_DP"]))
        chunk = chunks[0]
        assert chunk["call_DP"].shape[1] == 2
        nt.assert_array_equal(chunk["variant_position"], [1230237])

    def test_explicit_filter_samples_no_subset_is_noop(self, fx_sample_vcz):
        # With no sample subset, passing non_null_sample_indices (view
        # semantics) must return identical results to the default.
        root = fx_sample_vcz.group

        def build(view_semantics):
            reader = VczReader(root)
            reader.set_variants(
                regions_mod.build_chunk_plan(root, regions="20:1230236-")
            )
            if view_semantics:
                reader.set_filter_samples(reader.non_null_sample_indices)
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
                reader.set_filter_samples(reader.non_null_sample_indices)
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
    """``CachedVariantChunk._empty_call_array`` produces a zero-column
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
            call_fields={"DP": call_dp},
        )

    def _empty_plan_chunk(self, root, *, variant_chunk_idx, selection):
        empty_plan = samples_mod.SampleChunkPlan(chunk_reads=[], permutation=None)
        return CachedVariantChunk(
            root,
            utils.ChunkRead(index=variant_chunk_idx, selection=selection),
            sample_chunk_plan=empty_plan,
            output_columns=None,
            blocks={},
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
    """``set_samples`` and ``set_filter_samples`` are one-shot — a
    second call raises ``RuntimeError``. ``set_variants`` and
    ``set_variant_filter`` are re-callable (covered by
    :class:`TestSettersReplace`)."""

    def test_set_samples_twice_raises(self, fx_sample_vcz):
        reader = VczReader(fx_sample_vcz.group)
        reader.set_samples([0])
        with pytest.raises(RuntimeError, match="samples already configured"):
            reader.set_samples([1])

    def test_set_filter_samples_twice_raises(self, fx_sample_vcz):
        reader = VczReader(fx_sample_vcz.group)
        reader.set_filter_samples([0, 1])
        with pytest.raises(RuntimeError, match="filter_samples already configured"):
            reader.set_filter_samples([0, 1])


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


class TestNormalizeSampleIndexesSortedUnique:
    """``_normalize_sample_indexes`` rejects non-sorted-unique input
    when ``sorted_unique=True`` — the path used by
    ``set_filter_samples``."""

    def test_unsorted_filter_samples_raises(self, fx_sample_vcz):
        reader = VczReader(fx_sample_vcz.group)
        with pytest.raises(ValueError, match="must be sorted ascending and unique"):
            reader.set_filter_samples([2, 1])

    def test_duplicate_filter_samples_raises(self, fx_sample_vcz):
        reader = VczReader(fx_sample_vcz.group)
        with pytest.raises(ValueError, match="must be sorted ascending and unique"):
            reader.set_filter_samples([1, 1])


class TestFilterSampleChunkPlanDefault:
    """``filter_sample_chunk_plan`` returns the same plan as
    ``sample_chunk_plan`` when ``set_filter_samples`` has not been
    called."""

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
        assert "Per-chunk read size:" in caplog.text

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

    def test_debug_per_chunk_lines(self, fx_sample_vcz, caplog):
        reader = VczReader(fx_sample_vcz.group)
        with caplog.at_level(logging.DEBUG, logger="vcztools.retrieval"):
            list(reader.variant_chunks(fields=["variant_position"]))
        assert "ReadaheadPipeline init:" in caplog.text
        assert "schedule chunk 0:" in caplog.text
        assert "read complete in" in caplog.text
        assert "yielded" in caplog.text

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
