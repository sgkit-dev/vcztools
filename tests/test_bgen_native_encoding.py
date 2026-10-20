"""Native encoding integration, including the released-reader capability gate."""

import io

import cbgen
import numpy as np
import pytest

from tests import bgen_probability_reference, vcz_builder
from tests.test_bgen_probability import probability_store
from vcztools import VczReader, bgen, bgen_probability, constants, utils


@pytest.mark.parametrize("version", ["4.14.0", "5.0.0"])
def test_unsupported_backend_version_rejected_before_output(
    tmp_path, monkeypatch, version
):
    bgen._native_backend_module.cache_clear()
    monkeypatch.setattr(bgen.importlib.metadata, "version", lambda name: version)
    with VczReader(probability_store()) as reader:
        with pytest.raises(ImportError, match="bgen-limix>=4.15,<5"):
            bgen.write_bgen(
                reader,
                tmp_path / "out.bgen",
                sample_path=tmp_path / "out.sample",
                bgi_path=tmp_path / "out.bgi",
                metafile_path=tmp_path / "out.bgen.metafile",
            )
    assert not list(tmp_path.iterdir())


def test_native_installation_checked_once_but_capabilities_checked_each_time(
    monkeypatch,
):
    bgen._native_backend_module.cache_clear()
    distribution = bgen.importlib.metadata.distribution
    calls = []

    def lookup(name):
        calls.append(name)
        return distribution(name)

    monkeypatch.setattr(bgen.importlib.metadata, "distribution", lookup)
    assert bgen._native_backend(encoding=True) is cbgen
    initial = calls.copy()
    assert initial
    assert bgen._native_backend(encoding=True) is cbgen
    assert calls == initial
    monkeypatch.delattr(cbgen, "encode_layout2_fixed")
    with pytest.raises(ImportError, match="bgen-limix>=4.15,<5"):
        bgen._native_backend(encoding=True), constants, utils


@pytest.mark.parametrize("hardcalls", [False, True])
def test_encoding_capability_required_before_any_output(
    tmp_path, monkeypatch, hardcalls
):
    for name in ("encode_layout2", "encode_layout2_hardcalls"):
        monkeypatch.delattr(cbgen, name, raising=False)
    root = probability_store()
    if hardcalls:
        del root["call_GP"]
        del root["call_HP"]
        vcz_builder._create_array(
            root,
            "call_genotype",
            np.zeros((2, 4, 3), dtype=np.int8),
            chunks=(1, 2, 3),
            dimension_names=("variants", "samples", "ploidy"),
        )
    payload = tmp_path / "out.bgen"
    with VczReader(root) as reader:
        with pytest.raises(ImportError, match="bgen-limix>=4.15,<5"):
            bgen.write_bgen(
                reader,
                payload,
                sample_path=tmp_path / "out.sample",
                bgi_path=tmp_path / "out.bgi",
                metafile_path=str(payload) + ".metafile",
            )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("fixed", [False, True])
def test_backend_requires_specialized_encoding_before_output(
    tmp_path, monkeypatch, fixed
):
    for name in ("encode_layout2_hardcalls_legacy", "encode_layout2_fixed"):
        monkeypatch.delattr(cbgen, name, raising=False)
    root = vcz_builder.make_vcz(
        variant_contig=[0],
        variant_position=[1],
        alleles=[("A", "C")],
        num_samples=2,
        call_genotype=np.array([[[0, 1], [-1, -1]]], dtype=np.int8),
    )
    output = tmp_path / "out.bgen"
    with VczReader(root) as reader:
        with pytest.raises(ImportError, match="bgen-limix>=4.15,<5"):
            bgen.write_bgen(reader, output, fixed_variant_size=fixed)
    assert not output.exists()


def test_general_slices_are_bounded_by_output_not_input_padding():
    chunk = {
        "call_GP": np.empty((32, 1024, 0)),
        "call_PLOIDY": np.full((32, 1024), 4, dtype=np.uint8),
    }
    ranges = list(bgen_probability._slice_ranges(chunk, [8] * 32, 1024, 16, False, 1))
    assert ranges == [(0, 12), (12, 24), (24, 32)]


def test_oversized_single_variant_rejected_before_native_allocation():
    chunk = {
        "call_GP": np.empty((1, 1, 0)),
        "call_PLOIDY": np.array([[63]], dtype=np.uint8),
    }
    with pytest.raises(ValueError, match="256 MiB native batch limit"):
        list(bgen_probability._slice_ranges(chunk, [65535], 1, 32, False, 1))


@pytest.mark.parametrize("bits", [1, 8, 16, 23, 32])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_native_batches_match_python_oracle(bits, dtype):
    root = probability_store()
    with VczReader(root) as reader:
        chunks = list(reader.variant_chunks())
    chunk = {
        field: np.concatenate([part[field] for part in chunks]) for field in chunks[0]
    }
    for field in ("call_GP", "call_HP"):
        if dtype == np.float32:
            # Casting NaNs does not preserve VCZ sentinel payloads.
            source = chunk[field]
            target = np.empty(source.shape, dtype=dtype)
            valid = np.isfinite(source)
            target[valid] = source[valid]
            target[utils.is_missing(source)] = constants.FLOAT32_MISSING
            target[utils.is_fill(source)] = constants.FLOAT32_FILL
            chunk[field] = target
    raw, offsets = bgen_probability._encode_genotype_blocks(
        chunk, [3, 3], 4, bits, False, bgen._native_backend(encoding=True)
    )
    expected = [
        bgen_probability_reference.genotype_block(chunk, row, 3, 4, bits)
        for row in range(2)
    ]
    assert raw == b"".join(expected)
    np.testing.assert_array_equal(offsets, [0, len(expected[0]), len(raw)])


@pytest.mark.parametrize("bits", [1, 8, 16, 23, 32])
def test_general_hardcall_batch_matches_oracle(bits):
    chunk = {
        "call_genotype": np.array(
            [[[0, 1, 2], [1, -2, -2], [-2, -2, -2], [-1, -1, -2]]],
            dtype=np.int16,
        ),
        "call_BGEN_MISSING": np.array([[0, 0, 0, 1]], dtype=np.uint8),
    }
    for phased in (False, True):
        chunk["variant_bgen_phased"] = np.array([phased])
        raw = bgen_probability._genotype_block(chunk, 0, 3, 4, bits, False)
        assert raw == bgen_probability_reference.genotype_block(chunk, 0, 3, 4, bits)


def test_general_hardcalls_support_alleles_above_int8(tmp_path):
    alleles = tuple(f"A{i}" for i in range(129))
    root = vcz_builder.make_vcz(
        variant_contig=[0],
        variant_position=[7],
        alleles=[alleles],
        num_samples=1,
    )
    vcz_builder._create_array(
        root,
        "call_genotype",
        np.array([[[128]]], dtype=np.int16),
        chunks=(1, 1, 1),
        dimension_names=("variants", "samples", "ploidy"),
    )
    path = tmp_path / "wide.bgen"
    with VczReader(root) as reader:
        bgen.write_bgen(reader, path, metafile_path=str(path) + ".metafile")
    with (
        cbgen.BgenFile(str(path)) as source,
        cbgen.BgenMetafile(str(path) + ".metafile") as index,
    ):
        variant = index.read_partition(0).variants[0]
        genotype = source.read_genotype(int(variant.offset))
        assert genotype.probability[0, 128] == 1


def test_threaded_general_export_is_byte_identical():
    root = probability_store(variants_chunk_size=2)
    for selection in (None, [3, 1, 0], []):
        outputs = []
        for threads in (1, 2, 4):
            with VczReader(root) as reader:
                if selection is not None:
                    reader.set_sample_indexes(selection)
                output = io.BytesIO()
                bgen.write_bgen(reader, output, encode_threads=threads)
                outputs.append(output.getvalue())
        assert outputs[0] == outputs[1] == outputs[2]


@pytest.mark.parametrize("fixed", [False, True])
def test_legacy_export_uses_upstream_specialized_fastpath(monkeypatch, fixed):
    def encode(**kwargs):
        raise AssertionError(
            "Legacy hard calls must retain the specialized C fast path"
        )

    monkeypatch.setattr(cbgen, "encode_layout2_hardcalls", encode)
    name = "encode_layout2_fixed" if fixed else "encode_layout2_hardcalls_legacy"
    original = getattr(cbgen, name)
    calls = []

    def specialized(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(cbgen, name, specialized)
    root = vcz_builder.make_vcz(
        variant_contig=[0],
        variant_position=[1],
        alleles=[("A", "C")],
        num_samples=2,
        call_genotype=np.array([[[0, 1], [1, 1]]], dtype=np.int8),
    )
    with VczReader(root) as reader:
        bgen.write_bgen(reader, io.BytesIO(), fixed_variant_size=fixed)
    assert calls == [True]
