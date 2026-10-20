"""Public bgen-limix oracles for the probability-aware exporter."""

import io
import itertools
import math
import sqlite3

import cbgen
import click.testing
import numpy as np
import numpy.testing as nt
import pytest

from tests import vcz_builder
from vcztools import BgenEncoder, VczReader, bgen, bgen_probability, cli, constants


def probability_store(*, variants_chunk_size=1):
    """Mixed phase, triallelic, mixed ploidy, zero ploidy and missing calls."""
    ploidy = np.array([[3, 1, 0, 2], [3, 1, 0, 2]], dtype=np.uint8)
    missing = np.array([[0, 0, 0, 1], [0, 0, 1, 1]], dtype=np.uint8)
    gp = np.full((2, 4, 10), constants.FLOAT64_FILL)
    hp = np.full((2, 4, 9), constants.FLOAT64_FILL)
    gp[1, :, 0] = constants.FLOAT64_MISSING
    hp[0, :, 0] = constants.FLOAT64_MISSING
    gp[0, 0] = np.arange(1, 11) / 55
    gp[0, 1, :3] = [0.15, 0.25, 0.6]
    gp[0, 2, 0] = 1
    gp[0, 3, :6] = constants.FLOAT64_MISSING
    hp[1, 0] = [0.1, 0.2, 0.7, 0.3, 0.2, 0.5, 0.1, 0.4, 0.5]
    hp[1, 1, :3] = [0.2, 0.3, 0.5]
    hp[1, 3, :6] = constants.FLOAT64_MISSING
    return vcz_builder.make_vcz(
        variant_contig=[0, 0],
        variant_position=[11, 22],
        alleles=[("A", "C", "G")] * 2,
        num_samples=4,
        variants_chunk_size=variants_chunk_size,
        samples_chunk_size=2,
        variant_id=["rs1", "rs2"],
        call_fields={"GP": gp, "HP": hp, "PLOIDY": ploidy, "BGEN_MISSING": missing},
        info_fields={"bgen_phased": np.array([False, True]), "bgen_id": ["v1", "v2"]},
    )


def native_read(path):
    with cbgen.BgenFile(str(path)) as source:
        with cbgen.BgenMetafile(str(path) + ".metafile") as index:
            variants = [
                variant
                for partition in range(index.npartitions)
                for variant in index.read_partition(partition).variants
            ]
            return (
                source.read_samples(),
                variants,
                [source.read_genotype(int(v.offset)) for v in variants],
            )


def source_dosage(root, selected):
    result = np.zeros((2, len(selected), 2))
    for row in range(2):
        for j, sample in enumerate(selected):
            if root["call_BGEN_MISSING"][row, sample]:
                result[row, j] = np.nan
                continue
            p = int(root["call_PLOIDY"][row, sample])
            if row:
                result[row, j] = (
                    root["call_HP"][row, sample, : p * 3].reshape(p, 3).sum(axis=0)[1:]
                )
            else:
                genotypes = sorted(
                    itertools.combinations_with_replacement(range(3), p),
                    key=lambda g: g[::-1],
                )
                counts = np.array([[g.count(1), g.count(2)] for g in genotypes])
                result[row, j] = root["call_GP"][row, sample, : len(genotypes)] @ counts
    return result


@pytest.mark.parametrize("bits", [1, 8, 16, 23, 32])
def test_native_probability_roundtrip(tmp_path, bits):
    root = probability_store()
    path = tmp_path / "out.bgen"
    with VczReader(root) as reader:
        bgen.write_bgen(
            reader,
            path,
            bits_per_probability=bits,
            bgi_path=tmp_path / "out.bgi",
            metafile_path=str(path) + ".metafile",
        )
    samples, variants, calls = native_read(path)
    assert len(samples) == 4
    assert [v.id for v in variants] == ["v1", "v2"]
    assert [v.rsid for v in variants] == ["rs1", "rs2"]
    assert [v.position for v in variants] == [11, 22]
    assert all(v.nalleles == 3 for v in variants)
    for row, genotype in enumerate(calls):
        nt.assert_array_equal(genotype.ploidy, root["call_PLOIDY"][row])
        nt.assert_array_equal(genotype.missing, root["call_BGEN_MISSING"][row])
        assert genotype.phased == bool(row)
        field = root["call_HP" if row else "call_GP"]
        for sample, p in enumerate(genotype.ploidy):
            if genotype.missing[sample]:
                continue
            count = int(p) * 3 if row else math.comb(int(p) + 2, int(p))
            expected = field[row, sample, :count]
            actual = genotype.probability[sample, :count]
            assert (
                np.max(np.abs(actual - expected), initial=0)
                <= 1 / ((1 << bits) - 1) + 1e-15
            )
            if count:
                simplexes = actual.reshape(-1, 3) if row else actual[None, :]
                nt.assert_allclose(simplexes.sum(axis=1), 1, atol=1e-15)
    with sqlite3.connect(tmp_path / "out.bgi") as conn:
        assert conn.execute("SELECT number_of_alleles FROM Variant").fetchall() == [
            (3,),
            (3,),
        ]


def test_probabilities_take_precedence_over_gt(tmp_path):
    root = probability_store()
    vcz_builder._create_array(
        root,
        "call_genotype",
        np.zeros((2, 4, 2), dtype=np.int8),
        chunks=(1, 2, 2),
        dimension_names=("variants", "samples", "ploidy"),
    )
    path = tmp_path / "out.bgen"
    with VczReader(root) as reader:
        with pytest.raises(ValueError, match="preserve probability"):
            BgenEncoder(reader)
        bgen.write_bgen(reader, path, metafile_path=str(path) + ".metafile")
    _, _, genotypes = native_read(path)
    assert 0 < genotypes[0].probability[0, 0] < 1


@pytest.mark.parametrize(("k", "p"), [(1, 0), (3, 3), (8, 2), (2, 63)])
def test_general_hard_calls(tmp_path, k, p):
    g = np.full((1, 2, max(1, p)), -2, dtype=np.int16)
    g[0, 0, :p] = k - 1
    g[0, 1, :p] = -1
    root = vcz_builder.make_vcz(
        variant_contig=[0],
        variant_position=[5],
        alleles=[tuple("ACGTNXYZ"[:k])],
        num_samples=2,
        call_genotype=g,
        call_fields={
            "PLOIDY": np.full((1, 2), p, dtype=np.uint8),
            "BGEN_MISSING": np.array([[0, 1]], dtype=np.uint8),
        },
    )
    path = tmp_path / "out.bgen"
    with VczReader(root) as reader:
        bgen.write_bgen(reader, path, metafile_path=str(path) + ".metafile")
    _, _, calls = native_read(path)
    nt.assert_array_equal(calls[0].ploidy, [p, p])
    nt.assert_array_equal(calls[0].missing, [False, True])
    assert calls[0].probability[0, math.comb(p + k - 1, p) - 1] == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("bad_sum", "sum to 1"),
        ("bad_ploidy", "0..63"),
        ("bad_fill", "beyond active"),
        ("mixed_phase", "requires variant_bgen_phased"),
        ("missing_source", "requires call_GP"),
        ("inactive_values", "Inactive call_HP"),
        ("fill_as_missing", "missing NaNs"),
    ],
)
def test_invalid_probabilities_are_errors(tmp_path, mutation, message):
    root = probability_store()
    if mutation == "bad_sum":
        root["call_GP"][0, 0, 0] = 0.9
    elif mutation == "bad_ploidy":
        root["call_PLOIDY"][0, 0] = 64
    elif mutation == "bad_fill":
        root["call_GP"][0, 1, 3] = 0
    elif mutation == "mixed_phase":
        del root["variant_bgen_phased"]
    elif mutation == "missing_source":
        del root["call_GP"]
    elif mutation == "inactive_values":
        root["call_HP"][0, 0, 0] = 0.5
    elif mutation == "fill_as_missing":
        root["call_GP"][0, 3, 0] = constants.FLOAT64_FILL
    with VczReader(root) as reader:
        with pytest.raises(ValueError, match=message):
            bgen.write_bgen(reader, tmp_path / "out.bgen")


def test_native_index_lifecycle(tmp_path):
    root = probability_store()
    path = tmp_path / "out.bgen"
    index = str(path) + ".metafile"
    with VczReader(root) as reader:
        bgen.write_bgen(reader, path, metafile_path=index)
        original = path.read_bytes()
        with pytest.raises(FileExistsError, match="existing native index"):
            bgen.write_bgen(reader, path)
        with pytest.raises(FileExistsError, match="overwrite"):
            bgen.write_metafile(path)
        assert path.read_bytes() == original
        with pytest.raises(ValueError, match="filesystem"):
            bgen.write_bgen(
                reader, io.BytesIO(), metafile_path=tmp_path / "stream.index"
            )
        with pytest.raises(ValueError, match="distinct"):
            bgen.write_bgen(
                reader, tmp_path / "collision", sample_path=tmp_path / "collision"
            )
    assert not list(tmp_path.glob(".vcztools-bgen-*"))


def test_native_index_failure_leaves_no_partial_target(tmp_path, monkeypatch):
    path = tmp_path / "out.bgen"
    with VczReader(probability_store()) as reader:
        bgen.write_bgen(reader, path)

    class FailingBackend:
        nvariants = 2

        def __init__(self, filename):
            self.filename = filename

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def create_metafile(self, filename):
            from pathlib import Path  # noqa: PLC0415

            Path(filename).write_bytes(b"partial index")
            raise ValueError("native failure")

    monkeypatch.setattr(bgen, "_native_backend", lambda: FailingBackend)
    with pytest.raises(ValueError, match="native failure"):
        bgen.write_metafile(path)
    assert not (tmp_path / "out.bgen.metafile").exists()
    assert not list(tmp_path.glob(".vcztools-bgen-*"))


def test_cli_native_options(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cli.ViewBgenOptions,
        "make_reader",
        lambda self, path: VczReader(probability_store()),
    )
    runner = click.testing.CliRunner()
    result = runner.invoke(
        cli.vcztools_main,
        [
            "view-bgen",
            "input",
            "-o",
            str(tmp_path / "out"),
            "--metafile",
            "--bits-per-probability",
            "23",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "out.bgen.bgi").exists()
    assert (tmp_path / "out.sample").exists()
    assert (tmp_path / "out.bgen.metafile").exists()
    result = runner.invoke(cli.vcztools_main, ["view-bgen", "input", "--metafile"])
    assert result.exit_code != 0
    assert "requires -o" in result.output


def test_ds_only_is_rejected(tmp_path):
    root = probability_store()
    del root["call_GP"]
    del root["call_HP"]
    vcz_builder._create_array(
        root,
        "call_DS",
        np.zeros((2, 4)),
        chunks=(1, 2),
        dimension_names=("variants", "samples"),
    )
    with VczReader(root) as reader:
        with pytest.raises(ValueError, match="DS alone is insufficient"):
            bgen.write_bgen(reader, tmp_path / "out.bgen")


def test_hp_cannot_silently_lose_phase(tmp_path):
    with VczReader(probability_store()) as reader:
        with pytest.raises(ValueError, match="without data loss"):
            bgen.write_bgen(reader, tmp_path / "out.bgen", unphased=True)


def test_general_hard_calls_match_fast_path_bytes():
    g = np.array([[[0, 1], [1, -2], [-1, -1]]], dtype=np.int8)
    root = vcz_builder.make_vcz(
        variant_contig=[0],
        variant_position=[7],
        alleles=[("A", "T")],
        num_samples=3,
        call_genotype=g,
    )
    fast = io.BytesIO()
    with VczReader(root) as reader:
        bgen.write_bgen(reader, fast)
    vcz_builder._create_array(
        root,
        "call_PLOIDY",
        np.array([[2, 1, 2]], dtype=np.uint8),
        chunks=(1, 3),
        dimension_names=("variants", "samples"),
    )
    general = io.BytesIO()
    with VczReader(root) as reader:
        bgen.write_bgen(reader, general)
    assert general.getvalue() == fast.getvalue()


def test_float32_normalization_and_32bit_quantization():
    probabilities = np.array([0.1, 0.2, 0.7], dtype=np.float32)
    normalized = probabilities.astype(np.float64)
    normalized /= normalized.sum()
    scale = 2**32 - 1
    raw = bgen_probability._genotype_block(
        {
            "call_GP": probabilities.reshape(1, 1, 3),
            "call_PLOIDY": np.array([[2]], dtype=np.uint8),
        },
        0,
        2,
        1,
        32,
        False,
    )
    stored = np.frombuffer(raw, dtype="<u4", offset=11).astype(np.uint64)
    actual = np.append(stored, scale - int(stored.sum())) / scale
    assert np.max(np.abs(actual - normalized)) <= 1 / scale
    drift = np.max(np.abs(normalized - probabilities))
    assert np.max(np.abs(actual - probabilities)) <= drift + 1 / scale


def test_legacy_backend_conflict_is_explicit(monkeypatch):
    monkeypatch.setattr(bgen.importlib.metadata, "distribution", lambda name: object())
    with pytest.raises(ImportError, match="legacy cbgen distribution conflicts"):
        bgen._native_backend()


@pytest.mark.parametrize("identifier", ["has space", "NA", "."])
def test_sample_validation_precedes_payload_write(tmp_path, identifier):
    root = probability_store()
    root["sample_id"][0] = identifier
    path = tmp_path / "out.bgen"
    sample_path = tmp_path / "out.sample"
    with VczReader(root) as reader:
        with pytest.raises(ValueError, match="whitespace|reserved"):
            bgen.write_bgen(
                reader,
                path,
                sample_path=sample_path,
                bgi_path=str(path) + ".bgi",
                metafile_path=str(path) + ".metafile",
            )
    assert not path.exists()
    assert not sample_path.exists()
    assert not (tmp_path / "out.bgen.bgi").exists()
    assert not (tmp_path / "out.bgen.metafile").exists()


def test_na_embedded_without_sample_sidecar(tmp_path):
    root = probability_store()
    root["sample_id"][0] = "NA"
    path = tmp_path / "out.bgen"
    with VczReader(root) as reader:
        bgen.write_bgen(reader, path, metafile_path=str(path) + ".metafile")
    samples, _, _ = native_read(path)
    assert samples[0] in ("NA", b"NA")


def test_bio2zarr_embedded_na_roundtrip(tmp_path):
    converter = pytest.importorskip("bio2zarr.bgen")
    root = probability_store()
    root["sample_id"][0] = "NA"
    path = tmp_path / "out.bgen"
    with VczReader(root) as reader:
        bgen.write_bgen(reader, path, metafile_path=str(path) + ".metafile")
    target = tmp_path / "roundtrip.vcz"
    converter.convert(path, target, worker_processes=0)
    import zarr  # noqa: PLC0415

    result = zarr.open_group(target, mode="r")
    nt.assert_array_equal(result["sample_id"][:], root["sample_id"][:])


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("subset", [False, True])
def test_bio2zarr_probability_roundtrip(tmp_path, zarr_format, subset):
    converter = pytest.importorskip("bio2zarr.bgen")
    root = probability_store()
    root["sample_id"][0] = "0"
    path = tmp_path / "out.bgen"
    sample_path = tmp_path / "out.sample"
    selected = [1, 3] if subset else list(range(4))
    with VczReader(root) as reader:
        reader.set_sample_indexes(selected)
        bgen.write_bgen(
            reader,
            path,
            sample_path=sample_path,
            embed_header_samples=False,
            metafile_path=str(path) + ".metafile",
        )
    target = tmp_path / "roundtrip.vcz"
    converter.convert(
        path,
        target,
        sample_file=sample_path,
        worker_processes=0,
        zarr_format=zarr_format,
    )
    import zarr  # noqa: PLC0415

    result = zarr.open_group(target, mode="r")
    for field in (
        "sample_id",
        "variant_position",
        "variant_allele",
        "variant_id",
        "variant_bgen_id",
        "variant_bgen_phased",
        "call_PLOIDY",
        "call_BGEN_MISSING",
    ):
        expected = root[field][:]
        if field == "sample_id":
            expected = expected[selected]
        elif field.startswith("call_"):
            expected = expected[:, selected]
        nt.assert_array_equal(result[field][:], expected)
    assert "call_genotype" not in result
    nt.assert_allclose(
        result["call_DS"][:],
        source_dosage(root, selected),
        atol=3 * 10 / 65535,
        rtol=0,
    )
    for field in ("call_GP", "call_HP"):
        expected = root[field][:, selected]
        actual = result[field][:]
        # The importer sizes axes from the selected samples' actual ploidies.
        expected = expected[:, :, : actual.shape[2]]
        nt.assert_array_equal(np.isnan(actual), np.isnan(expected))
        nt.assert_array_equal(
            actual.view(np.uint64)[np.isnan(actual)],
            expected.view(np.uint64)[np.isnan(expected)],
        )
        nt.assert_allclose(actual, expected, atol=1 / 65535, rtol=0)
