"""Deterministic VCZ dataset generator for the benchmark suite.

Builds a VCZ directory via ``bio2zarr.tskit.convert`` and augments it
with synthetic INFO/FORMAT fields computed from ``np.arange``. The
formulae are deterministic so every filter task's expected record count
is exactly computable, emitted into ``benchmark_meta.json`` and asserted
by the runner.
"""

import argparse
import dataclasses
import json
import logging
import pathlib
import shutil

import bio2zarr.tskit
import bio2zarr.zarr_utils as zarr_utils
import msprime
import numpy as np

logger = logging.getLogger(__name__)

# Fixed msprime effective population size. With mutation rate 1e-7 this
# produces a useful variant density on all supported sequence lengths
# without exploding memory.
POPULATION_SIZE = 10_000
MUTATION_RATE = 1e-7
RECOMBINATION_RATE = 1e-8

# Cycled IMPACT values over a period-20 window. HIGH:MOD:LOW = 1:4:15.
IMPACT_CYCLE = np.array(
    ["HIGH"] + ["MOD"] * 4 + ["LOW"] * 15,
    dtype="<U8",
)
assert IMPACT_CYCLE.shape == (20,)


@dataclasses.dataclass(frozen=True)
class DatasetSpec:
    """Parameters for the simulated dataset."""

    num_samples: int
    seq_length: float
    seed: int


def simulate(spec: DatasetSpec):
    """Run msprime with deterministic parameters and return a tskit
    TreeSequence. ``spec.num_samples`` is the *diploid* sample count —
    msprime emits twice as many haploid genomes, which bio2zarr pairs
    back up into ``spec.num_samples`` diploid VCF rows."""
    ts = msprime.sim_ancestry(
        samples=spec.num_samples,
        sequence_length=spec.seq_length,
        recombination_rate=RECOMBINATION_RATE,
        population_size=POPULATION_SIZE,
        random_seed=spec.seed,
    )
    ts = msprime.sim_mutations(ts, rate=MUTATION_RATE, random_seed=spec.seed)
    return ts


def build_vcz(
    ts, vcz_path: pathlib.Path, *, variants_chunk_size=None, samples_chunk_size=None
):
    """Run ``bio2zarr.tskit.convert`` and return the opened group in r+
    mode so augmentation can append fields without reopening."""
    return bio2zarr.tskit.convert(
        ts,
        str(vcz_path),
        mode="r+",
        variants_chunk_size=variants_chunk_size,
        samples_chunk_size=samples_chunk_size,
    )


def _variant_dp_values(num_variants: int) -> np.ndarray:
    return (np.arange(num_variants) % 101).astype(np.int16)


def _variant_qual_values(num_variants: int) -> np.ndarray:
    return (np.arange(num_variants) % 101).astype(np.float32)


def _variant_impact_values(num_variants: int) -> np.ndarray:
    return IMPACT_CYCLE[np.arange(num_variants) % IMPACT_CYCLE.size]


def _call_dp_values(num_variants: int, num_samples: int) -> np.ndarray:
    return (
        (np.arange(num_variants * num_samples) % 101)
        .reshape(num_variants, num_samples)
        .astype(np.int16)
    )


def _call_gq_values(num_variants: int, num_samples: int) -> np.ndarray:
    return (
        (np.arange(num_variants * num_samples) % 100)
        .reshape(num_variants, num_samples)
        .astype(np.int8)
    )


def _delete_if_present(root, name: str) -> None:
    if name in root:
        del root[name]


def _add_variant_array(root, name, data, *, chunks):
    zarr_utils.create_group_array(
        root,
        name,
        data=data,
        shape=data.shape,
        dtype=data.dtype.str,
        chunks=chunks,
        compressor=zarr_utils.DEFAULT_COMPRESSOR_CONFIG,
        dimension_names=["variants"],
    )


def _add_call_array(root, name, data, *, chunks):
    zarr_utils.create_group_array(
        root,
        name,
        data=data,
        shape=data.shape,
        dtype=data.dtype.str,
        chunks=chunks,
        compressor=zarr_utils.DEFAULT_COMPRESSOR_CONFIG,
        dimension_names=["variants", "samples"],
    )


def augment_vcz(root) -> None:
    """Append deterministic synthetic fields to an already-open VCZ."""
    num_variants = int(root["variant_position"].shape[0])
    num_samples = int(root["sample_id"].shape[0])
    variants_chunk = int(root["variant_position"].chunks[0])
    samples_chunk = int(root["sample_id"].chunks[0])

    _add_variant_array(
        root,
        "variant_DP",
        _variant_dp_values(num_variants),
        chunks=(variants_chunk,),
    )

    # Overwrite variant_QUAL regardless of whether tskit.convert produced
    # one — we need the exact arange distribution for filter accounting.
    _delete_if_present(root, "variant_QUAL")
    _add_variant_array(
        root,
        "variant_QUAL",
        _variant_qual_values(num_variants),
        chunks=(variants_chunk,),
    )

    _add_variant_array(
        root,
        "variant_IMPACT",
        _variant_impact_values(num_variants),
        chunks=(variants_chunk,),
    )

    _add_call_array(
        root,
        "call_DP",
        _call_dp_values(num_variants, num_samples),
        chunks=(variants_chunk, samples_chunk),
    )
    _add_call_array(
        root,
        "call_GQ",
        _call_gq_values(num_variants, num_samples),
        chunks=(variants_chunk, samples_chunk),
    )


def _count_by_arange_modulo(num_variants: int, modulo: int, threshold: int) -> int:
    """Exact count of ``np.arange(num_variants) % modulo > threshold``.

    Works for both the int (DP) and float (QUAL) cases because the
    float-cast preserves integer equality for values that fit in 32-bit
    floats.
    """
    full_cycles = num_variants // modulo
    tail = num_variants % modulo
    per_cycle = max(0, modulo - 1 - threshold)
    # First ``tail`` values of the modulo cycle are [0, tail).
    tail_hits = max(0, tail - 1 - threshold)
    return full_cycles * per_cycle + tail_hits


def _count_impact_high(num_variants: int) -> int:
    """Exact count of IMPACT == HIGH in the arange-modulo-20 pattern.

    HIGH sits at position 0 of the period-20 cycle, so there is exactly
    one HIGH per complete period plus one extra iff the tail reaches
    position 0.
    """
    full_cycles = num_variants // IMPACT_CYCLE.size
    tail = num_variants % IMPACT_CYCLE.size
    return full_cycles + (1 if tail > 0 else 0)


def _count_call_dp_gt_80(num_variants: int, num_samples: int) -> int:
    """Variant-level count for ``FMT/DP > 80`` (bcftools view semantics:
    any sample passes → variant kept). Flattened arange modulo 101.
    Compute per-variant whether any of its num_samples consecutive
    values (at offset v*num_samples, modulo 101) exceed 80.
    """
    # Per variant v, sample s: value = (v*num_samples + s) % 101
    # Passes if value > 80. In a window of num_samples consecutive ints
    # (mod 101), the window is the full range iff num_samples >= 101.
    if num_samples >= 101:
        return num_variants
    count = 0
    for v in range(num_variants):
        start = (v * num_samples) % 101
        # window of num_samples consecutive integers (mod 101)
        # values hit: for s in [0, num_samples), (start + s) % 101
        if start + num_samples <= 101:
            # No wrap: values are [start, start + num_samples)
            if start + num_samples - 1 > 80:
                count += 1
        else:
            # Wrap crosses 101: window covers [start, 101); max is 100 > 80.
            count += 1
    return count


def _count_call_gq_gt_50(num_variants: int, num_samples: int) -> int:
    """Variant-level count for ``FMT/GQ > 50`` — same shape as DP but
    modulo 100, threshold 50.
    """
    if num_samples >= 100:
        return num_variants
    count = 0
    for v in range(num_variants):
        start = (v * num_samples) % 100
        if start + num_samples <= 100:
            if start + num_samples - 1 > 50:
                count += 1
        else:
            count += 1
    return count


def _expected_region_count(root, region_spec) -> int:
    """Count variants whose position falls in ``region_spec`` on its
    contig. ``region_spec`` is ``(contig_name, start, end)`` with 1-based
    inclusive positions (bcftools convention)."""
    contig_name, start, end = region_spec
    contig_ids = root["contig_id"][:].tolist()
    contig_idx = contig_ids.index(contig_name)
    variant_contig = root["variant_contig"][:]
    variant_position = root["variant_position"][:]
    mask = (
        (variant_contig == contig_idx)
        & (variant_position >= start)
        & (variant_position <= end)
    )
    return int(mask.sum())


def build_meta(root, spec: DatasetSpec, region_spec) -> dict:
    """Build the benchmark metadata dict. Every filter task's expected
    record count is recomputed here from the deterministic arange
    formulae rather than sampled from the stored arrays."""
    num_variants = int(root["variant_position"].shape[0])
    num_samples = int(root["sample_id"].shape[0])

    region_count = _expected_region_count(root, region_spec)
    dp_gt_80 = _count_by_arange_modulo(num_variants, 101, 80)
    gq_gt_50_variant_level = _count_call_gq_gt_50(num_variants, num_samples)

    meta = {
        "num_samples": num_samples,
        "num_variants": num_variants,
        "seed": spec.seed,
        "seq_length": spec.seq_length,
        "region_spec": {
            "contig": region_spec[0],
            "start": region_spec[1],
            "end": region_spec[2],
        },
        "expected_records": {
            "iter_no_fields": num_variants,
            "iter_info_only": num_variants,
            "iter_info_and_format": num_variants,
            "subset_10_samples": num_variants,
            "subset_half_samples": num_variants,
            "region_10pct": region_count,
            "filter_info_dp_gt_80": dp_gt_80,
            "filter_format_gq_gt_50": gq_gt_50_variant_level,
            "iter_genotypes_only": num_variants,
            "region_and_sample_subset": region_count,
        },
    }
    return meta


def write_meta(vcz_path: pathlib.Path, meta: dict) -> None:
    meta_path = vcz_path.parent / (vcz_path.name + ".benchmark_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    logger.info("wrote %s", meta_path)


def load_meta(vcz_path: pathlib.Path) -> dict:
    meta_path = vcz_path.parent / (vcz_path.name + ".benchmark_meta.json")
    return json.loads(meta_path.read_text())


def default_region_spec(root):
    """Return a ``(contig, start, end)`` tuple spanning ~10% of the
    first contig's variants. Uses the quantile positions so selectivity
    matches what the benchmark names imply.
    """
    contig_ids = root["contig_id"][:].tolist()
    contig = contig_ids[0]
    variant_contig = root["variant_contig"][:]
    variant_position = root["variant_position"][:]
    positions = variant_position[variant_contig == 0]
    positions = np.sort(positions)
    lo_idx = int(len(positions) * 0.45)
    hi_idx = int(len(positions) * 0.55)
    start = int(positions[lo_idx])
    end = int(positions[hi_idx - 1]) if hi_idx > 0 else start
    return (contig, start, end)


def generate(
    num_samples: int,
    seq_length: float,
    seed: int,
    output: pathlib.Path,
    *,
    variants_chunk_size: int | None = None,
    samples_chunk_size: int | None = None,
) -> dict:
    """End-to-end: simulate, convert, augment, write sibling ``.zip``,
    emit ``benchmark_meta.json``. Returns the metadata dict."""
    spec = DatasetSpec(num_samples=num_samples, seq_length=seq_length, seed=seed)
    ts = simulate(spec)
    if ts.num_sites == 0:
        raise RuntimeError(
            f"simulation produced 0 variants for spec={spec}; "
            f"increase seq_length or sample count"
        )
    if output.exists():
        shutil.rmtree(output)
    root = build_vcz(
        ts,
        output,
        variants_chunk_size=variants_chunk_size,
        samples_chunk_size=samples_chunk_size,
    )
    augment_vcz(root)

    zip_path = output.parent / (output.name + ".zip")
    if zip_path.exists():
        zip_path.unlink()
    zarr_utils.zip_zarr(str(output), str(zip_path))

    region_spec = default_region_spec(root)
    meta = build_meta(root, spec, region_spec)
    write_meta(output, meta)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=10_000)
    parser.add_argument("--seq-length", type=float, default=1e6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--variants-chunk-size", type=int, default=None)
    parser.add_argument("--samples-chunk-size", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    meta = generate(
        num_samples=args.num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        output=args.output,
        variants_chunk_size=args.variants_chunk_size,
        samples_chunk_size=args.samples_chunk_size,
    )
    logger.info(
        "generated %d variants x %d samples", meta["num_variants"], meta["num_samples"]
    )


if __name__ == "__main__":
    main()
