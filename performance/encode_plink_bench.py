"""CPU-only encoding microbenchmarks for the formats vcztools writes.

Allocates a synthetic ``call_genotype``-shaped int8 buffer plus minimal
per-variant metadata in memory, and times repeated calls to each
encoder kernel in isolation — no Zarr, no I/O, no readahead pipeline.
Gives the upper bound the corresponding end-to-end matrix benchmarks
(``output_plink`` / ``output_bed_stream`` for plink, ``view_vcf`` for
VCF, ``write_bgen`` / BGEN streaming for BGEN) can asymptotically
approach.

Formats:

* ``plink`` — calls ``_vcztools.encode_plink`` on the whole chunk.
* ``vcf`` — constructs ``_vcztools.VcfEncoder`` and runs the per-variant
  ``encode(j, buflen)`` loop, mirroring ``VcfWriter.write_chunk``. GT-only:
  no INFO/FORMAT fields, so the kernel cost tracks the same genotype
  payload as the other three benchmarks.
* ``bgen`` — variable-size, zlib-compressed BGEN; per-variant
  ``bgen._encode_variant_block`` loop. Compression level controlled by
  ``--compression-level`` (default 6, the zlib default).
* ``bgen-fixed`` — fixed-size BGEN; per-variant
  ``bgen._encode_variant_block_fixed_size`` loop (zlib level 0 stored).

Run from the repo root::

    uv run python performance/encode_plink_bench.py
    uv run python performance/encode_plink_bench.py --format bgen-fixed
    uv run python performance/encode_plink_bench.py --format all --sweep

Defaults to one variant-chunk's worth of work on the standard bench
shape (1000 variants x 100000 samples). Override with ``--variants``
and ``--samples``; ``--repeats`` controls the number of timed calls.

``--threads`` and ``--block-bytes`` exercise variant-axis slicing for
``plink``, ``bgen``, and ``bgen-fixed`` (same strategy as
:class:`vcztools.plink.BedEncoder` / :class:`vcztools.bgen.BgenEncoder`).
VCF has no parallel slicing path — the ``VcfEncoder`` is stateful
per-chunk — so ``--threads > 1 --format vcf`` falls back to sync with
a warning. ``--sweep`` runs the canonical ``(threads, block_bytes)``
matrix per active format and prints one markdown table each (VCF's
parallel cells are filled with ``--`` to make the limitation explicit).
"""

import concurrent.futures as cf
import dataclasses
import statistics
import time

import click
import numpy as np

from vcztools import _vcztools, bgen, constants

# Mix of values that exercises every branch in encode_diploid_fixed:
# 0/1 (REF/ALT calls), 2 (out-of-range diploid), -1 (missing
# sentinel), -2 (haploid sentinel paired with 0/1/2 in the b slot).
_GENOTYPE_VALUES = np.array([0, 1, 0, 0, 1, 1, 2, -1, -2, 0], dtype=np.int8)

_SWEEP_THREADS = (1, 2, 4, 8)
_SWEEP_BLOCK_BYTES = (1 << 20, 4 << 20, 10 << 20, 40 << 20)

_FORMATS = ("plink", "vcf", "bgen", "bgen-fixed")


def _make_genotypes(num_variants: int, num_samples: int, seed: int) -> np.ndarray:
    """Return a (V, S, 2) int8 array with a realistic value mix.

    Uses ``np.tile`` over a fixed pattern, with a per-row roll keyed
    by the seed so consecutive variants don't share the exact same
    layout (avoids the inner branch becoming trivially predictable
    in a way real data never is)."""
    rng = np.random.default_rng(seed)
    flat = np.empty(num_variants * num_samples * 2, dtype=np.int8)
    pattern = _GENOTYPE_VALUES
    for j in range(num_variants):
        offset = int(rng.integers(0, pattern.size))
        rolled = np.roll(pattern, offset)
        row_size = num_samples * 2
        repeats = (row_size + pattern.size - 1) // pattern.size
        row = np.tile(rolled, repeats)[:row_size]
        flat[j * row_size : (j + 1) * row_size] = row
    return flat.reshape(num_variants, num_samples, 2)


@dataclasses.dataclass
class _Chunk:
    """Synthetic per-variant chunk shared across all encoder runners.

    Fields are pre-built in shapes/dtypes each encoder accepts directly
    so the timed region contains only encoder work, never numpy fanout.
    The two ``bgen_*_prep`` slots hold the per-chunk byte-prepared inputs
    that ``vcztools.bgen._prepare_chunk`` produces — building them is
    chunk-level vectorised work that ``BgenEncoder._encode_chunk`` does
    once per chunk, so it's amortised outside the per-variant timing.
    """

    num_variants: int
    num_samples: int
    G: np.ndarray
    phased: np.ndarray
    # VCF arrays (C-contiguous, NPY_STRING / NPY_INT32 / NPY_FLOAT32 / NPY_BOOL).
    chrom_bytes: np.ndarray
    pos: np.ndarray
    id_bytes: np.ndarray
    ref_bytes: np.ndarray
    alt_bytes: np.ndarray
    qual: np.ndarray
    filter_ids_bytes: np.ndarray
    filter: np.ndarray
    # BGEN per-chunk prepared inputs (variable mode and fixed mode).
    bgen_var_prep: bgen._ChunkBytes
    bgen_fixed_prep: bgen._ChunkBytes


def _build_bgen_chunk_dict(
    G: np.ndarray, phased: np.ndarray, pos: np.ndarray, rsids: list, chrom: str
) -> tuple[dict, np.ndarray]:
    """Build the ``chunk`` dict shape ``bgen._prepare_chunk`` expects.

    Returns ``(chunk_dict, contig_ids)``. The synthetic dataset uses one
    contig string for every variant, so ``variant_contig`` is all zeros
    and ``contig_ids`` is a single-element array."""
    n = G.shape[0]
    contig_ids = np.array([chrom])
    chunk = {
        "call_genotype": G,
        "variant_allele": np.array([["A", "T"]] * n, dtype=np.str_),
        "variant_position": pos,
        "variant_contig": np.zeros(n, dtype=np.int32),
        "variant_id": np.array(rsids, dtype=np.str_),
        "call_genotype_phased": phased,
    }
    return chunk, contig_ids


def _make_chunk(num_variants: int, num_samples: int, seed: int) -> _Chunk:
    G = _make_genotypes(num_variants, num_samples, seed)
    phased = np.zeros((num_variants, num_samples), dtype=bool)

    chrom = "1"
    a1, a2 = "A", "T"
    rsids = [f"rs{j:08d}" for j in range(num_variants)]
    pos = np.arange(1, num_variants + 1, dtype=np.int32)

    chrom_max = len(chrom.encode("utf-8"))
    varid_max = max((len(r) for r in rsids), default=1)
    rsid_max = varid_max
    allele_max = 1

    bgen_chunk, contig_ids = _build_bgen_chunk_dict(G, phased, pos, rsids, chrom)
    bgen_var_prep = bgen._prepare_chunk(
        bgen_chunk, contig_ids, start=0, end=num_variants
    )
    bgen_fixed_prep = bgen._prepare_chunk(
        bgen_chunk,
        contig_ids,
        start=0,
        end=num_variants,
        varid_max_len=varid_max,
        rsid_max_len=rsid_max,
        chrom_max_len=chrom_max,
        allele_max_len=allele_max,
    )

    chrom_bytes = np.full(num_variants, chrom.encode("utf-8"), dtype=f"S{chrom_max}")
    id_bytes = np.array(
        [r.encode("utf-8") for r in rsids], dtype=f"S{varid_max}"
    ).reshape((num_variants, 1))
    ref_bytes = np.full(num_variants, a1.encode("utf-8"), dtype=f"S{allele_max}")
    alt_bytes = np.full((num_variants, 1), a2.encode("utf-8"), dtype=f"S{allele_max}")
    qual = np.full(num_variants, constants.FLOAT32_MISSING, dtype=np.float32)
    filter_ids_bytes = np.array([b"PASS"], dtype="S4")
    filter_arr = np.ones((num_variants, 1), dtype=bool)

    return _Chunk(
        num_variants=num_variants,
        num_samples=num_samples,
        G=G,
        phased=phased,
        chrom_bytes=chrom_bytes,
        pos=pos,
        id_bytes=id_bytes,
        ref_bytes=ref_bytes,
        alt_bytes=alt_bytes,
        qual=qual,
        filter_ids_bytes=filter_ids_bytes,
        filter=filter_arr,
        bgen_var_prep=bgen_var_prep,
        bgen_fixed_prep=bgen_fixed_prep,
    )


def _plink_slice(G: np.ndarray, start: int, end: int) -> bytes:
    # Matches the per-slice idiom in plink.BedEncoder._encode_chunk.
    return bytes(_vcztools.encode_plink(G[start:end]).data)


def _vcf_run(chunk: _Chunk) -> int:
    """Construct a VcfEncoder and run the per-variant encode loop.

    Returns total bytes (including the implicit trailing newline per
    line). Matches the layout of ``VcfWriter.write_chunk`` in vcf_writer.py.
    """
    encoder = _vcztools.VcfEncoder(
        chunk.num_variants,
        chunk.num_samples,
        chrom=chunk.chrom_bytes,
        pos=chunk.pos,
        id=chunk.id_bytes,
        ref=chunk.ref_bytes,
        alt=chunk.alt_bytes,
        qual=chunk.qual,
        filter_ids=chunk.filter_ids_bytes,
        filter=chunk.filter,
    )
    encoder.add_gt_field(chunk.G, chunk.phased)
    buflen = 1024
    total = 0
    for j in range(chunk.num_variants):
        while True:
            try:
                line = encoder.encode(j, buflen)
                break
            except _vcztools.VczBufferTooSmall:
                buflen *= 2
        # +1 for the newline that ``print(line, file=output)`` adds.
        total += len(line) + 1
    return total


def _bgen_block_slice(
    prep: bgen._ChunkBytes, level: int, start: int, end: int
) -> bytes:
    """Per-variant ``_encode_variant_block`` loop over ``prep[start:end]``.

    ``prep`` already holds chunk-level byte-prepared inputs (the
    vectorised ``_prepare_chunk`` work is amortised outside timing),
    so the timed region is the per-variant zlib + framing path that
    ``BgenEncoder._encode_variant_range`` runs in production."""
    position = prep.position
    parts = []
    for j in range(start, end):
        parts.append(
            bgen._encode_variant_block(
                varid_bytes=prep.varid[j],
                rsid_bytes=prep.rsid[j],
                chrom_bytes=prep.chrom[j],
                position=int(position[j]),
                allele1_bytes=prep.allele1[j],
                allele2_bytes=prep.allele2[j],
                geno_block_bytes=bytes(prep.geno_blocks[j]),
                compression_level=level,
            )
        )
    return b"".join(parts)


def _bgen_var_slice(chunk: _Chunk, level: int, start: int, end: int) -> bytes:
    return _bgen_block_slice(chunk.bgen_var_prep, level, start, end)


def _bgen_fixed_slice(chunk: _Chunk, start: int, end: int) -> bytes:
    return _bgen_block_slice(chunk.bgen_fixed_prep, 0, start, end)


def _encode_parallel(
    num_variants: int,
    executor: cf.ThreadPoolExecutor,
    block_variants: int,
    encode_slice,
) -> bytes:
    futures = [
        executor.submit(encode_slice, start, min(start + block_variants, num_variants))
        for start in range(0, num_variants, block_variants)
    ]
    out = bytearray()
    for f in futures:
        out.extend(f.result())
    return bytes(out)


def _time_one_sync(sync_call) -> float:
    t0 = time.perf_counter()
    sync_call()
    return time.perf_counter() - t0


def _time_one_parallel(
    num_variants: int,
    executor: cf.ThreadPoolExecutor,
    block_variants: int,
    encode_slice,
) -> float:
    t0 = time.perf_counter()
    _encode_parallel(num_variants, executor, block_variants, encode_slice)
    return time.perf_counter() - t0


def _median_rate(times: list, in_mib: float) -> float:
    return in_mib / statistics.median(times)


def _bytes_out_for(fmt: str, chunk: _Chunk, level: int) -> int:
    """Compute encoded byte count for one format, once, outside timing."""
    if fmt == "plink":
        return ((chunk.num_samples + 3) // 4) * chunk.num_variants
    if fmt == "vcf":
        return _vcf_run(chunk)
    if fmt == "bgen":
        return len(_bgen_var_slice(chunk, level, 0, chunk.num_variants))
    if fmt == "bgen-fixed":
        return len(_bgen_fixed_slice(chunk, 0, chunk.num_variants))
    raise ValueError(f"unknown format {fmt!r}")


def _make_runners(fmt: str, chunk: _Chunk, level: int):
    """Return ``(sync_call, slice_fn)``. ``slice_fn`` is None when no
    parallel slicing path is implemented for this format (VCF)."""
    if fmt == "plink":
        return (
            lambda: _vcztools.encode_plink(chunk.G),
            lambda s, e: _plink_slice(chunk.G, s, e),
        )
    if fmt == "vcf":
        return (lambda: _vcf_run(chunk), None)
    if fmt == "bgen":
        return (
            lambda: _bgen_var_slice(chunk, level, 0, chunk.num_variants),
            lambda s, e: _bgen_var_slice(chunk, level, s, e),
        )
    if fmt == "bgen-fixed":
        return (
            lambda: _bgen_fixed_slice(chunk, 0, chunk.num_variants),
            lambda s, e: _bgen_fixed_slice(chunk, s, e),
        )
    raise ValueError(f"unknown format {fmt!r}")


def _run_one_format(
    fmt: str,
    chunk: _Chunk,
    threads: int,
    block_bytes: int,
    repeats: int,
    level: int,
) -> None:
    sync_call, slice_fn = _make_runners(fmt, chunk, level)
    bytes_in = chunk.G.nbytes
    bytes_out = _bytes_out_for(fmt, chunk, level)

    # One untimed call to warm caches and (on first ever call) the
    # numpy/C extension boundary.
    sync_call()

    parallel = threads > 1 and slice_fn is not None
    if threads > 1 and slice_fn is None:
        click.echo(
            f"# {fmt}: no parallel slice path; "
            f"falling back to single-thread (--threads ignored)"
        )

    if not parallel:
        times = [_time_one_sync(sync_call) for _ in range(repeats)]
        threads_eff = 1
    else:
        block_variants = max(1, block_bytes // (chunk.num_samples * 2))
        with cf.ThreadPoolExecutor(
            max_workers=threads,
            thread_name_prefix=f"vcztools-encode-{fmt}-bench",
        ) as executor:
            times = [
                _time_one_parallel(
                    chunk.num_variants, executor, block_variants, slice_fn
                )
                for _ in range(repeats)
            ]
        threads_eff = threads

    median = statistics.median(times)
    fastest = min(times)
    slowest = max(times)

    in_mib = bytes_in / (1024 * 1024)
    out_mib = bytes_out / (1024 * 1024)

    click.echo(f"# {fmt}")
    click.echo(
        f"shape:        {chunk.num_variants} variants x {chunk.num_samples} samples"
    )
    click.echo(f"bytes_in:     {in_mib:.1f} MiB")
    click.echo(f"bytes_out:    {out_mib:.1f} MiB")
    click.echo(f"threads:      {threads_eff}")
    if threads_eff > 1:
        block_variants = max(1, block_bytes // (chunk.num_samples * 2))
        click.echo(
            f"block_bytes:  {block_bytes / (1024 * 1024):.1f} MiB "
            f"({block_variants} variants/block)"
        )
    if fmt == "bgen":
        click.echo(f"compression:  zlib level {level}")
    click.echo(f"repeats:      {repeats}")
    click.echo(
        f"time:         median {median * 1000:.1f} ms "
        f"(min {fastest * 1000:.1f}, max {slowest * 1000:.1f})"
    )
    click.echo(f"input rate:   {in_mib / median:.0f} MiB/s")
    click.echo(f"output rate:  {out_mib / median:.0f} MiB/s")


def _sweep_one_format(fmt: str, chunk: _Chunk, repeats: int, level: int) -> None:
    sync_call, slice_fn = _make_runners(fmt, chunk, level)
    in_mib = chunk.G.nbytes / (1024 * 1024)
    out_mib = _bytes_out_for(fmt, chunk, level) / (1024 * 1024)

    click.echo(f"# encode_{fmt} parallel sweep")
    click.echo(
        f"# shape: {chunk.num_variants} variants x {chunk.num_samples} samples "
        f"(in {in_mib:.1f} MiB, out {out_mib:.1f} MiB), repeats={repeats}"
    )
    if fmt == "bgen":
        click.echo(f"# compression: zlib level {level}")
    click.echo("")

    headers = ["threads"] + [f"{b // (1 << 20)} MiB" for b in _SWEEP_BLOCK_BYTES]
    click.echo("| " + " | ".join(headers) + " |")
    click.echo("|" + "|".join(["---"] * len(headers)) + "|")

    # Untimed warmup.
    sync_call()

    for threads in _SWEEP_THREADS:
        row = [str(threads)]
        if threads == 1:
            times = [_time_one_sync(sync_call) for _ in range(repeats)]
            rate = _median_rate(times, in_mib)
            row.extend([f"{rate:.0f}"] * len(_SWEEP_BLOCK_BYTES))
        elif slice_fn is None:
            # No parallel slicing path: leave the row blank rather than
            # repeat the threads=1 number (the cells aren't comparable).
            row.extend(["—"] * len(_SWEEP_BLOCK_BYTES))
        else:
            with cf.ThreadPoolExecutor(
                max_workers=threads,
                thread_name_prefix=f"vcztools-encode-{fmt}-bench",
            ) as executor:
                for block_bytes in _SWEEP_BLOCK_BYTES:
                    block_variants = max(1, block_bytes // (chunk.num_samples * 2))
                    times = [
                        _time_one_parallel(
                            chunk.num_variants,
                            executor,
                            block_variants,
                            slice_fn,
                        )
                        for _ in range(repeats)
                    ]
                    rate = _median_rate(times, in_mib)
                    row.append(f"{rate:.0f}")
        click.echo("| " + " | ".join(row) + " |")
    click.echo("")
    click.echo("# Cell values are input-MiB/s (median of repeats).")


@click.command()
@click.option("--variants", type=int, default=1000, show_default=True)
@click.option("--samples", type=int, default=100_000, show_default=True)
@click.option("--repeats", type=int, default=10, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option(
    "--threads",
    type=int,
    default=1,
    show_default=True,
    help="Worker threads. 1 = synchronous (existing single-thread bench).",
)
@click.option(
    "--block-bytes",
    type=int,
    default=10 * 1024 * 1024,
    show_default=True,
    help="Input-bytes target per sub-block. Ignored when --threads=1.",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice([*_FORMATS, "all"]),
    default="plink",
    show_default=True,
    help="Encoder to benchmark. 'all' runs every format in sequence.",
)
@click.option(
    "--compression-level",
    type=int,
    default=6,
    show_default=True,
    help="zlib level for --format bgen (variable-size BGEN). "
    "Ignored for other formats. 6 is the zlib default.",
)
@click.option(
    "--sweep",
    is_flag=True,
    help="Run the canonical (threads, block_bytes) matrix and emit a "
    "markdown table of MiB/s per format.",
)
def main(
    variants: int,
    samples: int,
    repeats: int,
    seed: int,
    threads: int,
    block_bytes: int,
    fmt: str,
    compression_level: int,
    sweep: bool,
) -> None:
    """Time vcztools encoder kernels over an in-memory chunk."""
    chunk = _make_chunk(variants, samples, seed)
    formats = list(_FORMATS) if fmt == "all" else [fmt]
    for i, f in enumerate(formats):
        if i > 0:
            click.echo("")
        if sweep:
            _sweep_one_format(f, chunk, repeats, compression_level)
        else:
            _run_one_format(f, chunk, threads, block_bytes, repeats, compression_level)


if __name__ == "__main__":
    main()
