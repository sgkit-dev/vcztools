"""CPU-only encoding microbenchmarks for the formats vcztools writes.

Builds a synthetic ``call_genotype`` chunk in memory and drives each
format's encoder class through its native chunk-write hook — no Zarr,
no I/O, no readahead pipeline. Gives the upper bound the corresponding
end-to-end matrix benchmarks (``output_plink`` for plink, ``view_vcf``
for VCF, ``write_bgen`` for BGEN) can asymptotically approach.

Formats:

* ``plink`` — :class:`vcztools.plink.BedEncoder`. Timed call:
  ``encoder._encode_chunk(chunk_dict) -> bytes``.
* ``vcf`` — :class:`vcztools.vcf_writer.VcfWriter`. Timed call:
  ``writer.write_chunk(chunk_dict)``; output sink is a write-counting
  null stream so repeated calls don't accumulate bytes.
* ``bgen`` — :class:`vcztools.bgen.BgenEncoder` (fixed-size, zlib
  level 0). Timed call: ``encoder._encode_chunk(chunk_dict) -> bytes``.
* ``bgen_python`` — :class:`vcztools.bgen.BgenEncoder` with its per-
  variant inner loop swapped back to the pre-C-kernel Python
  implementation. Same chunk-write hook, same prepare path; lets the
  C kernel be timed against the Python loop it replaced.

Run from the repo root::

    uv run python performance/encoding_microbench.py
    uv run python performance/encoding_microbench.py --format all
    uv run python performance/encoding_microbench.py --format all --sweep

Defaults to one variant-chunk's worth of work on the standard bench
shape (1000 variants x 100000 samples). Override with ``--variants``
and ``--samples``; ``--repeats`` controls the number of timed calls.

``--threads`` maps onto each encoder's ``encode_threads`` knob.
``--block-bytes`` maps onto ``encode_block_bytes`` for plink and bgen
(input-bytes target per parallel sub-block); VCF has no equivalent
knob — its parallelism splits each chunk into ``encode_threads``
contiguous blocks — so ``--block-bytes`` is ignored for ``--format vcf``.
``--sweep`` runs the canonical ``(threads, block_bytes)`` matrix for
plink and bgen, and a one-dimensional ``threads``-only table for VCF.
"""

import statistics
import time

import click
import numpy as np

from vcztools import bgen, plink, retrieval, vcf_writer

# Mix of values valid for biallelic input across all three encoders:
# 0/1 (REF/ALT calls) and -1 (missing sentinel). The BGEN C kernel
# rejects values outside {-2, -1, 0, 1} for biallelic, and further
# rejects -2 in slot 0 (haploid sentinel must sit in slot 1), so the
# pattern stays in the all-diploid-biallelic regime that all three
# encoders share — sufficient for an encode-rate microbench.
_GENOTYPE_VALUES = np.array([0, 1, 0, 1, 0, 0, 1, 1, -1, 0], dtype=np.int8)

_SWEEP_THREADS = (1, 2, 4, 8)
_SWEEP_BLOCK_BYTES = (1 << 20, 4 << 20, 10 << 20, 40 << 20)

_FORMATS = ("plink", "vcf", "bgen", "bgen_python")


def _encode_variant_range_python(
    self,
    chunk: dict,
    chunk_strings,
    start: int,
    end: int,
    phase_counts: list[int],
    out_view: np.ndarray,
) -> None:
    """Pre-C-kernel implementation of BgenEncoder._encode_variant_range.

    Identical control flow to the production version except the per-
    variant loop is back in Python: one ``_encode_variant_block`` call
    per variant, packing the BGEN framing struct-by-struct with
    ``zlib.compress(_, 0)`` per variant. Used by the ``bgen_python``
    benchmark format to expose the speed-up from the C kernel."""
    prep = bgen._prepare_chunk(chunk, chunk_strings, start=start, end=end)
    phase_counts.append(prep.mixed_phase_count)
    bpv = self._bytes_per_variant
    n = end - start
    uniform_len = self._uniform_geno_size
    if not (prep.geno_block_lens == uniform_len).all():
        raise NotImplementedError(
            "BgenEncoder requires uniform ploidy across all samples "
            "and variants; this chunk contains mixed ploidy."
        )
    buf = bytearray(n * bpv)
    for j in range(n):
        block = bgen._encode_variant_block(
            varid_bytes=prep.varid[j],
            rsid_bytes=prep.rsid[j],
            chrom_bytes=prep.chrom[j],
            position=int(prep.position[j]),
            allele1_bytes=prep.allele1[j],
            allele2_bytes=prep.allele2[j],
            geno_block_bytes=bytes(prep.geno_blocks[j, :uniform_len]),
            compression_level=0,
        )
        buf[j * bpv : (j + 1) * bpv] = block
    out_view[:] = np.frombuffer(bytes(buf), dtype=np.uint8)


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


def _make_chunk(num_variants: int, num_samples: int, seed: int) -> dict:
    """Build the chunk dict that all three encoders consume.

    The keys match ``VczReader.variant_chunks`` output for the iterator
    fields BedEncoder, BgenEncoder and VcfWriter request — so the dict
    can be fed straight into each class's chunk-write hook."""
    G = _make_genotypes(num_variants, num_samples, seed)
    phased = np.zeros((num_variants, num_samples), dtype=bool)
    rsids = [f"rs{j:08d}" for j in range(num_variants)]
    return {
        "call_genotype": G,
        "call_genotype_phased": phased,
        "variant_allele": np.array([["A", "T"]] * num_variants, dtype=np.str_),
        "variant_position": np.arange(1, num_variants + 1, dtype=np.int32),
        "variant_contig": np.zeros(num_variants, dtype=np.int32),
        "variant_id": np.array(rsids, dtype=np.str_),
    }


class _FakeReader:
    """Minimal VczReader stub: exposes only the attributes the encoder
    classes touch at construction and (for VcfWriter) at write_chunk
    time. No real iterator, no Zarr — the bench drives ``_encode_chunk``
    / ``write_chunk`` directly with an in-memory chunk dict."""

    def __init__(self, num_variants: int, num_samples: int):
        self.variant_filter = None
        self.sample_ids = np.array([f"s{i}" for i in range(num_samples)], dtype=np.str_)
        self.samples_selection = np.arange(num_samples)
        self.field_names = (
            "call_genotype",
            "call_genotype_phased",
            "variant_allele",
            "variant_contig",
            "variant_position",
            "variant_id",
        )
        # contig_ids is str-typed (BgenEncoder calls np.strings.encode on
        # it); contigs is fixed-length bytes (the VCF C encoder reads it
        # via NPY_STRING). The real VczReader exposes them with these
        # dtypes — see retrieval.py:contigs / retrieval.py:contig_ids.
        self.contig_ids = np.array(["1"], dtype=np.str_)
        self.contigs = np.array([b"1"], dtype="S1")
        self.filters = np.array([b"PASS"], dtype="S4")
        self._counts = np.array([num_variants], dtype=np.int64)
        self._num_variants = num_variants
        self._num_samples = num_samples

    def variant_counts_per_chunk(self) -> np.ndarray:
        return self._counts

    def get_field_info(self, name: str) -> retrieval.FieldInfo:
        if name == "call_genotype":
            return retrieval.FieldInfo(
                name=name,
                dtype=np.dtype("int8"),
                shape=(self._num_variants, self._num_samples, 2),
                dims=("variants", "samples", "ploidy"),
                attrs={},
            )
        raise KeyError(name)


class _CountingNull:
    """Write-counting sink that stands in for an open file.

    VcfWriter writes ASCII text via ``self.output.write(block_text)``;
    we discard the bytes and just track the cumulative count. This
    avoids ~hundreds of MiB of growth across repeats and isolates the
    encoder kernel cost from any sink-side memcpy."""

    def __init__(self):
        self.bytes_written = 0

    def write(self, s) -> int:
        n = len(s)
        self.bytes_written += n
        return n


def _open_encoder(fmt: str, reader: _FakeReader, threads: int, block_bytes: int):
    """Construct the encoder context manager for ``fmt``.

    Returns ``(encoder_cm, encode_call)`` where ``encode_call(enc)``
    runs one chunk-encode and returns the encoded byte count."""
    if fmt == "plink":
        encoder = plink.BedEncoder(
            reader, encode_threads=threads, encode_block_bytes=block_bytes
        )

        def encode_call(enc, chunk):
            return len(enc._encode_chunk(chunk))

        return encoder, encode_call
    if fmt == "bgen":
        encoder = bgen.BgenEncoder(
            reader, encode_threads=threads, encode_block_bytes=block_bytes
        )

        def encode_call(enc, chunk):
            return len(enc._encode_chunk(chunk))

        return encoder, encode_call
    if fmt == "bgen_python":
        encoder = bgen.BgenEncoder(
            reader, encode_threads=threads, encode_block_bytes=block_bytes
        )
        # Swap the per-variant loop back to the pre-C-kernel Python
        # implementation. _encode_chunk dispatches via this method (one
        # call per parallel sub-block), so this is the smallest patch
        # that exercises exactly the old code path.
        encoder._encode_variant_range = _encode_variant_range_python.__get__(
            encoder, type(encoder)
        )

        def encode_call(enc, chunk):
            return len(enc._encode_chunk(chunk))

        return encoder, encode_call
    if fmt == "vcf":
        sink = _CountingNull()
        writer = vcf_writer.VcfWriter(reader, sink, encode_threads=threads)

        def encode_call(enc, chunk):
            sink.bytes_written = 0
            enc.write_chunk(chunk)
            return sink.bytes_written

        return writer, encode_call
    raise ValueError(f"unknown format {fmt!r}")


def _time_repeats(
    fmt: str,
    chunk: dict,
    reader: _FakeReader,
    threads: int,
    block_bytes: int,
    repeats: int,
) -> tuple[list[float], int]:
    """Run ``repeats`` timed chunk-encodes for ``fmt``.

    Returns ``(times, bytes_out)``. One untimed warmup happens inside
    the context manager so the executor is alive across the timed
    work."""
    encoder_cm, encode_call = _open_encoder(fmt, reader, threads, block_bytes)
    times = []
    with encoder_cm as enc:
        bytes_out = encode_call(enc, chunk)
        for _ in range(repeats):
            t0 = time.perf_counter()
            bytes_out = encode_call(enc, chunk)
            times.append(time.perf_counter() - t0)
    return times, bytes_out


def _run_one_format(
    fmt: str,
    chunk: dict,
    reader: _FakeReader,
    threads: int,
    block_bytes: int,
    repeats: int,
) -> None:
    bytes_in = chunk["call_genotype"].nbytes
    times, bytes_out = _time_repeats(fmt, chunk, reader, threads, block_bytes, repeats)

    median = statistics.median(times)
    fastest = min(times)
    slowest = max(times)

    in_mib = bytes_in / (1024 * 1024)
    out_mib = bytes_out / (1024 * 1024)

    num_variants = chunk["call_genotype"].shape[0]
    num_samples = chunk["call_genotype"].shape[1]

    click.echo(f"# {fmt}")
    click.echo(f"shape:        {num_variants} variants x {num_samples} samples")
    click.echo(f"bytes_in:     {in_mib:.1f} MiB")
    click.echo(f"bytes_out:    {out_mib:.1f} MiB")
    click.echo(f"threads:      {threads}")
    if fmt != "vcf" and threads > 1:
        block_variants = max(1, block_bytes // (num_samples * 2))
        click.echo(
            f"block_bytes:  {block_bytes / (1024 * 1024):.1f} MiB "
            f"({block_variants} variants/block)"
        )
    click.echo(f"repeats:      {repeats}")
    click.echo(
        f"time:         median {median * 1000:.1f} ms "
        f"(min {fastest * 1000:.1f}, max {slowest * 1000:.1f})"
    )
    click.echo(f"input rate:   {in_mib / median:.0f} MiB/s")
    click.echo(f"output rate:  {out_mib / median:.0f} MiB/s")


def _sweep_one_format(fmt: str, chunk: dict, reader: _FakeReader, repeats: int) -> None:
    bytes_in = chunk["call_genotype"].nbytes
    in_mib = bytes_in / (1024 * 1024)
    num_variants = chunk["call_genotype"].shape[0]
    num_samples = chunk["call_genotype"].shape[1]

    # One untimed pre-run to size the output and warm the C extension
    # / first-touch numpy paths.
    _, bytes_out = _time_repeats(fmt, chunk, reader, 1, _SWEEP_BLOCK_BYTES[0], 1)
    out_mib = bytes_out / (1024 * 1024)

    # Collect median times once, then derive both input-MiB/s and
    # output-MiB/s from them — avoids running the sweep twice.
    if fmt == "vcf":
        block_bytes_axis = (_SWEEP_BLOCK_BYTES[0],)
    else:
        block_bytes_axis = _SWEEP_BLOCK_BYTES
    medians: dict[tuple[int, int], float] = {}
    for threads in _SWEEP_THREADS:
        for block_bytes in block_bytes_axis:
            times, _ = _time_repeats(fmt, chunk, reader, threads, block_bytes, repeats)
            medians[(threads, block_bytes)] = statistics.median(times)

    click.echo(f"# encode_{fmt} parallel sweep")
    click.echo(
        f"# shape: {num_variants} variants x {num_samples} samples "
        f"(in {in_mib:.1f} MiB, out {out_mib:.1f} MiB), repeats={repeats}"
    )
    click.echo("")

    def emit_table(label: str, total_mib: float) -> None:
        if fmt == "vcf":
            click.echo(f"## {label}")
            click.echo("| threads | MiB/s |")
            click.echo("|---|---|")
            for threads in _SWEEP_THREADS:
                rate = total_mib / medians[(threads, _SWEEP_BLOCK_BYTES[0])]
                click.echo(f"| {threads} | {rate:.0f} |")
        else:
            click.echo(f"## {label}")
            headers = ["threads"] + [
                f"{b // (1 << 20)} MiB" for b in _SWEEP_BLOCK_BYTES
            ]
            click.echo("| " + " | ".join(headers) + " |")
            click.echo("|" + "|".join(["---"] * len(headers)) + "|")
            for threads in _SWEEP_THREADS:
                row = [str(threads)]
                for block_bytes in _SWEEP_BLOCK_BYTES:
                    rate = total_mib / medians[(threads, block_bytes)]
                    row.append(f"{rate:.0f}")
                click.echo("| " + " | ".join(row) + " |")
        click.echo("")

    emit_table("input MiB/s", in_mib)
    emit_table("output MiB/s", out_mib)
    click.echo("# Cell values are MiB/s (median of repeats); input table")
    click.echo("# divides bytes_in by median time, output table divides bytes_out.")


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
    help="Encoder worker threads. Maps onto each encoder's encode_threads.",
)
@click.option(
    "--block-bytes",
    type=int,
    default=10 * 1024 * 1024,
    show_default=True,
    help="Input-bytes target per parallel sub-block. Maps onto "
    "encode_block_bytes for plink and bgen. Ignored for vcf (VcfWriter "
    "splits by threads, not by byte budget).",
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
    "--sweep",
    is_flag=True,
    help="Run the canonical sweep matrix per format and emit a markdown "
    "table of MiB/s: (threads x block_bytes) for plink and bgen, "
    "(threads) only for vcf.",
)
def main(
    variants: int,
    samples: int,
    repeats: int,
    seed: int,
    threads: int,
    block_bytes: int,
    fmt: str,
    sweep: bool,
) -> None:
    """Time vcztools encoder classes over an in-memory chunk."""
    chunk = _make_chunk(variants, samples, seed)
    reader = _FakeReader(variants, samples)
    formats = list(_FORMATS) if fmt == "all" else [fmt]
    for i, f in enumerate(formats):
        if i > 0:
            click.echo("")
        if sweep:
            _sweep_one_format(f, chunk, reader, repeats)
        else:
            _run_one_format(f, chunk, reader, threads, block_bytes, repeats)


if __name__ == "__main__":
    main()
