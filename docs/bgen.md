(sec-bgen)=
# BGEN output

The {ref}`vcztools view-bgen<cmd-vcztools-view-bgen>` command writes
BGEN output from a VCZ store.
By default, we stream the `.bgen` payload to stdout (mirroring
{ref}`vcztools view<cmd-vcztools-view>`). Pass `-o STEM` to write files
including the `.bgen.bgi` (bgenix
SQLite index) and `.sample` (Oxford text) sidecars, both on by default
and individually suppressible.

```bash
# Streaming: pipe straight into another tool, or redirect to a file.
vcztools view-bgen sample.vcz > sample.bgen

# File output with the full sidecar set.
vcztools view-bgen sample.vcz -o sample
# produces sample.bgen, sample.bgen.bgi, sample.sample

# File output, .bgen only.
vcztools view-bgen sample.vcz -o sample --no-bgi --no-sample-file
```

The `-o` value is a stem taken **verbatim**: `-o sample` produces
`sample.bgen`, `sample.bgen.bgi`, `sample.sample`.

Sample, region and filter selection mirrors
{ref}`vcztools view<cmd-vcztools-view>`
(`-s`/`-S`/`-r`/`-R`/`-t`/`-T`/`-i`/`-e`/`-v`/`-V`/`-m`/`-M`); the
per-flag reference is in {ref}`sec-cli-ref`.

:::{note}
Note that the filtering semantics differs slightly from ``vcztools view``:
see {ref}`sec-bgen-subset-filtering` for details.
:::

## Sample-ID embedding

By default the BGEN header carries sample IDs (the `SAMPLE_IDS_PRESENT`
flag is set), so the `.bgen` is self-describing. Pass `--no-header-samples`
to clear the flag and omit the sample-ID block.

## Hard calls and probabilities

BGEN's native data type is genotype probabilities.
`view-bgen` currently encodes the hard calls in `call_genotype` as
1.0 probability on the called genotype and 0.0 elsewhere.

## Phasing

If the VCZ store has a `call_genotype_phased` field, each variant is
emitted as phased iff every sample is phased for that variant. (BGEN
has a single phase flag per variant; mixed-phase variants degrade
silently to unphased and a warning is logged.) Stores without
`call_genotype_phased` emit unphased.

## Sidecars

With `-o STEM`, `view-bgen` writes the `.bgen` payload plus, by default,
two sidecar files:

| File | Format | Default | Suppress with | Purpose |
| --- | --- | --- | --- | --- |
| `<stem>.bgen` | binary | always | — | the BGEN payload (header + variant blocks) |
| `<stem>.bgen.bgi` | SQLite | on | `--no-bgi` | bgenix index|
| `<stem>.sample` | text | on | `--no-sample-file` | Oxford `.sample` (sample IDs) |

The `.bgen` header also embeds sample IDs, so consumers that read sample
names from the BGEN itself (qctool, BGENIE, PLINK 2) work without the
`.sample` sidecar. Pass
`--no-header-samples` to drop the in-header IDs (e.g. for samples whose
identifiers must not appear in the binary).

Streaming mode (no `-o`) writes only the `.bgen` payload to stdout;
sidecar flags have no effect.

## Multi-allelic variants

Multi-allelic variants are rejected by default, and must be
filtered out:

```bash
vcztools view-bgen sample.vcz -o sample --max-alleles 2
```

## Ploidy

`view-bgen` supports both haploid and diploid input, plus mixed
ploidy where some samples are haploid and some are diploid for the
same variant (e.g. chrX with both sexes). The classification depends
on each `(a, b)` pair in `call_genotype`:

| VCZ pair | Meaning | BGEN ploidy byte | Prob bytes |
| --- | --- | --- | --- |
| `a >= 0`, `b >= 0` | diploid call | `0x02` | 2 |
| `a >= 0`, `b == -2` | haploid call | `0x01` | 1 |
| `a == -1`, `b == -1` | missing diploid | `0x82` | 2 (zero) |
| `a == -1`, `b == -2` | missing haploid | `0x81` | 1 (zero) |
| half-missing diploid | missing diploid | `0x82` | 2 (zero) |

The per-variant header `Pmin` / `Pmax` reflect the actual range of
sample ploidies in each variant.

## Missingness

A sample is treated as missing for a variant if any of its diploid
alleles is `-1`, or if its haploid allele is `-1`.
Missing samples have the BGEN ploidy/missing byte's high bit
set and their probability bytes are zeroed.

(sec-bgen-subset-filtering)=
## Filtering over the sample subset

`-i`/`-e` expression filters that reference sample-derived INFO fields
(`AC`, `AN`, `AF`, `NS`) are evaluated **over the selected samples**. With
a `-s`/`-S` subset, those values are recomputed for that subset rather
than read from the file's stored (full-cohort) INFO, so
`view-bgen -s cohort.txt -i 'AC>0'` keeps the variants polymorphic *in
your cohort*.

This is deliberately **different from
{ref}`vcztools view<cmd-vcztools-view>` and
{ref}`vcztools query<cmd-vcztools-query>`**, which follow bcftools and
evaluate `-i/-e` against the original
record's stored INFO. A BGEN fileset feeds an association analysis on the
exported cohort, so filters should reflect that cohort, not the source
dataset; it is also cheaper on large datasets, since only the selected
samples are read. The allele-based filters (`-m`/`-M`/`-v`/`-V`) remain
record-level.

## Compression level

Per the format specification, `view-bgen`
zlib-compresses each variant's genotype-probability block
independently. The level is controlled by `--compression-level`, which
accepts the standard zlib values (`-1` = zlib default ≈ 6; `0` =
stored, no DEFLATE; `1`-`9` = fastest to maximum). The BGEN flag word
always advertises `COMPRESSION_ZLIB` regardless of level, so every
reader handles the output identically.

**The default is `--compression-level 1`.** Hard-call BGEN payloads
are short, repetitive byte runs (mostly 1.0/0.0 in 8-bit form,
duplicated across samples), so the longer-match Lempel-Ziv search at
level 6 (zlib default) buys very little: in our benchmarks level 6
shrinks the file by ~10-30% but spends several times more CPU than
level 1.

## Limitations

- **Layout 2, 8-bit, zlib only.** Higher precision (16/32-bit) and
  alternative compressions (zstd, none) are not exposed.
- **Ploidy 1 or 2.** Polyploid input (`call_genotype.shape[2] > 2`)
  is rejected.
- **Hard calls only.** Genotype probabilities (`call_GP`) are not
  read; the encoder emits 1.0/0.0 from `call_genotype` only.
- **Whitespace in sample IDs.** Rejected with a clear error message —
  the `.sample` format is whitespace-separated.

## Fixed-size random-access encoding

For applications that need to address arbitrary regions of the encoded
`.bgen` byte stream without iterating from the start — FUSE
filesystems, HTTP range serving, and similar — vcztools also exposes
a Python-only `BgenEncoder` class, the sibling of `plink.BedEncoder`:

```python
from vcztools import bgen, retrieval

reader = retrieval.VczReader("sample.vcz")
with bgen.BgenEncoder(reader) as enc:
    print(enc.bgen_size, enc.bytes_per_variant)
    # POSIX-read semantics: enc.read(off, size) returns bytes
    chunk = enc.read(0, 4096)
```

`BgenEncoder` requires uniform ploidy across the store. The encoder
auto-detects haploid vs diploid from `reader.call_genotype.shape[2]`
(1 or 2); a mixed-ploidy store (a `-2` sentinel under declared
diploid) raises `NotImplementedError` lazily on read with a message
pointing users at `write_bgen` (the variable-size streaming encoder).

Every per-variant block is exactly `bytes_per_variant` bytes wide, so
`byte offset → variant index` is O(1):

```
bytes_per_variant = 28 + total_string_length + zlib_stored_size(geno_size)

geno_size = 10 + (uniform_ploidy + 1) * num_samples
```

For haploid stores `geno_size = 10 + 2 * num_samples`; for diploid
`geno_size = 10 + 3 * num_samples`.

The five BGEN string fields (varid, rsid, chrom, allele1, allele2) share
a single `total_string_length` budget per variant. Four of them — chrom,
allele1, allele2, and whichever of varid/rsid is selected by
`variant_id_field` — are emitted at their actual UTF-8 byte lengths.
The fifth slot is the **padding field**: its content is
`b"." + pad_byte * (slack - 1)`, where `slack` is whatever's left of
`total_string_length` after the other four. The genotype block is
emitted as zlib **level 0** (stored, no DEFLATE) so its compressed
length is a deterministic function of the uncompressed length.

Defaults target biobank biallelic SNP-array data:

| Argument | Default | Notes |
| --- | --- | --- |
| `total_string_length` | 64 | combined byte budget for all five string fields per variant |
| `pad_byte` | `b"."` | single-byte pad written into the padding field after the leading `"."` |
| `variant_id_field` | `"rsid"` | BGEN slot that carries the zarr `variant_id`; the other slot is the padding field |
| `encode_threads` | 4 | thread pool size for parallel encode |
| `encode_block_bytes` | 10 MiB | input target per sub-block |

If a variant's content sums past `total_string_length - 1` (the padding
field can't even fit its leading `"."`), `read()` raises `ValueError`
naming the variant index and the configured `total_string_length`.
Stores with longer rsIDs / alleles opt in by passing a larger
`total_string_length`.

The output `.bgen` is larger than `view-bgen`'s zlib-compressed
output — it runs no real compression on genotype data and reserves
`total_string_length` bytes per variant for strings — but the byte
stream is addressable and bytes-per-variant is fully known up front. The encoder serves the `.bgen` stream only; `write_sample()`
remains the path for the `.sample` sidecar, and a `.bgi` index (if
needed) can be built from the deterministic offsets
(`header_size + i * bytes_per_variant`) without iterating the encoder.

Unlike `write_bgen`, the encoder does not auto-materialise a configured
`set_variant_filter` — it raises `NotImplementedError` at construction.
Apply the filter externally or use `set_variants` first.

## See also

- {ref}`The view-bgen flag reference<cmd-vcztools-view-bgen>`.
- [BGEN format specification](https://www.chg.ox.ac.uk/~gav/bgen_format/spec/latest.html).
- [bgenix index file format](https://enkre.net/cgi-bin/code/bgen/wiki/The%20bgenix%20index%20file%20format).
