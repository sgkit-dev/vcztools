(sec-bgen)=
# BGEN output

The {ref}`vcztools view-bgen<cmd-vcztools-view-bgen>` command writes Oxford
BGEN output from a VCZ store. Output profile: **layout 2, zlib-compressed, 8 bits per
probability, biallelic, embedded sample IDs**. Diploid, haploid (e.g.
chrX in males, mitochondrial) and mixed-ploidy stores (mixed-sex chrX)
are supported via per-sample ploidy bytes. This is the consumer
lowest-common-denominator: REGENIE, SAIGE, BOLT-LMM, BGENIE, qctool,
and PLINK 2 all accept it without further conversion.

Default: stream the `.bgen` payload to stdout (mirroring
{ref}`vcztools view<cmd-vcztools-view>`). Pass `-o STEM` to write files including the `.bgen.bgi` (bgenix
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

The `-o` value is a stem taken **verbatim** — `-o sample` produces
`sample.bgen`, `sample.bgen.bgi`, `sample.sample`; `-o sample.bgen`
would produce `sample.bgen.bgen` etc.

Sample, region and filter selection mirrors
{ref}`vcztools view<cmd-vcztools-view>`
(`-s`/`-S`/`-r`/`-R`/`-t`/`-T`/`-i`/`-e`/`-v`/`-V`/`-m`/`-M`); the
per-flag reference is in {ref}`sec-cli-ref`.

:::{note}
Note that the filtering semantics differs slightly from ``vcztools view``:
see {ref}`sec-plink-subset-filtering` for details.
:::

## Sample-ID embedding

By default the BGEN header carries sample IDs (the `SAMPLE_IDS_PRESENT`
flag is set), so the `.bgen` is self-describing. Pass `--no-header-samples`
to clear the flag and omit the sample-ID block; in that case downstream
tools must rely on the `.sample` sidecar — combining `--no-header-samples`
with `--no-sample-file` leaves sample IDs nowhere and logs a warning
(most readers will refuse the resulting BGEN).

## Hard calls and probabilities

BGEN's native data type is genotype probabilities (the equivalent of
VCF `GP`). `view-bgen` encodes the hard calls in `call_genotype` as
1.0 probability on the called genotype and 0.0 elsewhere. At 8-bit
precision this round-trips exactly, so reading the output back with
any BGEN reader recovers the original hard calls.

Genotype probability (`call_GP`) input is not yet supported.

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
| `<stem>.bgen.bgi` | SQLite | on | `--no-bgi` | bgenix index; required by SAIGE Step 2; strongly recommended for REGENIE |
| `<stem>.sample` | text | on | `--no-sample-file` | Oxford `.sample` (sample IDs); required by SAIGE, accepted by REGENIE |

The `.bgen` header also embeds sample IDs, so consumers that read sample
names from the BGEN itself (qctool, BGENIE, PLINK 2) work without the
`.sample` sidecar. SAIGE Step 2 still requires the `.sample` file. Pass
`--no-header-samples` to drop the in-header IDs (e.g. for samples whose
identifiers must not appear in the binary).

Streaming mode (no `-o`) writes only the `.bgen` payload to stdout;
sidecar flags have no effect.

## Reading `view-bgen` output with downstream tools

| Tool | Works as-is? | Notes |
| --- | --- | --- |
| qctool | yes | layout 2 / zlib / 8-bit is fully supported |
| PLINK 2 | yes | `--bgen out.bgen ref-first` reads it; `ref-first` matches our REF/ALT ordering |
| REGENIE | yes | provide `--bgen out.bgen --bgi out.bgen.bgi --sample out.sample`; `.bgi` is strongly recommended for Step 2 speed |
| BOLT-LMM | yes (≥ v2.4) | accepts BGEN v1.2 at 8-bit only; phased & missing supported since 2.4 / 2.3.3 respectively |
| SAIGE | yes | Step 2 requires `--bgenFile`, `--bgenFileIndex` (= `.bgi`), `--sampleFile` (= `.sample`) |
| BGENIE | yes | UK Biobank-style; uses the `.bgi` for indexed access |

## Multi-allelic variants

Multi-allelic variants are rejected by default. Although BGEN layout 2
supports them natively, every major downstream tool above assumes
biallelic SNPs. Two ways to handle them:

- **Skip** with `--max-alleles 2`:

  ```bash
  vcztools view-bgen sample.vcz -o sample --max-alleles 2
  ```

- **Split** before conversion with ``bcftools norm -m-``. Each ALT
  allele becomes its own biallelic record.

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

VCZ stores haploid data either as `(V, S, 1)` arrays or as `(V, S, 2)`
arrays with `-2` in the second slot (the haploid-padding sentinel).
`view-bgen` accepts both: shape `(V, S, 1)` is promoted to the
`-2`-padded form before encoding. A `-2` in slot 0 (zero-ploidy /
unused sample) is not representable in BGEN and surfaces as a
`ValueError`. The encoder is biallelic and rejects any
``call_genotype`` value outside ``{-2, -1, 0, 1}`` with a
``ValueError``; pre-split multi-allelic input with ``bcftools norm
-m-`` if your store has higher allele indices.

The per-variant header `Pmin` / `Pmax` reflect the actual range of
sample ploidies in each variant.

## Missingness

A sample is treated as missing for a variant if any of its diploid
alleles is `-1`, or if its haploid allele (under `-2` padding) is
`-1`. Missing samples have the BGEN ploidy/missing byte's high bit
set; their probability bytes are zeroed (the BGEN spec says these
bytes should be ignored when the missing flag is set, but zeroing
keeps the output deterministic).

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

`view-bgen` zlib-compresses each variant's genotype-probability block
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
level 1. Most BGEN workloads are write-once-and-read-by-one-tool, so
encode throughput wins.

If you're producing an archival BGEN (write once, read many),
`--compression-level 9` is worth the time. If you're streaming
through a tool that re-compresses anyway, `--compression-level 0`
(stored) is fastest.

## Limitations

- **Layout 2, 8-bit, zlib only.** Higher precision (16/32-bit) and
  alternative compressions (zstd, none) are not exposed. Adding them
  is straightforward when a downstream tool requires them.
- **Ploidy 1 or 2.** Polyploid input (`call_genotype.shape[2] > 2`)
  is rejected.
- **Hard calls only.** Genotype probabilities (`call_GP`) are not
  read; the encoder emits 1.0/0.0 from `call_genotype` only.
- **Whitespace in sample IDs.** Rejected with a clear error message —
  the `.sample` format is whitespace-separated.

## Validation

`view-bgen` output is validated end-to-end against two external
references:

- **`bgen-reader`** (Limix) parses every test fixture's output and
  recovers the source VCZ genotypes exactly at 8-bit precision. See
  `tests/test_bgen_validation.py::TestBgenReaderRoundtrip`.
- **PLINK 2** (`--export bgen-1.2 bits=8 ref-first --double-id`)
  produces semantically equivalent BGEN — same variant count, same
  positions, same alleles, same dosage matrix — for fixtures that
  exercise plink2's BGEN export code path. See
  `TestPlink2BgenCrossCheck`.

Known field-level differences between `view-bgen` and `plink2 --export
bgen-1.2`, observed in the cross-check tests:

| Field | `view-bgen` | `plink2 --export bgen` |
| --- | --- | --- |
| Sample IDs (BGEN header) | bare IID (`HG00096`) | `FID_IID` (`HG00096_HG00096` or `0_NA00001`) |
| Chromosome names | passes through (`chr22`) | normalises like `--make-bed` (`22`) |
| Variant ID (rsid) | `.` if VCZ `variant_id` is empty | synthesises `chrom:pos:ref:alt` |
| Compression bytes | zlib at Python's default level | zlib at plink2's level |

Decoded probability matrices match exactly. Compression-byte and
metadata-format differences are documented but not aligned.

PLINK 2 v2.00a6 has a known assertion (`compressed_bytect`) that
crashes on inputs with mixed-phase variants — `view-bgen` handles
them cleanly (degrades to unphased per-variant with a warning). The
test suite asserts this divergence on the bundled `sample.vcf.gz`
fixture so a future plink2 release fixing the bug surfaces here.

## Fixed-size random-access encoding (`BgenEncoder`)

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
