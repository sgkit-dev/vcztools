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

## Format details

### Sample-ID embedding

By default the BGEN header carries sample IDs (the `SAMPLE_IDS_PRESENT`
flag is set), so the `.bgen` is self-describing. Pass `--no-header-samples`
to clear the flag and omit the sample-ID block.

### Hard calls and probabilities

BGEN's native data type is genotype probabilities.
`view-bgen` currently encodes the hard calls in `call_genotype` as
1.0 probability on the called genotype and 0.0 elsewhere.

### Phasing

If the VCZ store has a `call_genotype_phased` field, each variant is
emitted as phased iff every sample is phased for that variant. (BGEN
has a single phase flag per variant; mixed-phase variants degrade
silently to unphased and a warning is logged.) Stores without
`call_genotype_phased` emit unphased. Use `--unphased` to force
unphased output regardless of the `call_genotype_phased` field.

### Sidecars

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

### Multi-allelic variants

Multi-allelic variants are rejected by default, and must be
filtered out:

```bash
vcztools view-bgen sample.vcz -o sample --max-alleles 2
```

### Ploidy

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

### Missingness

A sample is treated as missing for a variant if any of its diploid
alleles is `-1`, or if its haploid allele is `-1`.
Missing samples have the BGEN ploidy/missing byte's high bit
set and their probability bytes are zeroed.

(sec-bgen-subset-filtering)=
### Filtering over the sample subset

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

### Compression level

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

## Fixed-size encoding

Pass `--fixed-variant-size` to write a BGEN in which every variant block is
exactly the same number of bytes wide. The mapping from a byte offset to a
variant index is then O(1).

```bash
vcztools view-bgen sample.vcz -o sample --fixed-variant-size
```

The mode requires **uniform ploidy** across the store (every sample haploid,
or every sample diploid) and **no genotype compression**. Three options tune the
per-variant layout:

| Option | Default | Purpose |
| --- | --- | --- |
| `--total-string-length` | 64 | combined byte budget for the five BGEN string slots (varid, rsid, chrom, allele1, allele2) per variant |
| `--pad-byte` | `.` | single ASCII character filling the padding slot beyond its leading `.` |
| `--variant-id-field` | `rsid` | which slot carries the zarr `variant_id`; the other becomes the padding slot |

### Properties

Each variant block is exactly:

```
bytes_per_variant = 28 + total_string_length + zlib_stored_size(geno_size)
geno_size         = 10 + (ploidy + 1) * num_samples
```

so `geno_size = 10 + 2 * num_samples` for haploid stores and
`10 + 3 * num_samples` for diploid. The genotype block is stored at zlib
level 0 (no DEFLATE), making its length a deterministic function of the sample
count rather than the data.

Of the five string fields, four — chrom, allele1, allele2, and whichever of
varid/rsid carries `variant_id` — are written at their actual UTF-8 lengths;
the fifth (padding) slot absorbs the remaining budget as a leading `.` followed
by `--pad-byte` filler. If a variant's strings exceed `total_string_length - 1`,
the command fails with an error naming that variant. Use a larger
`--total-string-length` for stores with longer rsIDs or alleles.

## See also

- {ref}`The view-bgen flag reference<cmd-vcztools-view-bgen>`.
- [BGEN format specification](https://www.chg.ox.ac.uk/~gav/bgen_format/spec/latest.html).
- [bgenix index file format](https://enkre.net/cgi-bin/code/bgen/wiki/The%20bgenix%20index%20file%20format).
