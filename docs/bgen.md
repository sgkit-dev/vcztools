(sec-bgen)=
# BGEN

This page describes the BGEN output format produced by
{ref}`vcztools view-bgen<cmd-vcztools-view-bgen>`. For a command-line
walkthrough, see the {ref}`BGEN conversion<sec-bgen-conversion>` page.

## Format details

### Sample-ID embedding

By default the BGEN header carries sample IDs (the `SAMPLE_IDS_PRESENT`
flag is set), so the `.bgen` is self-describing. Pass `--no-header-samples`
to clear the flag and omit the sample-ID block.

### Hard calls and probabilities

BGEN's native data type is genotype probabilities.
`view-bgen` uses `call_GP` (unphased, colexicographic genotype order) or
`call_HP` (phased, flattened haplotype-major allele probabilities) when
present, even if `call_genotype` also exists. It never infers hard calls
from probabilities or dosages. Without GP/HP, hard calls in `call_genotype`
are encoded as 1.0 on the called state and 0.0 elsewhere.

For probability input, provide `call_PLOIDY[variants,samples]`, or a
`call_genotype` array whose trailing `-2` padding identifies ploidy.
`call_BGEN_MISSING[variants,samples]` preserves the missing flag independently
of ploidy, including zero-ploidy calls. Otherwise, active missing probability
sentinels identify missing calls. Probability-only stores without ploidy
information are rejected rather than guessing.

GP's active width is `C(P+K-1,P)`; HP's is `P*K`. Extra entries must use
VCZ floating-point fill sentinels, and missing active entries must use missing
sentinels, not fill. `variant_bgen_phased[variants]` chooses GP or HP per
variant and is required when both fields exist. Inactive probability fields
must contain one missing sentinel followed by fill. `variant_id` supplies
the rsID; `variant_bgen_id`, when present, preserves BGEN's separate variant ID.
Allele ordering and one-based positions are preserved.

The default is 16 bits for GP/HP and 8 bits for hard calls.
`--bits-per-probability 1..32` selects precision. Each simplex is validated
(finite values in `[0,1]`, sum within `1e-6` of 1), normalized, and rounded
using largest remainders so reconstructed probabilities sum to 1.
Each component differs from the **normalized** source by at most
`1/(2^bits-1)` plus floating-point roundoff. For HP this applies separately
to each haplotype; alternate-allele dosage error is at most `P/(2^bits-1)`.
For GP a conservative dosage-error bound is
`P*C(P+K-1,P)/(2^bits-1)`. Normalization drift is additional when comparing
against unnormalized source values (notably float32).
Hard-call probabilities are represented exactly.

### Native encoding dependency (development)

The general GP/HP, multiallelic and polyploid encoder delegates numerical
encoding to the public `cbgen.encode_layout2` and
`cbgen.encode_layout2_hardcalls` interfaces from the **bgen-limix 4.15
development build**. This is a staged cross-repository change: those
interfaces are **not present in released bgen-limix 4.14**, and installing
`vcztools[bgen]` from the public package index alone does not yet enable them.
Install the matching development wheel before using general export.
Missing capabilities raise `ImportError` before creating the payload or
any sidecar; there is no Python numerical fallback.

The existing biallelic haploid/diploid 8-bit hard-call paths remain usable
without the new interfaces. Variable-size hard-call export uses the public
native encoder when available; fixed-stride `BgenEncoder` retains its fused
vcztools C kernel. Released bgen-limix 4.14 remains sufficient for reading
and `.metafile` creation. Dependency requirements must be raised when the
new upstream encoder is published; no local wheel paths are recorded in
the dependency manifest or lockfile.

### Phasing

For probability/BGEN metadata stores, `variant_bgen_phased` is authoritative.
Without it, GP-only is unphased and HP-only is phased. Phased HP with
`--unphased` is rejected rather than silently discarding phase.

For legacy hard-call stores with a `call_genotype_phased` field, each variant is
emitted as phased iff every sample is phased for that variant. (BGEN
has a single phase flag per variant; mixed-phase variants degrade
to unphased and a warning is logged.) Stores without
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
| `<stem>.bgen.metafile` | native bgen-limix | off | omit `--metafile` | native variant index |

`--metafile` additionally creates the native bgen-limix index.
It requires `vcztools[bgen]`
(`bgen-limix>=4.14,<5`, imported as `cbgen`) and `-o`.
The legacy distribution named `cbgen` is incompatible: do not install both.
Native index creation uses the public `BgenFile.create_metafile` API after
closing the payload. It publishes only a complete index and refuses to
overwrite an existing one. Native index generation requires at least one
variant. Export also refuses to overwrite a BGEN payload
with an existing canonical `.metafile`, to avoid leaving a stale index.
Remove or relocate old files deliberately before regenerating a fileset.

The `.sample` format is `ID_1 ID_2 missing`, then `0 0 0`, followed by
`sample_id sample_id 0` per selected sample, in export order. It carries IDs
only, not phenotypes or per-sample missingness estimates.

The `.bgen` header also embeds sample IDs, so consumers that read sample
names from the BGEN itself (qctool, BGENIE, PLINK 2) work without the
`.sample` sidecar. Pass
`--no-header-samples` to drop the in-header IDs (e.g. for samples whose
identifiers must not appear in the binary).

Streaming mode (no `-o`) writes only the `.bgen` payload to stdout;
the suppression flags have no effect. `--metafile` instead errors without `-o`.

### Multi-allelic variants

Variable-size export supports multi-allelic variants. To restrict output
for biallelic-only consumers:

```bash
vcztools view-bgen sample.vcz -o sample --max-alleles 2
```

### Ploidy

Probability/general-ploidy export supports ploidy 0..63 and mixed
ploidy within a variant. GP at ploidy zero contains `[1]` for a nonmissing
call; HP at ploidy zero has an empty active vector. Missing zero-ploidy
calls remain distinguishable through `call_BGEN_MISSING`.

The legacy fast hard-call path supports both haploid and diploid input, plus mixed
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

- **Layout 2, zlib only.** Alternative compressions and layout 1 are
  not exposed.
- **Not universally lossless.** Probabilities are quantized. Derived DS
  is recomputed by importers; DS-only input cannot determine probabilities
  and is rejected. Unrelated VCF INFO/FORMAT fields, filters, phenotypes,
  and arbitrary VCZ attributes are not represented by BGEN.
- **General encoder resources.** Reads remain chunked and sample-subset
  aware. Numerical encoding runs in bgen-limix C with the GIL released;
  `encode_threads` also controls ordered, bounded batches on this path.
  Batches target at most 8 MiB of raw genotype output (a larger single
  variant forms its own batch); individual genotype blocks above 256 MiB
  are rejected before native allocation. Hard-call unphased state spaces
  exceeding one million stored states per call are rejected explicitly.
  Native encoding does not eliminate combinatorial genotype-state growth.
- **Legacy monomorphic profile.** Hard-call stores without BGEN metadata
  retain the historical `REF,.` representation; probability/BGEN-metadata
  stores preserve true single-allele records.
- **Whitespace in sample IDs.** Rejected with a clear error message —
  the `.sample` format is whitespace-separated. Sidecar IDs must also
  be nonempty and unique, and cannot be the reserved missing identifiers
  `NA` or `.` (case-sensitive). Literal `0`, `na`, and `Na` are valid.
  This sidecar restriction does not apply to IDs embedded in BGEN;
  omit `.sample` with `--no-sample-file` to retain an embedded `NA` ID.

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
