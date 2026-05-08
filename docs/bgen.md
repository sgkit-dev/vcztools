(sec-bgen)=
# BGEN output

The `vcztools view-bgen` command writes an Oxford BGEN fileset
(`.bgen`/`.sample`/`.bgen.bgi`) from a VCZ store. Output profile:
**layout 2, zlib-compressed, 8 bits per probability, biallelic,
diploid, embedded sample IDs**. This is the consumer
lowest-common-denominator: REGENIE, SAIGE, BOLT-LMM, BGENIE, qctool,
and PLINK 2 all accept it without further conversion.

```bash
vcztools view-bgen sample.vcz --out sample
# produces sample.bgen, sample.sample, sample.bgen.bgi
```

Sample, region and filter selection mirrors `vcztools view`
(`-s`/`-S`/`-r`/`-R`/`-t`/`-T`/`-i`/`-e`/`-v`/`-V`/`-m`/`-M`); the
per-flag reference is in {ref}`sec-cli-ref`.

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

`view-bgen` writes three files for every run:

| File | Format | Purpose |
| --- | --- | --- |
| `<out>.bgen` | binary | the BGEN payload (header + variant blocks) |
| `<out>.sample` | text | Oxford `.sample` (sample IDs); required by SAIGE, accepted by REGENIE |
| `<out>.bgen.bgi` | SQLite | bgenix index; required by SAIGE Step 2; strongly recommended for REGENIE |

The `.bgen` header also embeds sample IDs, so consumers that read sample
names from the BGEN itself (qctool, BGENIE, PLINK 2) work without the
`.sample` sidecar. SAIGE Step 2 still requires the `.sample` file.

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
  vcztools view-bgen sample.vcz --out sample --max-alleles 2
  ```

- **Split** before conversion with ``bcftools norm -m-``. Each ALT
  allele becomes its own biallelic record.

## Missingness

A sample is treated as missing for a variant if any of its alleles is
negative (the VCZ conventions for missing alleles, including `-1` and
the `-2` haploid-padding sentinel). Missing samples have the BGEN
ploidy/missing byte's high bit set; their probability bytes are
zeroed (the BGEN spec says these bytes should be ignored when the
missing flag is set, but zeroing keeps the output deterministic).

## Limitations

- **Layout 2, 8-bit, zlib only.** Higher precision (16/32-bit) and
  alternative compressions (zstd, none) are not exposed. Adding them
  is straightforward when a downstream tool requires them.
- **Diploid only.** `view-bgen` raises if `call_genotype.shape[2] != 2`.
- **Hard calls only.** Genotype probabilities (`call_GP`) are not
  read; the encoder emits 1.0/0.0 from `call_genotype` only.
- **Whitespace in sample IDs.** Rejected with a clear error message —
  the `.sample` format is whitespace-separated.

## See also

- {ref}`sec-cli-ref` for the full `view-bgen` flag reference.
- [BGEN format specification](https://www.chg.ox.ac.uk/~gav/bgen_format/spec/latest.html).
- [bgenix index file format](https://enkre.net/cgi-bin/code/bgen/wiki/The%20bgenix%20index%20file%20format).
