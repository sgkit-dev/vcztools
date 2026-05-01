(sec-plink)=
# PLINK 1 binary output

The `vcztools view-bed` command writes a PLINK 1 binary fileset
(`.bed`/`.bim`/`.fam`) from a VCZ store. The on-disk layout is the
one PLINK 1, 1.9 and 2 all read; the `.bed` payload is byte-identical
to `plink2 --vcf X --make-bed` for biallelic variants.

```bash
vcztools view-bed sample.vcz --out sample
# produces sample.bed, sample.bim, sample.fam
```

Sample, region and filter selection mirrors `vcztools view`
(`-s`/`-S`/`-r`/`-R`/`-t`/`-T`/`-i`/`-e`); the per-flag reference is
in {ref}`sec-cli-ref`.

## A1/A2 convention

vcztools follows **plink 2**: A1 = ALT, A2 = REF. This is the
modern convention expected by tools like REGENIE.

This is **not** plink 1.9's default *in-memory* ordering. plink 1.9
reorders A1/A2 on load to put the minor allele in A1 unless invoked
with `--keep-allele-order` (or `--real-ref-alleles`); the on-disk
bytes are unchanged, but plink 1.9's outputs (e.g. `--freq`,
`--assoc`) reflect the reordered labelling.

## Reading `view-bed` output with downstream tools

| Tool | Works as-is? | Notes |
| --- | --- | --- |
| plink 2 | yes | reads A2 as REF natively; no flags needed |
| plink 1.9 | with `--keep-allele-order` | otherwise silently relabels A1/A2 on load and any frequency-derived statistic flips sign relative to a plink 2 run on the same file |
| REGENIE | yes | default expects A1 = non-reference (`--ref-first` opts the other way) |
| BOLT-LMM | yes | ALLELE1 is the effect allele; the manual's "usually the minor allele" remark is descriptive, not normative |
| GCTA | yes | per-variant standardisation is invariant to allele labelling |
| KING | yes | allele-frequency-free by construction |
| flashpca | yes | PCA standardisation is allele-symmetric; cross-cohort projection is more stable under REF/ALT than minor-allele |
| ADMIXTURE | yes | the likelihood is symmetric in allele labels |

## Multi-allelic variants

Multi-allelic variants are rejected by default, mirroring `plink2
--make-bed` (which errors with "cannot contain multiallelic
variants"). Two ways to handle them:

- **Skip** with `--max-alleles 2`:

  ```bash
  vcztools view-bed sample.vcz --out sample --max-alleles 2
  ```

  Drops every variant whose record lists more than two alleles. The
  filter is record-driven — see {ref}`sec-plink-allele-list-driven`.

- **Split** before conversion with ``bcftools norm``.
  Each ALT allele becomes its own biallelic record.

## Monomorphic variants

Single-allele (monomorphic) sites emit `A1 = "."` in the `.bim`
(plink 2's missing-allele encoding), and every genotype bit in the
`.bed` is set to MISSING.

(sec-plink-allele-list-driven)=
## Allele-list-driven semantics

A1/A2 labelling and the `--max-alleles` filter come from the variant
record's allele list, not from genotypes observed in the kept-sample
subset. Two consequences:

- A site that's biallelic in the full dataset but has no ALT
  carriers in a `-s`/`-S` subset still emits A1=ALT, A2=REF in the
  `.bim`, with HOM-REF / MISSING bits in the `.bed` for the kept
  samples. The `.bim` does *not* collapse to A1=".".
- A site with three or more alleles in the record is dropped by
  `--max-alleles 2` even if only two alleles survive in the subset.

This matches `plink2 --make-bed --keep`, which applies record-level
filters before sample projection.

## Chromosome-name normalisation

vcztools writes `.bim` chromosome names to match plink 2's
`--make-bed` normalisation:

- The `chr` prefix is stripped from the human standard chromosomes
  (1-22, X, Y, MT). `chr1` → `1`, `chrX` → `X`.
- `chrM` is rewritten to `MT`.
- Non-standard contigs (e.g. `chrUnknown`) pass through unchanged.
  Under plink 2 these require `--allow-extra-chr`; vcztools does
  not enforce that flag.

## Limitations and known divergences from plink 2

- **chrX without sex info.** plink 2 errors on chrX rows under
  `--make-bed` unless given `--update-sex` or `--split-par`.
  vcztools writes the rows pass-through because it doesn't track
  sex. Use `--exclude 'CHROM=="X"'` (or `-e`) if you need
  plink-2-equivalent behaviour.
- **Diploid only.** `view-bed` raises if `call_genotype` ploidy
  is not 2.
- **Whitespace in sample IDs.** Rejected with a clear error
  message — the `.fam` format is whitespace-separated.
- **Family ID (FID).** Always written as `0`, matching `plink2
  --vcf X --make-bed` defaults. plink 2's `--double-id` /
  `--id-delim` are not mirrored.

## See also

- {ref}`sec-cli-ref` for the full `view-bed` flag reference.
- [PLINK 2 `.bim` format](https://www.cog-genomics.org/plink/2.0/formats#bim)
  and [`.bed` / `.fam`](https://www.cog-genomics.org/plink/2.0/formats)
  specifications.
- [PLINK 1.9 data-management notes](https://www.cog-genomics.org/plink/1.9/data)
  for the `--keep-allele-order` flag.
