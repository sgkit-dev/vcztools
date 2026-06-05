(sec-plink)=
# PLINK 1

This page describes the structure and semantics of the PLINK 1 binary fileset
produced by {ref}`vcztools view-plink<cmd-vcztools-view-plink>`. For a
command-line walkthrough, see the
{ref}`PLINK conversion<sec-plink-conversion>` page.

## Format details

### A1/A2 convention

vcztools follows **plink 2**: A1 = ALT, A2 = REF. This is the
modern convention expected by tools like REGENIE.

This is **not** plink 1.9's default *in-memory* ordering. plink 1.9
reorders A1/A2 on load to put the minor allele in A1 unless invoked
with `--keep-allele-order` (or `--real-ref-alleles`); the on-disk
bytes are unchanged, but plink 1.9's outputs (e.g. `--freq`,
`--assoc`) reflect the reordered labelling.

### Multi-allelic variants

Multi-allelic variants are rejected by default, mirroring `plink2
--make-bed` (which errors with "cannot contain multiallelic
variants"). These must be skipped using ``--max-alleles 2``:

```bash
vcztools view-plink sample.vcz -o sample --max-alleles 2
```
This drops every variant whose record lists more than two alleles. The
filter is record-driven — see {ref}`sec-plink-allele-list-driven`.

### Monomorphic variants

Single-allele (monomorphic) sites emit `A1 = "."` in the `.bim`
(plink 2's missing-allele encoding), and every genotype bit in the
`.bed` is set to MISSING.

(sec-plink-allele-list-driven)=
### Allele-list-driven semantics

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

### Chromosome-name normalisation

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
- **Diploid only.** `view-plink` raises if `call_genotype` ploidy
  is not 2.
- **Whitespace in sample IDs.** Rejected with a clear error
  message — the `.fam` format is whitespace-separated.

## See also

- {ref}`The view-plink flag reference<cmd-vcztools-view-plink>`.
- [PLINK 2 `.bim` format](https://www.cog-genomics.org/plink/2.0/formats#bim)
  and [`.bed` / `.fam`](https://www.cog-genomics.org/plink/2.0/formats)
  specifications.
- [PLINK 1.9 data-management notes](https://www.cog-genomics.org/plink/1.9/data)
  for the `--keep-allele-order` flag.
