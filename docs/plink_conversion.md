---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

(sec-plink-conversion)=
# PLINK conversion

{ref}`vcztools view-plink<cmd-vcztools-view-plink>` transcodes a VCZ store into a
PLINK 1 binary fileset (`.bed`/`.bim`/`.fam`). The format semantics — the
A1=ALT/A2=REF convention, missingness, and chromosome-name normalisation — are
described in the {ref}`PLINK 1 file format page<sec-plink>`; this page walks
through the command itself.

The examples run against the example dataset `data/sample.vcz.zip` (see
{ref}`sec-reading-vcz`).

## Basic conversion

`-o` gives the output stem. The dataset contains multi-allelic sites, which
PLINK cannot represent, so `-M 2` is required to skip them (mirroring `plink2
--vcf X --max-alleles 2 --make-bed`):

```{code-cell} ipython3
!vcztools view-plink data/sample.vcz.zip -M 2 -o sample
```

This writes three files:

```{code-cell} ipython3
!ls sample.bed sample.bim sample.fam
```

The `.bed` holds the packed genotypes (binary). The `.fam` lists one line per
sample and the `.bim` one line per variant:

```{code-cell} ipython3
!cat sample.fam
```

```{code-cell} ipython3
!cat sample.bim
```

## Subsetting by sample

Pass `-s` (or `-S FILE`) to convert only a subset of samples. Here we keep two
of the three:

```{code-cell} ipython3
!vcztools view-plink data/sample.vcz.zip -M 2 -s NA00001,NA00003 -o subset
```

The `.fam` now lists just those samples:

```{code-cell} ipython3
!cat subset.fam
```

Only the selected samples are read from the store, so pulling a handful of
samples out of a very wide cohort is cheap — the full cohort is never
materialised.

(sec-plink-subset-filtering)=
### Filtering over the sample subset

`-i`/`-e` expression filters that reference sample-derived INFO fields
(`AC`, `AN`, `AF`, `NS`) are evaluated **over the selected samples**. With
a `-s`/`-S` subset, those values are recomputed for that subset rather
than read from the file's stored (full-cohort) INFO. So
`view-plink -s cohort.txt -i 'AC>0'` keeps the variants that are
polymorphic *in your cohort*, dropping sites whose ALT alleles are carried
only by excluded samples — even where the source file's stored `INFO/AC`
is non-zero.

This is deliberately **different from
{ref}`vcztools view<cmd-vcztools-view>` and
{ref}`vcztools query<cmd-vcztools-query>`**, which follow bcftools and
evaluate `-i/-e` against the original
record's stored INFO (so the same `-i 'AC>0'` keeps a site on its stored
full-cohort count). The rationale: a PLINK fileset is the input to an
association analysis run on the exported cohort, so filters should reflect
that cohort, not the source dataset. It is also the cheaper path on large
datasets — only the selected samples are read.

Note this applies to *expression* filters. The allele-based filters
(`-m`/`-M`/`-v`/`-V`) remain record-level.

## Sidecars

The `.bim` and `.fam` sidecars are written by default; pass `--no-bim` or
`--no-fam` to suppress them.

## See also

- The {ref}`PLINK 1 file format page<sec-plink>` — format details.
- {ref}`sec-format-conversion` — the equivalent Python writer API.
- {ref}`The view-plink flag reference<cmd-vcztools-view-plink>`.
