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
described in {ref}`sec-plink`; this page walks through the command itself.

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
materialised. Filter expressions also follow the subset: `-i`/`-e` conditions on
`AC`/`AN`/`AF`/`NS` are recomputed over the selected samples rather than read
from the file's stored values, as described in
{ref}`sec-plink-subset-filtering`.

## Sidecars

The `.bim` and `.fam` sidecars are written by default; pass `--no-bim` or
`--no-fam` to suppress them.

## See also

- {ref}`sec-plink` — the PLINK 1 format details.
- {ref}`sec-format-conversion` — the equivalent Python writer API.
- {ref}`The view-plink flag reference<cmd-vcztools-view-plink>`.
