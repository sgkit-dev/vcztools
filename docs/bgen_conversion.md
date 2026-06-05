---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

(sec-bgen-conversion)=
# BGEN conversion

{ref}`vcztools view-bgen<cmd-vcztools-view-bgen>` transcodes a VCZ store into the
Oxford BGEN format. Without `-o` it streams the `.bgen` payload to stdout; with
`-o STEM` it writes `STEM.bgen` plus the `STEM.bgen.bgi` (bgenix index) and
`STEM.sample` (Oxford text) sidecars. The format details — layout, ploidy,
phasing and compression — are in the {ref}`BGEN file format page<sec-bgen>`; this
page walks through the
command.

The examples run against the example dataset `data/sample.vcz.zip` (see
{ref}`sec-reading-vcz`).

## Basic conversion

As with PLINK, multi-allelic sites are rejected, so `-M 2` skips them. The
sample data is mixed-phase; `--unphased` emits every variant unphased (see
the {ref}`BGEN file format page<sec-bgen>` for the phasing rules):

```{code-cell} ipython3
!vcztools view-bgen data/sample.vcz.zip -M 2 --unphased -o sample
```

This writes the payload and its two sidecars:

```{code-cell} ipython3
!ls sample.bgen sample.bgen.bgi sample.sample
```

The `.bgen` is binary. The Oxford `.sample` sidecar lists the samples:

```{code-cell} ipython3
!cat sample.sample
```

## Subsetting by sample

Pass `-s` (or `-S FILE`) to convert only a subset of samples:

```{code-cell} ipython3
!vcztools view-bgen data/sample.vcz.zip -M 2 --unphased -s NA00001,NA00003 -o subset
```

The `.sample` sidecar now lists just those samples:

```{code-cell} ipython3
!cat subset.sample
```

Only the selected samples are read from the store, so subsetting a handful of
samples out of a very wide cohort is cheap — the full cohort is never
materialised.

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

## Sidecars and compression

Streaming mode (no `-o`) writes only the `.bgen` payload. With `-o`, suppress
either sidecar with `--no-bgi` or `--no-sample-file`. The genotype blocks are
zlib-compressed; `--compression-level` tunes the level (default `1`).

## See also

- The {ref}`BGEN file format page<sec-bgen>` — format details.
- {ref}`sec-format-conversion` — the equivalent Python writer API.
- {ref}`The view-bgen flag reference<cmd-vcztools-view-bgen>`.
