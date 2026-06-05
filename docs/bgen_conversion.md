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
phasing and compression — are in {ref}`sec-bgen`; this page walks through the
command.

The examples run against the example dataset `data/sample.vcz.zip` (see
{ref}`sec-reading-vcz`).

## Basic conversion

As with PLINK, multi-allelic sites are rejected, so `-M 2` skips them. The
sample data is mixed-phase; `--unphased` emits every variant unphased (see
{ref}`sec-bgen` for the phasing rules):

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
materialised. Filter expressions follow the subset too: `-i`/`-e` conditions on
`AC`/`AN`/`AF`/`NS` are recomputed over the selected samples, as described in
{ref}`sec-bgen-subset-filtering`.

## Sidecars and compression

Streaming mode (no `-o`) writes only the `.bgen` payload. With `-o`, suppress
either sidecar with `--no-bgi` or `--no-sample-file`. The genotype blocks are
zlib-compressed; `--compression-level` tunes the level (default `1`).

## See also

- {ref}`sec-bgen` — the BGEN format details.
- {ref}`sec-format-conversion` — the equivalent Python writer API.
- {ref}`The view-bgen flag reference<cmd-vcztools-view-bgen>`.
