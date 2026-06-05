---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

(sec-bcftools-emulation)=
# bcftools emulation

`vcztools` is a drop-in replacement for a subset of
[bcftools](https://samtools.github.io/bcftools/bcftools.html): the
{ref}`view<cmd-vcztools-view>`, {ref}`query<cmd-vcztools-query>` and
{ref}`index<cmd-vcztools-index>` commands take the same flags and produce the
same output, with a VCZ store in place of a VCF/BCF file.

Every command below is run against the example dataset
`data/sample.vcz.zip` (9 variants across 3 samples; see {ref}`sec-reading-vcz`
for its contents). The output shown is generated when the documentation is
built. For the full flag list, see {ref}`sec-cli-ref`.

## Viewing records

{ref}`vcztools view<cmd-vcztools-view>` writes VCF. With no options it emits the
complete file — header plus every record:

```{code-cell} ipython3
!vcztools view data/sample.vcz.zip
```

`-H` suppresses the header, which keeps the remaining examples compact:

```{code-cell} ipython3
!vcztools view -H data/sample.vcz.zip
```

`-s` selects a subset of samples. INFO fields are recomputed for the subset by
default:

```{code-cell} ipython3
!vcztools view -H -s NA00001,NA00003 data/sample.vcz.zip
```

## Per-record queries

{ref}`vcztools query<cmd-vcztools-query>` extracts fields in a custom format.
`-l` lists the samples:

```{code-cell} ipython3
!vcztools query -l data/sample.vcz.zip
```

`-f` takes a format string; `%`-tags pull out site fields and a `[…]` block
loops over samples:

```{code-cell} ipython3
!vcztools query -f '%CHROM\t%POS\t%REF\t%ALT\t%INFO/DP\n' data/sample.vcz.zip
```

```{code-cell} ipython3
!vcztools query -f '%CHROM\t%POS\t%REF\t%ALT[\t%GT]\n' data/sample.vcz.zip
```

## Counting records

{ref}`vcztools index<cmd-vcztools-index>` does **not** build `.csi`/`.tbi` index
files; it only answers the bcftools `-n`/`-s` queries. `-n` prints the total
number of records:

```{code-cell} ipython3
!vcztools index -n data/sample.vcz.zip
```

`-s` prints per-contig statistics:

```{code-cell} ipython3
!vcztools index -s data/sample.vcz.zip
```

## Region and target selection

`-r`/`--regions` restricts output to one or more genomic intervals:

```{code-cell} ipython3
!vcztools view -H -r 20:1000000-1300000 data/sample.vcz.zip
```

`-t`/`--targets` selects the same way but with bcftools' streaming semantics
(the two differ on how records straddling a region boundary are treated):

```{code-cell} ipython3
!vcztools view -H -t 20:1000000-1300000 data/sample.vcz.zip
```

`vcztools` resolves regions and targets directly from the Zarr arrays, so no
external index is required. The `-R`/`--regions-file` and `-T`/`--targets-file`
variants read the same interval specifications from a file.

## Recomputing INFO tags

`--fill-tags` (re)computes selected INFO tags from the genotypes and emits them,
replacing any stored value — the equivalent of piping bcftools output through
`+fill-tags`. The supported tags are `AC`, `AN`, `AF` and `NS`:

```{code-cell} ipython3
!vcztools view -H --fill-tags AC,AN,AF,NS data/sample.vcz.zip
```

The tags are filled even for sites whose source INFO was empty. With a sample
subset the counts are recomputed over the selected samples:

```{code-cell} ipython3
!vcztools view -H -s NA00001,NA00003 --fill-tags AC,AN data/sample.vcz.zip
```

`--fill-tags` is mutually exclusive with `-I/--no-update` and
`-G/--drop-genotypes`.

## Filtering

`-i`/`--include` and `-e`/`--exclude` take bcftools filter expressions over the
site fields. Keep the high-quality sites:

```{code-cell} ipython3
!vcztools view -H -i 'QUAL>10' data/sample.vcz.zip
```

Drop a particular FILTER value:

```{code-cell} ipython3
!vcztools view -H -e 'FILTER="q10"' data/sample.vcz.zip
```

Expressions can reference INFO fields:

```{code-cell} ipython3
!vcztools view -H -i 'INFO/DP>10' data/sample.vcz.zip
```

`-v`/`--types` (and `-V`/`--exclude-types`) filter by variant type:

```{code-cell} ipython3
!vcztools view -H -v snps data/sample.vcz.zip
```

`-m`/`--min-alleles` and `-M`/`--max-alleles` filter on the number of alleles;
`-m2 -M2` keeps only biallelic sites:

```{code-cell} ipython3
!vcztools view -H -m2 -M2 data/sample.vcz.zip
```

## See also

- {ref}`The full CLI flag reference<sec-cli-ref>`.
- [The bcftools manual](https://samtools.github.io/bcftools/bcftools.html).
