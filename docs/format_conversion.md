---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

(sec-format-conversion)=
# Format conversion

`vcztools` converts a configured {class}`vcztools.VczReader` to three output
formats from Python, so any sample or variant selection on the reader (see
{ref}`sec-reading-vcz`) flows through to the output. Each writer is documented
in full in the {ref}`sec-python-api`; the linked file-format pages describe the
output semantics (allele conventions, missingness, multi-allelic handling).

| Function | Output format |
|---|---|
| {func}`vcztools.write_vcf` | {ref}`VCF<sec-vcf>` |
| {func}`vcztools.write_plink` | {ref}`PLINK 1 binary fileset<sec-plink>` |
| {func}`vcztools.write_bgen` | {ref}`Oxford BGEN<sec-bgen>` |

`write_vcf` streams records and filters on the fly, so a configured variant
filter and a sample subset are honoured as-is. `write_plink` and `write_bgen`
are fixed-width per variant: they are biallelic-only and cannot consume a
still-configured filter, which must first be resolved into a fixed selection
with {meth}`~vcztools.VczReader.materialise_variant_filter`.

The examples below run against the example dataset `data/sample.vcz.zip` (9
variants across 3 samples; see {ref}`sec-reading-vcz`).

## Writing VCF

A sample subset and a {class}`vcztools.BcftoolsFilter` can be set on the reader
and passed straight to `write_vcf`:

```{code-cell} ipython3
import vcztools

root = vcztools.open_zarr("data/sample.vcz.zip")
with vcztools.VczReader(root) as reader:
    reader.set_samples(["NA00001", "NA00003"])
    reader.set_variant_filter(vcztools.BcftoolsFilter(reader, include="QUAL>10"))
    vcztools.write_vcf(reader, "subset.vcf")
```

The output reflects both the sample subset and the filter. The `#CHROM` line
lists only the two selected samples, and every record that survives has
`QUAL > 10` (4 of the 9 variants):

```{code-cell} ipython3
records = []
for line in open("subset.vcf"):
    if line.startswith("#CHROM"):
        print("samples:", line.split()[9:])
    elif not line.startswith("#"):
        records.append(line.split())

quals = [float(record[5]) for record in records]
print("records written:", len(records))
print("QUAL values:", quals)
print("all QUAL > 10:", all(qual > 10 for qual in quals))
```

## Writing PLINK

PLINK is biallelic-only, so multi-allelic sites must be dropped, and the
fixed-width writers cannot consume a still-configured filter — it must first be
resolved with {meth}`~vcztools.VczReader.materialise_variant_filter`. Filtering
on `N_ALT == 1` (biallelic) `&` `AC > 0` keeps the sites that are biallelic and
polymorphic *in the chosen cohort*; `AC` is recomputed over the selected samples
(see {ref}`sec-plink-subset-filtering`). Use the variant-scope `&` rather than
the sample-scope `&&`, so the combined filter stays variant-scope:

```{code-cell} ipython3
root = vcztools.open_zarr("data/sample.vcz.zip")
with vcztools.VczReader(root) as reader:
    reader.set_samples(["NA00001", "NA00003"])
    reader.set_variant_filter(
        vcztools.BcftoolsFilter(reader, include="N_ALT == 1 & AC > 0")
    )
    reader.materialise_variant_filter()
    vcztools.write_plink(reader, "cohort")
```

The `.fam` lists one line per selected sample:

```{code-cell} ipython3
!cat cohort.fam
```

The `.bim` lists one line per retained variant. Only the biallelic sites that
are polymorphic in the cohort remain (`CHROM ID GeneticDistance POS ALT REF`):

```{code-cell} ipython3
!cat cohort.bim
```

The packed genotypes are in the binary `.bed`, which opens with the PLINK magic
bytes `6c 1b 01`:

```{code-cell} ipython3
header = open("cohort.bed", "rb").read(3)
print("magic bytes:", header.hex(" "))
```
