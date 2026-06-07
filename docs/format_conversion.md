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

## Writing VCF

A sample subset and a {class}`vcztools.BcftoolsFilter` can be set on the reader
and passed straight to `write_vcf`:

```python
import vcztools

root = vcztools.open_zarr("sample.vcz.zip")
with vcztools.VczReader(root) as reader:
    reader.set_samples(["NA00001", "NA00003"])
    reader.set_variant_filter(vcztools.BcftoolsFilter(reader, include="QUAL>10"))
    vcztools.write_vcf(reader, "subset.vcf")
```

## Writing PLINK

PLINK is biallelic-only, so multi-allelic sites must be dropped, and the filter
must be materialised before writing. Filtering on `N_ALT == 1` (biallelic) `&`
`AC > 0` keeps the sites that are biallelic and polymorphic *in the chosen
cohort* — `AC` is recomputed over the selected samples (see
{ref}`sec-plink-subset-filtering`):

```python
root = vcztools.open_zarr("sample.vcz.zip")
with vcztools.VczReader(root) as reader:
    reader.set_samples(["NA00001", "NA00003"])
    reader.set_variant_filter(
        vcztools.BcftoolsFilter(reader, include="N_ALT == 1 & AC > 0")
    )
    reader.materialise_variant_filter()
    vcztools.write_plink(reader, "cohort")
```

Use the variant-scope `&` rather than the sample-scope `&&` so the combined
filter stays variant-scope; `materialise_variant_filter` raises `ValueError` on
a sample-scope filter.
