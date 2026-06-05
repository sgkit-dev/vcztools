---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

(sec-reading-vcz)=
# Reading VCZ

This page is a worked introduction to reading VCF Zarr (VCZ) datasets from
Python. Every code cell below runs against a small example dataset,
`data/sample.vcz.zip` (9 variants across 3 samples, plus a null sample at index
1 used to illustrate masking; see {ref}`sec-null-samples`).
For exact signatures of everything used here, see the {ref}`sec-python-api`.

## Opening a dataset

{func}`vcztools.open_zarr` is the recommended entry point. It returns a
{class}`zarr.Group` and dispatches to one of four storage backends; see
{ref}`sec-storage-backends` for the full description of each. A local directory
or `.zip` archive needs no explicit backend:

```{code-cell} ipython3
import vcztools

root = vcztools.open_zarr("data/sample.vcz.zip")
root
```

Remote URLs require selecting a backend, and `storage_options` is forwarded to
it (the fsspec filesystem, `obstore.store.from_url`, or the chosen Icechunk
storage constructor):

```python
root = vcztools.open_zarr(
    "s3://bucket/sample.vcz",
    backend_storage="fsspec",
    storage_options={"anon": True},
)
```

An already-built {class}`zarr.Group` or {class}`zarr.abc.store.Store` is
accepted and passed through unchanged, which is useful when you've configured a
store yourself:

```python
import zarr

store = zarr.storage.MemoryStore()
# ... populate ``store`` ...
root = vcztools.open_zarr(store)
```

## Creating a reader

{class}`vcztools.VczReader` wraps an opened group and provides metadata and
variant iteration. Use it as a context manager so resources are released
deterministically:

```{code-cell} ipython3
with vcztools.VczReader(root) as reader:
    print(reader.num_variants, "variants")
    print(reader.num_samples, "samples")
```

Displaying a reader in a notebook renders a summary of the store and the
reader's current state — its dimensions, null-sample count, sample and variant
selection, any configured filter, and the available fields:

```{code-cell} ipython3
reader = vcztools.VczReader(root)
reader
```

The reader starts with every sample and every variant selected. The remaining
sections each open a fresh reader so their selections stay independent.

## Inspecting metadata

```{code-cell} ipython3
reader = vcztools.VczReader(root)

print("sample_ids:", reader.sample_ids)
print("contig_ids:", reader.contig_ids)
```

`sample_ids` lists only the real samples, so it is shorter than `num_samples`
(printed above), which counts every slot in the store including null samples;
see {ref}`sec-null-samples`.

`field_names` lists the arrays stored in the dataset, while
`virtual_field_names` lists fields computed on demand (allele counts and
frequencies, missingness, and so on) that are never emitted unless you request
them by name:

```{code-cell} ipython3
print("stored fields:", sorted(reader.field_names))
print()
print("virtual fields:", sorted(reader.virtual_field_names))
```

## Field metadata

{meth}`~vcztools.VczReader.get_field_info` returns a {class}`vcztools.FieldInfo`
snapshot describing a single field's dtype, shape, dimensions and attributes:

```{code-cell} ipython3
info = reader.get_field_info("call_DP")
info
```

## Iterating variants

{meth}`~vcztools.VczReader.variants` yields one dict per variant row. Pass
`fields` to restrict what is read; here we also request the virtual
`variant_AF` (allele frequency) field. We take the first three rows:

```{code-cell} ipython3
import itertools

rows = itertools.islice(
    reader.variants(fields=["variant_position", "variant_allele", "variant_AF"]),
    3,
)
for row in rows:
    print(row["variant_position"], row["variant_allele"], row["variant_AF"])
```

{meth}`~vcztools.VczReader.variant_chunks` yields one dict per variant *chunk*,
with each field as a NumPy array whose first axis is the variants in that chunk.
This is the efficient path for bulk processing:

```{code-cell} ipython3
for chunk in reader.variant_chunks(fields=["variant_position", "call_genotype"]):
    positions = chunk["variant_position"]
    genotypes = chunk["call_genotype"]
    print(f"chunk of {len(positions)} variants, "
          f"call_genotype shape {genotypes.shape}")
```

(sec-missing-fill)=
## Missing and fill values

The arrays you get back are *raw*: they still contain the sentinel values VCZ
uses to encode two distinct ideas, and you must interpret them yourself.

- **Missing** — a value that is absent ("no data"). Detect it with
  {func}`vcztools.is_missing`.
- **Fill** (end-of-vector) — padding that makes a *ragged* field rectangular. A
  VCF INFO or FORMAT field can hold a variable number of values per variant or
  sample — for example one `AF` per ALT allele, so a biallelic site has one
  value and a triallelic site has two. VCZ stores these in a fixed-width array
  and pads the unused tail of each row with a fill sentinel. Detect it with
  {func}`vcztools.is_fill`, or drop the trailing fill from a single 1-D vector
  with {func}`vcztools.trim_fill`.

The sentinels differ by dtype (`-1`/`-2` for integers, distinct
not-a-number values for floats, `"."`/`""` for strings), so printing a raw
array often can't tell the two apart by eye — the helpers can. The VCF and
`query` output paths already trim fill and render missing as `.`; these helpers
let you do the same when working with the arrays directly.

`variant_AF` is ragged over ALT alleles. Trimming the fill and then marking the
missing entries recovers the true per-variant vector:

```{code-cell} ipython3
reader = vcztools.VczReader(root)

fields = ["variant_position", "variant_allele", "variant_AF"]
for row in reader.variants(fields=fields):
    af = row["variant_AF"]
    trimmed = vcztools.trim_fill(af)
    clean = [
        "." if missing else round(float(value), 3)
        for value, missing in zip(trimmed, vcztools.is_missing(trimmed))
    ]
    alleles = [a for a in row["variant_allele"] if a != ""]
    print(f"POS {row['variant_position']:>7}  {'/'.join(alleles):8}  "
          f"raw {af}  ->  AF {clean}")
```

Multiallelic sites (e.g. `A/G/T`) keep both allele frequencies, biallelic sites
have their single value followed by one trimmed-away fill, and sites where `AF`
is absent come back as missing (`.`).

The same two sentinels appear in genotypes. Missing marks a no-call (`.` in
VCF), while fill pads a call shorter than the array's ploidy — here one haploid
call in a diploid field:

```{code-cell} ipython3
reader = vcztools.VczReader(root)

missing_cells = 0
fill_cells = 0
fill_example = None
for row in reader.variants(fields=["variant_position", "call_genotype"]):
    genotype = row["call_genotype"]
    missing_cells += int(vcztools.is_missing(genotype).sum())
    fill = vcztools.is_fill(genotype)
    fill_cells += int(fill.sum())
    if fill.any() and fill_example is None:
        fill_example = (row["variant_position"], genotype)

print("missing (no-call) cells:", missing_cells)
print("fill (ploidy-padding) cells:", fill_cells)
position, genotype = fill_example
print(f"POS {position} has a haploid call padded with fill (-2):")
print(genotype)
```

## Selecting samples and variants

Call {meth}`~vcztools.VczReader.set_samples` with sample IDs *before* iterating
to restrict the sample selection. The output follows the order you request:

```{code-cell} ipython3
reader = vcztools.VczReader(root)
reader.set_samples(["NA00003", "NA00001"])
print("selected samples:", reader.sample_ids)
```

Pass `complement=True` to select every sample *except* those named, or
`ignore_missing_samples=True` to warn-and-drop unknown names instead of raising:

```{code-cell} ipython3
reader = vcztools.VczReader(root)
reader.set_samples(["NA00002"], complement=True)
print("selected samples:", reader.sample_ids)
```

{meth}`~vcztools.VczReader.set_variants` restricts the variant selection with a
sorted array of global variant indexes:

```{code-cell} ipython3
import numpy as np

reader = vcztools.VczReader(root)
reader.set_variants(np.array([0, 1, 2]))
positions = [row["variant_position"]
             for row in reader.variants(fields=["variant_position"])]
print("positions:", positions)
```

(sec-null-samples)=
## Null samples

Some VCZ datasets contain *null* (masked) samples: slots whose `sample_id` is
the empty string `""`. They must never be read, filtered on, or output. The
example dataset has one at index 1. The reader hides them automatically —
`sample_ids` and every iterated `call_*` array exclude null samples, and
sample-dependent virtual fields (`AC`/`AN`/`AF`/`NS`) ignore them.

The raw picture is available alongside the filtered view:

```{code-cell} ipython3
reader = vcztools.VczReader(root)

print("num_samples (raw, counts nulls):", reader.num_samples)
print("raw_sample_ids:", reader.raw_sample_ids)
print("non_null_sample_indices:", reader.non_null_sample_indices)
print("sample_ids (nulls hidden):", reader.sample_ids)
```

Iterating drops the null sample's columns, so `call_*` arrays have one column
per *real* sample:

```{code-cell} ipython3
chunk = next(reader.variant_chunks(fields=["call_genotype"]))
print("call_genotype sample columns:", chunk["call_genotype"].shape[1])
```

Null samples have no `sample_id`, so {meth}`~vcztools.VczReader.set_samples`
cannot name one — null slots are simply unreachable by ID.

For the lower-level case where you select by raw integer index, use
{meth}`~vcztools.VczReader.set_sample_indexes`. Indexes are positions in the raw
`sample_id` array, and selecting a null index raises a `ValueError`:

```{code-cell} ipython3
reader = vcztools.VczReader(root)
try:
    reader.set_sample_indexes([1])  # index 1 is the null sample
except ValueError as exc:
    print(exc)
```

Build index-based selections from `non_null_sample_indices` to avoid them:

```{code-cell} ipython3
reader = vcztools.VczReader(root)
reader.set_sample_indexes(reader.non_null_sample_indices[[0, 1]])
print("selected samples:", reader.sample_ids)
```

## Filtering

{class}`vcztools.BcftoolsFilter` compiles a bcftools `-i`/`-e` expression into a
filter. Construct it from the reader (so bare VCF names resolve against the
dataset's fields), then attach it with
{meth}`~vcztools.VczReader.set_variant_filter`:

```{code-cell} ipython3
reader = vcztools.VczReader(root)
variant_filter = vcztools.BcftoolsFilter(reader, include="QUAL>10")
reader.set_variant_filter(variant_filter)

for row in reader.variants(fields=["variant_position", "variant_quality"]):
    print(row["variant_position"], row["variant_quality"])
```

The fixed-width writers and encoders in {ref}`sec-format-conversion` cannot
iterate a still-configured filter directly; resolve it into a fixed selection
first with {meth}`~vcztools.VczReader.materialise_variant_filter`.
