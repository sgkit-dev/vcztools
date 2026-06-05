(sec-format-conversion)=
# Format conversion

`vcztools` can convert a VCZ dataset into the PLINK 1 binary fileset and the
Oxford BGEN format from Python. The same conversions are available on the
command line as {ref}`view-plink<cmd-vcztools-view-plink>` and
{ref}`view-bgen<cmd-vcztools-view-bgen>`; the {ref}`sec-plink` and
{ref}`sec-bgen` pages describe the format semantics (allele conventions,
missingness, multi-allelic handling) in detail. This page covers the Python
entry points — see the {ref}`sec-python-api` for full signatures.

All writers take a configured {class}`vcztools.VczReader`, so any sample or
variant selection (see {ref}`sec-reading-vcz`) is applied to the output. A
variant filter must first be resolved into a fixed selection with
{meth}`~vcztools.VczReader.materialise_variant_filter`; the fixed-width formats
cannot consume a still-configured filter.

## One-shot writers

{func}`vcztools.write_plink` writes a complete PLINK 1 fileset under a stem,
producing `out.bed` plus, by default, the `out.bim` and `out.fam` sidecars:

```python
import vcztools

root = vcztools.open_zarr("sample.vcz.zip")
with vcztools.VczReader(root) as reader:
    vcztools.write_plink(reader, "out")
```

{func}`vcztools.write_bgen` writes a `.bgen` file (to a path or a writable
binary file object), optionally with `.sample` and `.bgen.bgi` sidecars:

```python
with vcztools.VczReader(root) as reader:
    vcztools.write_bgen(reader, "out.bgen")
```

Both writers are biallelic-only and reject sample-scope filters, since the
output is fixed-width per variant.

## Sidecar functions

The individual sidecar files can be written on their own when you need finer
control: {func}`vcztools.write_bim` and {func}`vcztools.write_fam` for PLINK,
and {func}`vcztools.write_sample` and {func}`vcztools.write_bgi` for BGEN.
`write_bgi` needs the per-variant byte offsets, available as
{attr}`BgenEncoder.variant_offsets <vcztools.BgenEncoder>`.

## Streaming encoders

For callers that need the raw byte stream rather than a file, the
{class}`vcztools.BedEncoder` and {class}`vcztools.BgenEncoder` classes expose
the `.bed` / `.bgen` payload through the random-access read API of their shared
base, {class}`vcztools.FormatEncoder`. This is the building block the one-shot
writers use, and it lets you stream a conversion into another sink:

```python
with vcztools.VczReader(root) as reader:
    with vcztools.BedEncoder(reader) as encoder:
        print(encoder.bed_size, "bytes")
        encoder.write_to(open("out.bed", "wb"))
```
