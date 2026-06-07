(sec-vcf)=
# VCF

The VCF emitted by {ref}`vcztools view<cmd-vcztools-view>` and
{func}`vcztools.write_vcf` conforms to the VCF specification and is
parse-equivalent to bcftools: the same records, fields, and values. It is
**not** guaranteed to be byte-for-byte identical to the corresponding bcftools
output. The differences, all of which remain spec-conformant, are documented
below.

## Missing and fill values

VCZ stores ragged fields by padding short rows with a *fill* sentinel, distinct
from the *missing* sentinel. The two are collapsed on output, following bcftools:

- A missing or fill **scalar** is written as `.`.
- In a **multi-valued** field (`Number=A`, `R`, `G`, or `.`), trailing fill is
  trimmed, an interior missing element is written as `.`, and a row that is
  entirely missing or fill collapses to a single `.`.

For example, an `INFO/AF` field stored as the rows `[1.5, 2.5]`, `[3.5, fill]`,
`[missing, fill]` is written as:

```
AF=1.5,2.5
AF=3.5
.
```

The same trimming applies per sample in `FORMAT` fields, where an all-fill row
and an all-missing row are *not* equivalent. A two-element row stored as
`[fill, fill]` trims to empty and is written as `.`, whereas `[missing, missing]`
keeps both elements and is written as `.,.`.

This exposes one case where the output cannot match bcftools. vcztools does not
always distinguish a field that is *absent* for a record from one that is
*present but entirely missing*, so it may write `.,.` where bcftools omits the
field altogether. Both forms are valid per the specification
([issue #14](https://github.com/sgkit-dev/vcztools/issues/14)).

## Other differences from bcftools

- **Floating-point precision.** Floats are rounded to a fixed three decimal
  places, with trailing zeros dropped (so `0.5` not `0.500`); bcftools formats
  with libc's `%g`. This is the most common source of textual differences — for
  example a stored `0.3333` is written as `0.333`.
- **Field ordering.** `INFO` and `FORMAT` tags are emitted in sorted
  (alphabetical) order for deterministic output, regardless of their declaration
  order in the header; bcftools preserves declaration order. `GT`, when present,
  is always the first `FORMAT` field, as the specification requires.

## Uncompressed only

The output is plain text VCF; compressed or binary suffixes (`.gz`, `.bgz`,
`.bcf`) are rejected. Pipe through `bgzip` or `bcftools` if you need bgzipped VCF
or BCF.

## See also

- {ref}`The view flag reference<cmd-vcztools-view>`.
- [VCF specification](https://samtools.github.io/hts-specs/VCFv4.3.pdf).
