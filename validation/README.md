# Downstream-tool validation suite

This directory exercises `vcztools view-plink` and `vcztools view-bgen`
output against the production downstream consumers of those formats:

- **qctool v2** — BGEN reader / SNP-stats
- **REGENIE** — GWAS, both PLINK and BGEN input paths
- **PLINK 1.9** — `.bed`/`.bim`/`.fam` reader (rejects BGEN v1.2)
- **bgenix** — BGEN indexer / lister / range subsetter
- **BOLT-LMM** — PLINK + BGEN reader for LMM input

The suite is **opt-in and Linux-only**. It is not wired into CI: it
downloads multi-MB binaries, builds bgenix from source, and runs
external tools that take minutes on synthetic-but-realistic data
(5000 samples × ~19k variants for BOLT-LMM).

## Quick start

```bash
# From the repo root, install all 5 tools (parallel ok),
# generate fixtures, then run the 24-test suite.
make -C validation -j tools
make -C validation data
make -C validation test

# Or combined:
make -C validation -j all
```

A clean run from scratch takes ~10 min wall (most of which is tool
downloads and the bgenix waf build).

## Prerequisites

- Linux x86_64
- `uv` (for `uv run pytest`)
- A C/C++ toolchain (GCC 11+) for the bgenix source build
- `curl`, `tar`, `python3` (the bgenix waf script invokes
  `python3` directly, not via uv)

## What gets tested

For each downstream tool, we check four things:

1. **File-format contract** — the tool accepts the bytes vcztools
   wrote and reports the expected sample / variant counts.
2. **Round-trip statistic** — where the tool offers a simple per-
   variant statistic (ALT allele frequency, minor allele frequency),
   we compute the same number directly from the source VCZ in
   `reference.py` and assert numerical agreement. vcztools writes
   hard calls (P=1.0 on the called genotype), so BGEN-derived
   frequencies round-trip exactly — comparisons use `atol=1e-10`.
3. **Variant ID round-trip** — the per-tool variant-ID output column
   matches the IDs injected into the VCZ. IDs are deterministic and
   span three length buckets (~3, 15, 46 bytes) so truncation and
   width-allocation bugs surface as mismatches.
4. **Phased-BGEN behaviour** — vcztools writes phased BGEN when the
   source VCZ has `call_genotype_phased=True`. Each tool gets a
   `TestXxxPhasedBgen` class that pins what it does on phased input:
   qctool and REGENIE both refuse phased BGEN with documented
   diagnostics; bgenix and BOLT-LMM accept it. PLINK 1.9 rejects all
   BGEN v1.2 regardless of phase (covered by the existing layout-2
   rejection test).

The ground truth is computed from the **source VCZ**, not from the
tool's own output round-tripped back in. This is the point: it would
defeat the suite to ask a tool to validate against its own input.

### Per-tool coverage

| Test file | Tool | Coverage |
|---|---|---|
| `test_qctool.py` | qctool v2.2.5 | `-snp-stats` ALT and minor allele frequency match VCZ; AA+AB+BB+NULL = N; `alternate_ids` round-trip |
| `test_plink19.py` | PLINK 1.9 v20231211 | `--freq` MAF matches VCZ exactly from PLINK input; `.bim` SNP column round-trip; pins "BGEN v1.2 input requires PLINK 2.0" |
| `test_bgenix.py` | bgenix (waf build) | `-list` positions/alleles/rsid/alternate_ids match VCZ; `-index` rebuild; `-incl-range` subset count |
| `test_regenie.py` | REGENIE v4.1 | step-1 + step-2 end-to-end on PLINK and BGEN input; per-variant `A1FREQ` and `ID` round-trip |
| `test_bolt_lmm.py` | BOLT-LMM v2.5 | BOLT parses both PLINK and BGEN input with the right N/M counts and runs LINREG. Uses `--exclude` to skip all but the first 20 SNPs so each run is ~1.5 s instead of minutes; the summary lines we assert on report the pre-exclude count. |

BGEN tests are parametrised over three flavours: `view-bgen` at
compression level `-1` (zlib default) and `0` (uncompressed /
"stored"), plus a third flavour produced by `vcztools.bgen.BgenEncoder`
— a Python-API-only random-access encoder that writes fixed-width
variant blocks (every variant exactly `bytes_per_variant` bytes,
strings NUL-padded). The encoder has no CLI flag; the suite drives
it directly via `helpers.run_bgen_encoder` and writes the byte
stream + a separate `.sample` sidecar.

`BgenEncoder` writes no `.bgen.bgi` index — the bgenix tests build
one inline via `bgenix -index -clobber`. REGENIE does not strip the
encoder's trailing NUL padding from variant-ID strings on read; the
ID round-trip test compensates by `rstrip("\x00")`-ing REGENIE's `ID`
column before comparing.

## Tool versions

Pinned in the `install/install_*.sh` scripts; bump the constant at the
top of each script to upgrade.

| Tool | Version | Source |
|---|---|---|
| qctool | 2.2.5 | static tarball from `www.well.ox.ac.uk/~gav/resources/` |
| REGENIE | 4.1 | release zip from `github.com/rgcgithub/regenie/releases` |
| PLINK 1.9 | v20231211 | static binary from `s3.amazonaws.com/plink1-assets/` |
| bgenix | latest release tarball | source tarball from `enkre.net/cgi-bin/code/bgen` |
| BOLT-LMM | 2.5 | static tarball from `alkesgroup.broadinstitute.org` |

The bgenix install script applies a one-line patch to
`src/View.cpp` (`std::ios::streampos origin =` → `auto origin =`)
that is required to compile under GCC 13+.

## Test data

Each size produces two synthetic fixtures, generated by
`generate_data.py` from one msprime simulation via
`bio2zarr.tskit.convert`:

Fixtures live outside the repo. Default location is
`../datasets/validation/` (relative to the vcztools repo root, i.e.
`../../datasets/validation/` relative to this directory). They are
kept outside the repo so recursive tools (`find`, `grep -r`, `rg`)
don't trip on the multi-MB Zarr stores. `DATA_DIR` is defined in
`generate_data.py` / `conftest.py` and as `$(DATA_DIR)` in the
Makefile; point your editor / shell at that path if you need to
inspect the artefacts.

| Fixture | Samples | Variants | Used by |
|---|---|---|---|
| `small_unphased.vcz` + `small_phased.vcz` | 200 | ~1000 | qctool, PLINK 1.9, bgenix, REGENIE |
| `large_unphased.vcz` + `large_phased.vcz` | 5000 | ~19500 | BOLT-LMM |

The unphased and phased stores share genotypes; only
`call_genotype_phased` differs. The default tests run against the
unphased store (the production path for most downstream tools); the
`TestXxxPhasedBgen` classes use the phased store to pin per-tool
behaviour.

Phenotypes (`<size>.pheno.tsv`, in `DATA_DIR`) are simulated with
`tstrait` on the source tree sequence (h²=0.5, 100 causal SNPs) and
shared between the phased / unphased stores of a given size. The phenotype
values are not load-bearing for any of our checks — we compare allele
frequencies, not effect sizes — but having a non-trivial genetic
architecture lets REGENIE step-1 fit a null model cleanly.

`generate_data.py` zeros out `call_genotype_phased` on the unphased
store after writing. msprime data is fully phased, but qctool refuses
our phased BGEN layout (`order_type=6 value_type=1`); flattening the
phase flag works around this without altering the genotypes.

The same script also injects a `variant_id` array — `bio2zarr.tskit.convert`
doesn't populate one from a tree sequence. The injected IDs cycle
through three format strings (e.g. `rs0`, `rs_var_00000001`,
`rs_variant_0000000002_chr_00_pos_0000000178`) so the downstream
round-trip checks have variable-length signal to verify. All IDs are
≤ 50 bytes, fitting `BgenEncoder`'s default `rsid_max=64`.

## Caveats

### BOLT-LMM heritability is not asserted

BOLT-LMM's variance-component fit needs realistic LD across multiple
chromosomes; on the 5000-sample × 19k-SNP fixture it consistently
estimates h²g ≈ 0 and aborts with a floating-point exception in
the inf-model step. Real BOLT-LMM benchmarks use UK-Biobank-scale
data and that's not reproducible in seconds.

The BOLT-LMM tests therefore only validate the file-format contract:
BOLT parses the genotype inputs, reports the right sample/SNP counts
in its log, and runs the LINREG step on PLINK input. The
allele-frequency cross-check is left to qctool, PLINK 1.9, and
REGENIE, all of which give reliable per-variant frequency output on
the same fixtures.

### PLINK 1.9 rejects BGEN v1.2

PLINK 1.9 only supports BGEN v1.1; vcztools writes BGEN layout-2
(v1.2). The test `TestPlink19RejectsBgenV12` pins the diagnostic
message so a future PLINK release that adds layout-2 support will
flag a test update.

### FID convention mismatch

`view-plink` writes `.fam` with `FID=0`, `IID=sample_id`. `view-bgen`
writes `.sample` with `ID_1=ID_2=sample_id`. When BOLT-LMM and
REGENIE consume the PLINK fileset, the pheno file is rewritten in-
test to `FID=0`. When BOLT-LMM is given both `--bfile` and
`--bgenFile` it cross-checks the two; the `.sample` is also
rewritten in-test to `FID=0`.

### Parallel pytest workers

The repo's `pyproject.toml` sets `addopts = "-n auto"` (pytest-
xdist). The validation suite overrides this to `-o "addopts="` in
its Makefile target because BOLT-LMM OOMs when run in parallel
worker processes on the 5000-sample fixture. Per-test cost is
dominated by external binary launch anyway, so serial execution is
the right default here.

## Make targets

| Target | Effect |
|---|---|
| `make tools` | Install every tool into `tools/<name>/`. Idempotent (skipped if `.installed` marker present). |
| `make <tool>` | Install one tool, e.g. `make qctool`. |
| `make data` | Generate `small.vcz`, `large.vcz`, plus pheno files into `$(DATA_DIR)` (`../../datasets/validation/`). |
| `make test` | `uv run pytest validation/` serially. Tests for missing tools are skipped. |
| `make all` | `tools` + `data` + `test`. |
| `make clean-tools` | `rm -rf tools/`. |
| `make clean-data` | `rm -rf $(DATA_DIR)`. |
| `make clean` | Both. |

## Layout

```
validation/
├── README.md                    # this file
├── Makefile                     # entry point
├── conftest.py                  # tool/fixture discovery
├── helpers.py                   # subprocess + view-plink/view-bgen wrappers
├── reference.py                 # ground-truth allele frequency from VCZ
├── generate_data.py             # msprime → bio2zarr → VCZ + tstrait phenotype
├── install/install_<tool>.sh    # one install script per tool
├── tools/<tool>/                # installed binaries (gitignored)
└── test_<tool>.py               # one test module per tool

# Fixtures live outside this directory, at the default location:
#   ../datasets/validation/<size>.vcz/   (not in the repo tree)
```
