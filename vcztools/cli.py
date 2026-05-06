import contextlib
import json
import logging
import os
import sys
import warnings
from functools import wraps

import click
import numpy as np

from . import bcftools_filter, plink, provenance, retrieval, vcf_writer
from . import query as query_module
from . import regions as regions_mod
from . import samples as samples_mod
from . import stats as stats_module
from .utils import open_zarr


@contextlib.contextmanager
def handle_broken_pipe(output):
    """
    Handle sigpipe following official advice:
    https://docs.python.org/3/library/signal.html#note-on-sigpipe
    """
    try:
        yield
        # flush output here to force SIGPIPE to be triggered
        # while inside this try block.
        output.flush()
    except BrokenPipeError:
        # Python flushes standard streams on exit; redirect remaining output
        # to devnull to avoid another BrokenPipeError at shutdown
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        sys.exit(1)  # Python exits with error code 1 on EPIPE


def handle_exception(func):
    """
    Handle known application exceptions (ValueError) by converting to
    a ClickException, so the message is written to stderr and a non-zero exit
    code is set.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            raise click.ClickException(e) from e

    return wrapper


include = click.option(
    "-i", "--include", type=str, help="Filter expression to include variant sites."
)
exclude = click.option(
    "-e", "--exclude", type=str, help="Filter expression to exclude variant sites."
)
force_samples = click.option(
    "--force-samples", is_flag=True, help="Only warn about unknown sample subsets."
)
output = click.option(
    "-o",
    "--output",
    type=click.File("w"),
    default="-",
    help="File path to write output to (defaults to stdout '-').",
)
regions = click.option(
    "-r",
    "--regions",
    type=str,
    default=None,
    help="Regions to include.",
)
regions_file = click.option(
    "-R",
    "--regions-file",
    type=str,
    default=None,
    help="File of regions to include.",
)
samples = click.option(
    "-s",
    "--samples",
    type=str,
    default=None,
    help="Samples to include.",
)
samples_file = click.option(
    "-S",
    "--samples-file",
    type=str,
    default=None,
    help="File of sample names to include.",
)
targets = click.option(
    "-t",
    "--targets",
    type=str,
    default=None,
    help="Target regions to include.",
)
targets_file = click.option(
    "-T",
    "--targets-file",
    type=str,
    default=None,
    help="File of target regions to include.",
)
types_opt = click.option(
    "-v",
    "--types",
    type=str,
    default=None,
    help=(
        "Comma-separated list of variant types to include "
        "(snps,indels,mnps,other). A site is selected if any of its "
        "alleles matches one of the listed types."
    ),
)
exclude_types_opt = click.option(
    "-V",
    "--exclude-types",
    type=str,
    default=None,
    help="Comma-separated list of variant types to exclude.",
)
min_alleles_opt = click.option(
    "-m",
    "--min-alleles",
    type=int,
    default=None,
    help="Print sites with at least INT alleles listed in REF and ALT.",
)
max_alleles_opt = click.option(
    "-M",
    "--max-alleles",
    type=int,
    default=None,
    help="Print sites with at most INT alleles listed in REF and ALT.",
)
version = click.version_option(version=f"{provenance.__version__}")

log_level = click.option(
    "--log-level",
    type=click.Choice(["WARNING", "INFO", "DEBUG"], case_sensitive=False),
    default="WARNING",
    show_default=True,
    help="Logging verbosity.",
)
log_file = click.option(
    "--log-file",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Write log output to FILE instead of stderr.",
)


def setup_logging(level: str, log_file: str | None) -> None:
    if log_file is not None:
        handler: logging.Handler = logging.FileHandler(log_file)
    else:
        handler = logging.StreamHandler()
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[handler],
        force=True,
    )


def _zarr_backend_storage_alias_callback(ctx, param, value):
    # ``--zarr-backend-storage`` is accepted as a deprecated alias for
    # ``--backend-storage``. Forward the value to ``backend_storage`` so
    # the command receives it under the new name, and emit a warning.
    if value is not None:
        warnings.warn(
            "--zarr-backend-storage is deprecated; use --backend-storage",
            DeprecationWarning,
            stacklevel=2,
        )
        if ctx.params.get("backend_storage") is None:
            ctx.params["backend_storage"] = value
    return value


_backend_storage_option = click.option(
    "--backend-storage",
    "backend_storage",
    type=str,
    default=None,
    help=(
        "Zarr backend storage: omit for local-only (default), 'fsspec', "
        "'obstore', or 'icechunk'. The default supports .zip → ZipStore "
        "and local directories → LocalStore; URLs require an explicit "
        "backend."
    ),
)
_zarr_backend_storage_alias_option = click.option(
    "--zarr-backend-storage",
    "_zarr_backend_storage_alias",
    type=str,
    default=None,
    expose_value=False,
    callback=_zarr_backend_storage_alias_callback,
    hidden=True,
)


def backend_storage(f):
    """Decorator stack: primary ``--backend-storage`` option plus the
    deprecated, hidden ``--zarr-backend-storage`` alias."""
    return _backend_storage_option(_zarr_backend_storage_alias_option(f))


storage_option = click.option(
    "--storage-option",
    "storage_options",
    multiple=True,
    type=str,
    help=(
        "Backend storage option as KEY=VALUE (repeatable). VALUE is parsed "
        "as JSON if possible, falling back to a string."
    ),
)


def _parse_storage_options(pairs: tuple[str, ...]) -> dict | None:
    """Convert a tuple of ``KEY=VALUE`` strings into a dict.

    Each ``VALUE`` is decoded with :func:`json.loads` first; on
    decode failure the raw string is kept. Returns ``None`` for an
    empty input so callers can pass it straight through to
    :func:`vcztools.open_zarr`.
    """
    if len(pairs) == 0:
        return None
    options: dict = {}
    for pair in pairs:
        if "=" not in pair:
            raise click.BadParameter(f"--storage-option must be KEY=VALUE: {pair!r}")
        key, raw = pair.split("=", 1)
        try:
            options[key] = json.loads(raw)
        except json.JSONDecodeError:
            options[key] = raw
    return options


# Match bcftools' accepted set for ``view -v/-V``; bcftools rejects
# ``refs`` and ``bnd`` outright. Only ``snps`` is fully wired through
# to the TYPE operator at the moment — the others reach
# ``UnsupportedTypeFieldError`` via the parser (see issue #166).
_BCFTOOLS_TYPE_KEYWORDS = ("snps", "indels", "mnps", "other")
_BCFTOOLS_TYPE_TO_SINGULAR = {
    "snps": "snp",
    "indels": "indel",
    "mnps": "mnp",
    "other": "other",
}


def _parse_types_option(value, option_name):
    """Validate a comma-separated --types/--exclude-types argument and
    return the list of TYPE-operator singular forms.
    """
    types = value.split(",")
    invalid = [t for t in types if t not in _BCFTOOLS_TYPE_KEYWORDS]
    if len(invalid) > 0:
        raise ValueError(
            f"Invalid type(s) {invalid} in --{option_name}; "
            f"valid types are: {', '.join(_BCFTOOLS_TYPE_KEYWORDS)}."
        )
    return [_BCFTOOLS_TYPE_TO_SINGULAR[t] for t in types]


def _build_view_filter_expression(types, exclude_types, min_alleles):
    """Translate the bcftools-style ``-v/-V/-m`` view options to a
    synthetic filter expression, or ``None`` if none of them are set.

    ``-v LIST`` becomes a disjunction of ``TYPE~"X"`` matches (any-allele
    semantics, matching bcftools), ``-V LIST`` becomes a conjunction of
    ``TYPE!~"X"``, and ``-m N`` becomes ``N_ALT >= N - 1`` (a site has
    REF + N_ALT alleles).
    """
    parts = []
    if types is not None:
        type_list = _parse_types_option(types, "types")
        parts.append(" || ".join(f'TYPE~"{t}"' for t in type_list))
    if exclude_types is not None:
        type_list = _parse_types_option(exclude_types, "exclude-types")
        parts.append(" && ".join(f'TYPE!~"{t}"' for t in type_list))
    if min_alleles is not None:
        if min_alleles < 1:
            raise ValueError(f"--min-alleles must be >= 1, got {min_alleles}")
        if min_alleles >= 2:
            parts.append(f"N_ALT >= {min_alleles - 1}")
    if len(parts) == 0:
        return None
    return " && ".join(f"({p})" for p in parts)


class _AndFilter:
    """AND-compose multiple :class:`~vcztools.variant_filter.VariantFilter`
    objects, including across scopes.

    A variant-scope (1-D) result is broadcast along axis 1 against any
    sample-scope (2-D) result, so a synthetic ``-m/-M/-v/-V`` mask
    composes naturally with a user ``-i 'FMT/...'`` mask. The combined
    scope is ``sample`` when any input is sample-scope, else ``variant``.
    """

    def __init__(self, filters):
        self._filters = list(filters)
        self.scope = (
            "sample" if any(f.scope == "sample" for f in self._filters) else "variant"
        )
        self.referenced_fields = frozenset().union(
            *(f.referenced_fields for f in self._filters)
        )

    def evaluate(self, chunk_data):
        result = self._filters[0].evaluate(chunk_data)
        for f in self._filters[1:]:
            other = f.evaluate(chunk_data)
            if result.ndim == 1 and other.ndim == 2:
                result = np.expand_dims(result, axis=1) & other
            elif result.ndim == 2 and other.ndim == 1:
                result = result & np.expand_dims(other, axis=1)
            else:
                result = result & other
        return result


def _compose_filter(existing, new_filter):
    """AND-compose ``new_filter`` onto an existing filter, allowing
    mixed variant/sample scope via :class:`_AndFilter`. Returns
    ``new_filter`` unchanged when there's nothing to compose with.
    """
    if existing is None:
        return new_filter
    return _AndFilter([existing, new_filter])


def make_reader(
    path,
    *,
    regions=None,
    regions_file=None,
    targets=None,
    targets_file=None,
    samples=None,
    samples_file=None,
    include=None,
    exclude=None,
    types=None,
    exclude_types=None,
    min_alleles=None,
    max_alleles=None,
    view_semantics=False,
    force_samples=False,
    drop_genotypes=False,
    backend_storage=None,
    storage_options=None,
):
    """Resolve file arguments and create a VczReader."""
    if regions is not None and regions_file is not None:
        raise ValueError(
            "Cannot specify both a regions string (-r) and a regions file (-R)"
        )
    if targets is not None and targets_file is not None:
        raise ValueError(
            "Cannot specify both a target string (-t) and a targets file (-T)"
        )
    if samples is not None and samples_file is not None:
        raise ValueError(
            "Cannot specify both a samples string (-s) and a samples file (-S)"
        )
    if regions_file is not None:
        regions = regions_mod.read_regions_file(regions_file)
    elif regions is not None:
        regions = regions.split(",")
    targets_complement = False
    if targets_file is not None:
        if targets_file.startswith("^"):
            targets_complement = True
            targets_file = targets_file[1:]
        targets = regions_mod.read_regions_file(targets_file)
    elif targets is not None:
        if targets.startswith("^"):
            targets_complement = True
            targets = targets[1:]
        targets = targets.split(",")
    samples_complement = False
    if samples_file is not None:
        if samples_file.startswith("^"):
            samples_complement = True
            samples_file = samples_file[1:]
        samples = samples_mod.read_samples_file(samples_file)
    elif samples is not None:
        if samples.startswith("^"):
            samples_complement = True
            samples = samples[1:]
        samples = samples.split(",")
    root = open_zarr(
        path,
        mode="r",
        backend_storage=backend_storage,
        storage_options=storage_options,
    )
    reader = retrieval.VczReader(root)

    variant_filter = None
    if include is not None or exclude is not None:
        variant_filter = bcftools_filter.BcftoolsFilter(
            field_names=reader.field_names, include=include, exclude=exclude
        )

    if max_alleles is not None:
        variant_filter = _compose_filter(
            variant_filter, plink.MaxAllelesFilter(max_alleles)
        )

    view_expr = _build_view_filter_expression(types, exclude_types, min_alleles)
    if view_expr is not None:
        synthetic_filter = bcftools_filter.BcftoolsFilter(
            field_names=reader.field_names, include=view_expr
        )
        variant_filter = _compose_filter(variant_filter, synthetic_filter)

    if drop_genotypes:
        # --drop-genotypes can't coexist with a sample-scope filter:
        # the filter needs per-sample genotype data, --drop-genotypes
        # says you don't want any.
        if variant_filter is not None and variant_filter.scope == "sample":
            raise ValueError(
                "sample-scope variant_filter is incompatible with drop_genotypes=True"
            )
        reader.set_samples([])
    elif samples is not None:
        samples = samples_mod.resolve_sample_selection(
            samples,
            reader.raw_sample_ids,
            complement=samples_complement,
            ignore_missing_samples=force_samples,
        )
        if samples is not None and len(samples) == 0:
            raise ValueError(
                "Empty sample set is not supported. The bcftools semantics "
                "for this corner case (AC/AN recomputed over the full "
                "non-null sample set while emitting zero sample columns) "
                "would require significant internal complexity. "
                "If you need this feature, please open an issue at "
                "https://github.com/sgkit-dev/vcztools/issues."
            )
        reader.set_samples(samples)

    if regions is not None or targets is not None:
        variant_chunk_plan = regions_mod.build_chunk_plan(
            root,
            regions=regions,
            targets=targets,
            targets_complement=targets_complement,
        )
        reader.set_variants(variant_chunk_plan)

    if view_semantics:
        # bcftools view evaluates sample-scope filters over the
        # pre-subset (non-null) axis and exposes that axis for AC/AN
        # recompute.
        reader.set_filter_samples(reader.non_null_sample_indices)

    if variant_filter is not None:
        reader.set_variant_filter(variant_filter)

    return reader


class NaturalOrderGroup(click.Group):
    """
    List commands in the order they are provided in the help text.
    """

    def list_commands(self, ctx):
        return self.commands.keys()


@click.command
@click.argument("path", type=click.Path())
@click.option(
    "-n",
    "--nrecords",
    is_flag=True,
    help="Print the number of records (variants).",
)
@click.option(
    "-s",
    "--stats",
    is_flag=True,
    help="Print per contig stats.",
)
@backend_storage
@storage_option
@log_level
@log_file
@handle_exception
def index(path, nrecords, stats, backend_storage, storage_options, log_level, log_file):
    """
    Query the number of records in a VCZ dataset. This subcommand only
    implements the --nrecords and --stats options and does not build any
    indexes.
    """
    setup_logging(log_level, log_file)
    if nrecords and stats:
        raise click.UsageError("Expected only one of --stats or --nrecords options")
    if not nrecords and not stats:
        raise click.UsageError("Building region indexes is not supported")
    root = open_zarr(
        path,
        mode="r",
        backend_storage=backend_storage,
        storage_options=_parse_storage_options(storage_options),
    )
    with retrieval.VczReader(root) as reader:
        if nrecords:
            stats_module.nrecords(reader, sys.stdout)
        else:
            stats_module.stats(reader, sys.stdout)


@click.command
@click.argument("path", type=click.Path())
@output
@click.option(
    "-l",
    "--list-samples",
    is_flag=True,
    help="List the sample IDs and exit.",
)
@click.option(
    "-f",
    "--format",
    type=str,
    help="The format of the output.",
    default=None,
)
@regions
@regions_file
@force_samples
@samples
@samples_file
@targets
@targets_file
@include
@exclude
@click.option(
    "-N",
    "--disable-automatic-newline",
    is_flag=True,
    help="Disable automatic addition of a missing newline character at the end "
    "of the formatting expression.",
)
@backend_storage
@storage_option
@log_level
@log_file
@handle_exception
def query(
    path,
    output,
    list_samples,
    format,
    regions,
    regions_file,
    targets,
    targets_file,
    force_samples,
    samples,
    samples_file,
    include,
    exclude,
    disable_automatic_newline,
    backend_storage,
    storage_options,
    log_level,
    log_file,
):
    """
    Transform VCZ into user-defined formats with efficient subsetting and
    filtering. Intended as a drop-in replacement for bcftools query, where we
    replace the VCF file path with a VCZ dataset URL.

    This is an early version and not feature complete: if you are missing a
    particular piece of functionality please open an issue at
    https://github.com/sgkit-dev/vcztools/issues
    """
    setup_logging(log_level, log_file)
    parsed_storage_options = _parse_storage_options(storage_options)
    if list_samples:
        # bcftools query -l ignores the --output option and always writes to stdout
        output = sys.stdout
        root = open_zarr(
            path,
            mode="r",
            backend_storage=backend_storage,
            storage_options=parsed_storage_options,
        )
        with retrieval.VczReader(root) as reader, handle_broken_pipe(output):
            query_module.list_samples(reader, output)
        return

    if format is None:
        raise click.UsageError("Missing option -f / --format")

    reader = make_reader(
        path,
        regions=regions,
        regions_file=regions_file,
        targets=targets,
        targets_file=targets_file,
        samples=samples,
        samples_file=samples_file,
        include=include,
        exclude=exclude,
        force_samples=force_samples,
        backend_storage=backend_storage,
        storage_options=parsed_storage_options,
    )
    with reader, handle_broken_pipe(output):
        query_module.write_query(
            reader,
            output,
            query_format=format,
            disable_automatic_newline=disable_automatic_newline,
        )


@click.command
@click.argument("path", type=click.Path())
@output
@click.option(
    "-h",
    "--header-only",
    is_flag=True,
    help="Output the VCF header only.",
)
@click.option(
    "-H",
    "--no-header",
    is_flag=True,
    help="Suppress the header in VCF output.",
)
@click.option(
    "--no-version",
    is_flag=True,
    help="Do not append version and command line information to the output VCF header.",
)
@regions
@regions_file
@force_samples
@click.option(
    "-I",
    "--no-update",
    is_flag=True,
    help="Do not recalculate INFO fields for the sample subset.",
)
@samples
@samples_file
@click.option(
    "-G",
    "--drop-genotypes",
    is_flag=True,
    help="Drop genotypes.",
)
@targets
@targets_file
@include
@exclude
@types_opt
@exclude_types_opt
@min_alleles_opt
@max_alleles_opt
@backend_storage
@storage_option
@log_level
@log_file
@handle_exception
def view(
    path,
    output,
    header_only,
    no_header,
    no_version,
    regions,
    regions_file,
    targets,
    targets_file,
    force_samples,
    no_update,
    samples,
    samples_file,
    drop_genotypes,
    include,
    exclude,
    types,
    exclude_types,
    min_alleles,
    max_alleles,
    backend_storage,
    storage_options,
    log_level,
    log_file,
):
    """
    Convert VCZ dataset to VCF with efficient subsetting and filtering.
    Intended as a drop-in replacement for bcftools view, where
    we replace the VCF file path with a VCZ dataset URL.

    This is an early version and not feature complete: if you are missing a
    particular piece of functionality please open an issue at
    https://github.com/sgkit-dev/vcztools/issues
    """
    setup_logging(log_level, log_file)
    suffix = output.name.split(".")[-1]
    # Exclude suffixes which require bgzipped or BCF output:
    # https://github.com/samtools/htslib/blob/329e7943b7ba3f0af15b0eaa00a367a1ac15bd83/vcf.c#L3815
    if suffix in ["gz", "bcf", "bgz"]:
        raise ValueError(
            f"Only uncompressed VCF output supported, suffix .{suffix} not allowed"
        )

    if (samples or samples_file) and drop_genotypes:
        raise ValueError("Cannot select samples and drop genotypes.")

    if types is not None and exclude_types is not None:
        raise ValueError("Cannot use --types and --exclude-types together.")

    reader = make_reader(
        path,
        regions=regions,
        regions_file=regions_file,
        targets=targets,
        targets_file=targets_file,
        samples=samples,
        samples_file=samples_file,
        include=include,
        exclude=exclude,
        types=types,
        exclude_types=exclude_types,
        min_alleles=min_alleles,
        max_alleles=max_alleles,
        view_semantics=True,
        force_samples=force_samples,
        drop_genotypes=drop_genotypes,
        backend_storage=backend_storage,
        storage_options=_parse_storage_options(storage_options),
    )
    subsetting_samples = (
        samples is not None or samples_file is not None or drop_genotypes
    )
    with reader, handle_broken_pipe(output):
        vcf_writer.write_vcf(
            reader,
            output,
            subsetting_samples=subsetting_samples,
            header_only=header_only,
            no_header=no_header,
            no_version=no_version,
            no_update=no_update,
            drop_genotypes=drop_genotypes,
        )


@click.command
@click.argument("path", type=click.Path())
@regions
@regions_file
@force_samples
@samples
@samples_file
@targets
@targets_file
@include
@exclude
@types_opt
@exclude_types_opt
@min_alleles_opt
@max_alleles_opt
@click.option(
    "--out",
    default="plink",
    help="Output prefix for the .bed/.bim/.fam fileset.",
)
@backend_storage
@storage_option
@log_level
@log_file
@handle_exception
def view_bed(
    path,
    regions,
    regions_file,
    force_samples,
    samples,
    samples_file,
    targets,
    targets_file,
    include,
    exclude,
    types,
    exclude_types,
    min_alleles,
    max_alleles,
    out,
    backend_storage,
    storage_options,
    log_level,
    log_file,
):
    """
    Generate a PLINK 1 binary fileset (.bed/.bim/.fam) from a VCZ
    dataset.

    A1=ALT, A2=REF (plink 2's convention); the .bed payload is
    byte-identical to ``plink2 --vcf X --make-bed`` for biallelic
    variants. Sample/region/filter selection mirrors bcftools view
    (-s/-S/-r/-R/-t/-T/-i/-e/-v/-V/-m/-M). Multi-allelic variants
    are rejected by default; pass ``-M 2`` (or ``--max-alleles 2``)
    to skip them, matching ``plink2 --vcf X --max-alleles 2 --make-bed``.

    See the "PLINK 1 binary output" documentation page for the full
    reference, including how to read this output with plink 1.9
    (``--keep-allele-order``), REGENIE, BOLT-LMM, and other
    downstream tools.
    """
    setup_logging(log_level, log_file)

    if types is not None and exclude_types is not None:
        raise ValueError("Cannot use --types and --exclude-types together.")

    reader = make_reader(
        path,
        regions=regions,
        regions_file=regions_file,
        targets=targets,
        targets_file=targets_file,
        samples=samples,
        samples_file=samples_file,
        force_samples=force_samples,
        include=include,
        exclude=exclude,
        types=types,
        exclude_types=exclude_types,
        min_alleles=min_alleles,
        max_alleles=max_alleles,
        backend_storage=backend_storage,
        storage_options=_parse_storage_options(storage_options),
    )
    with reader:
        plink.write_plink(reader, out)


@version
@click.group(cls=NaturalOrderGroup, name="vcztools")
def vcztools_main():
    pass


vcztools_main.add_command(index)
vcztools_main.add_command(query)
vcztools_main.add_command(view)
vcztools_main.add_command(view_bed)
