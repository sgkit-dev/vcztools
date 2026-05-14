import contextlib
import dataclasses
import json
import logging
import os
import pathlib
import sys
import warnings
from functools import wraps

import click
import humanfriendly

from . import bcftools_filter, bgen, plink, provenance, retrieval, vcf_writer
from . import query as query_module
from . import regions as regions_mod
from . import samples as samples_mod
from . import stats as stats_module
from . import variant_filter as variant_filter_mod
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


# Help-text section ordering for ``GroupedCommand.format_options``. Options
# without an explicit ``help_group`` render under ``"Options"`` (category 4 —
# command-specific). The other three categories form the long-term stable
# CLI groupings exposed to biofuse and other downstream consumers.
GROUP_ORDER = (
    "Options",
    "Selection options",
    "Zarr store options",
    "Reader options",
    "Logging options",
)


class GroupedOption(click.Option):
    """A click Option tagged with the help-text section it belongs to."""

    def __init__(self, *args, help_group="Options", **kwargs):
        super().__init__(*args, **kwargs)
        self.help_group = help_group


class GroupedCommand(click.Command):
    """A click Command that renders options grouped by ``help_group``."""

    def format_options(self, ctx, formatter):
        buckets: dict[str, list] = {g: [] for g in GROUP_ORDER}
        for param in self.get_params(ctx):
            record = param.get_help_record(ctx)
            if record is None:
                continue
            group = getattr(param, "help_group", "Options")
            buckets.setdefault(group, []).append(record)
        for group in GROUP_ORDER:
            rows = buckets[group]
            if len(rows) > 0:
                with formatter.section(group):
                    formatter.write_dl(rows)


include = click.option(
    "-i",
    "--include",
    type=str,
    help="Filter expression to include variant sites.",
    cls=GroupedOption,
    help_group="Selection options",
)
exclude = click.option(
    "-e",
    "--exclude",
    type=str,
    help="Filter expression to exclude variant sites.",
    cls=GroupedOption,
    help_group="Selection options",
)
force_samples = click.option(
    "--force-samples",
    is_flag=True,
    help="Only warn about unknown sample subsets.",
    cls=GroupedOption,
    help_group="Selection options",
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
    cls=GroupedOption,
    help_group="Selection options",
)
regions_file = click.option(
    "-R",
    "--regions-file",
    type=str,
    default=None,
    help="File of regions to include.",
    cls=GroupedOption,
    help_group="Selection options",
)
samples = click.option(
    "-s",
    "--samples",
    type=str,
    default=None,
    help="Samples to include.",
    cls=GroupedOption,
    help_group="Selection options",
)
samples_file = click.option(
    "-S",
    "--samples-file",
    type=str,
    default=None,
    help="File of sample names to include.",
    cls=GroupedOption,
    help_group="Selection options",
)
targets = click.option(
    "-t",
    "--targets",
    type=str,
    default=None,
    help="Target regions to include.",
    cls=GroupedOption,
    help_group="Selection options",
)
targets_file = click.option(
    "-T",
    "--targets-file",
    type=str,
    default=None,
    help="File of target regions to include.",
    cls=GroupedOption,
    help_group="Selection options",
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
    cls=GroupedOption,
    help_group="Selection options",
)
exclude_types_opt = click.option(
    "-V",
    "--exclude-types",
    type=str,
    default=None,
    help="Comma-separated list of variant types to exclude.",
    cls=GroupedOption,
    help_group="Selection options",
)
min_alleles_opt = click.option(
    "-m",
    "--min-alleles",
    type=int,
    default=None,
    help="Print sites with at least INT alleles listed in REF and ALT.",
    cls=GroupedOption,
    help_group="Selection options",
)
max_alleles_opt = click.option(
    "-M",
    "--max-alleles",
    type=int,
    default=None,
    help="Print sites with at most INT alleles listed in REF and ALT.",
    cls=GroupedOption,
    help_group="Selection options",
)
version = click.version_option(version=f"{provenance.__version__}")

log_level = click.option(
    "--log-level",
    type=click.Choice(["WARNING", "INFO", "DEBUG"], case_sensitive=False),
    default="WARNING",
    show_default=True,
    help="Logging verbosity.",
    cls=GroupedOption,
    help_group="Logging options",
)
log_file = click.option(
    "--log-file",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Write log output to FILE instead of stderr.",
    cls=GroupedOption,
    help_group="Logging options",
)


class _SizeParam(click.ParamType):
    name = "size"

    # binary=True so '256M' == '256MB' == '256MiB' == 2**28, matching
    # the underlying VczReader default of 256 MiB.
    def convert(self, value, param, ctx):
        if value is None:
            return None
        if isinstance(value, int):
            return value
        try:
            return humanfriendly.parse_size(value, binary=True)
        except humanfriendly.InvalidSize as e:
            self.fail(str(e), param, ctx)


SIZE = _SizeParam()

readahead_workers = click.option(
    "--readahead-workers",
    type=int,
    default=None,
    help=("Worker threads servicing the cross-chunk readahead pool. Default: 32."),
    cls=GroupedOption,
    help_group="Reader options",
)
readahead_buffer_size = click.option(
    "--readahead-buffer-size",
    "readahead_bytes",
    type=SIZE,
    default=None,
    help=(
        "Cap on the readahead window. Accepts a byte count or a "
        "size string with suffix (e.g. '100M', '2G', '256MiB'). "
        "Default: 256MiB."
    ),
    cls=GroupedOption,
    help_group="Reader options",
)
encode_threads = click.option(
    "--encode-threads",
    type=int,
    default=4,
    help=(
        "Worker threads for per-chunk VCF line encoding. With >1, each "
        "chunk is split into this many contiguous row blocks encoded in "
        "parallel. Default: 4."
    ),
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
    cls=GroupedOption,
    help_group="Zarr store options",
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
_storage_option = click.option(
    "--storage-option",
    "storage_options",
    multiple=True,
    type=str,
    help=(
        "Backend storage option as KEY=VALUE (repeatable). VALUE is parsed "
        "as JSON if possible, falling back to a string."
    ),
    cls=GroupedOption,
    help_group="Zarr store options",
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


def _build_view_filter_expression(types, exclude_types, min_alleles, max_alleles):
    """Translate the bcftools-style ``-v/-V/-m/-M`` view options to a
    synthetic filter expression, or ``None`` if none of them are set.

    ``-v LIST`` becomes a disjunction of ``TYPE~"X"`` matches (any-allele
    semantics, matching bcftools), ``-V LIST`` becomes a conjunction of
    ``TYPE!~"X"``, ``-m N`` becomes ``N_ALT >= N - 1`` and ``-M N``
    becomes ``N_ALT <= N - 1`` (a site has REF + N_ALT alleles).
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
    if max_alleles is not None:
        if max_alleles < 1:
            raise ValueError(f"--max-alleles must be >= 1, got {max_alleles}")
        parts.append(f"N_ALT <= {max_alleles - 1}")
    if len(parts) == 0:
        return None
    return " && ".join(f"({p})" for p in parts)


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
    readahead_workers=None,
    readahead_bytes=None,
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
    if types is not None and exclude_types is not None:
        raise ValueError("Cannot use --types and --exclude-types together.")
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
    reader = retrieval.VczReader(
        root,
        readahead_workers=readahead_workers,
        readahead_bytes=readahead_bytes,
    )

    variant_filter = None
    if include is not None or exclude is not None:
        variant_filter = bcftools_filter.BcftoolsFilter(
            field_names=reader.field_names, include=include, exclude=exclude
        )

    view_expr = _build_view_filter_expression(
        types, exclude_types, min_alleles, max_alleles
    )
    if view_expr is not None:
        synthetic_filter = bcftools_filter.BcftoolsFilter(
            field_names=reader.field_names, include=view_expr
        )
        variant_filter = variant_filter_mod.compose(variant_filter, synthetic_filter)

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
            reader,
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


@dataclasses.dataclass(frozen=True)
class ZarrStoreOptions:
    """Bundled category-1 (Zarr store) options.

    Public API: stable for downstream consumers such as biofuse. Pair this
    dataclass with the :meth:`decorator` Click decorator on the same class.
    """

    backend_storage: str | None = None
    storage_options: dict | None = None

    @staticmethod
    def decorator(f):
        """Click decorator: add ``--backend-storage`` plus ``--storage-option``
        (and the hidden, deprecated ``--zarr-backend-storage`` alias) to f."""
        return _backend_storage_option(
            _zarr_backend_storage_alias_option(_storage_option(f))
        )

    @classmethod
    def pop_from_click_kwargs(cls, kwargs: dict) -> "ZarrStoreOptions":
        """Pop the Zarr store fields out of ``kwargs``.

        Click's repeatable ``KEY=VALUE`` tuple is parsed to a ``dict | None``
        at this seam so consumers always see the resolved dict.
        """
        raw_pairs = kwargs.pop("storage_options", ())
        return cls(
            backend_storage=kwargs.pop("backend_storage", None),
            storage_options=_parse_storage_options(raw_pairs),
        )


@dataclasses.dataclass(frozen=True)
class ReaderOptions:
    """Bundled category-2 (Reader) options. Pair this dataclass with the
    :meth:`decorator` Click decorator on the same class."""

    readahead_workers: int | None = None
    readahead_bytes: int | None = None

    @staticmethod
    def decorator(f):
        """Click decorator: add ``--readahead-workers`` plus
        ``--readahead-buffer-size`` to f."""
        return readahead_workers(readahead_buffer_size(f))

    @classmethod
    def pop_from_click_kwargs(cls, kwargs: dict) -> "ReaderOptions":
        return cls(
            readahead_workers=kwargs.pop("readahead_workers", None),
            readahead_bytes=kwargs.pop("readahead_bytes", None),
        )


@dataclasses.dataclass(frozen=True)
class LogOptions:
    """Bundled category-3 (Logging) options. Pair this dataclass with the
    :meth:`decorator` Click decorator on the same class."""

    log_level: str = "WARNING"
    log_file: str | None = None

    @staticmethod
    def decorator(f):
        """Click decorator: add ``--log-level`` plus ``--log-file`` to f."""
        return log_level(log_file(f))

    @classmethod
    def pop_from_click_kwargs(cls, kwargs: dict) -> "LogOptions":
        return cls(
            log_level=kwargs.pop("log_level", "WARNING"),
            log_file=kwargs.pop("log_file", None),
        )

    def apply(self) -> None:
        setup_logging(self.log_level, self.log_file)


@dataclasses.dataclass(frozen=True)
class SelectionOptions:
    """Bundled bcftools-view-shaped selection options (regions / targets /
    samples / include / exclude / types / min-max-alleles).

    Renders as its own ``Selection options`` section in ``--help``. Consumed
    by multiple commands (view / view-plink / view-bgen) and by downstream
    tools like biofuse. Pair this dataclass with the :meth:`decorator` Click
    decorator on the same class.
    """

    regions: str | None = None
    regions_file: str | None = None
    targets: str | None = None
    targets_file: str | None = None
    samples: str | None = None
    samples_file: str | None = None
    force_samples: bool = False
    include: str | None = None
    exclude: str | None = None
    types: str | None = None
    exclude_types: str | None = None
    min_alleles: int | None = None
    max_alleles: int | None = None

    @staticmethod
    def decorator(f):
        """Click decorator: add the bcftools-view-shaped selection options
        to f.

        Order matches ``vcztools view --help`` so help-text rows stay
        identical between ``vcztools view`` and any consumer that applies
        this decorator.
        """
        decorators = [
            regions,
            regions_file,
            targets,
            targets_file,
            samples,
            samples_file,
            force_samples,
            include,
            exclude,
            types_opt,
            exclude_types_opt,
            min_alleles_opt,
            max_alleles_opt,
        ]
        for d in reversed(decorators):
            f = d(f)
        return f

    @classmethod
    def pop_from_click_kwargs(cls, kwargs: dict) -> "SelectionOptions":
        field_names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: kwargs.pop(k) for k in list(kwargs) if k in field_names})


@dataclasses.dataclass(frozen=True)
class ViewPlinkOptions:
    """Bundled options for ``view-plink``-shaped commands — every option
    except ``-o/--output``.

    Composes the four category groups (Selection / Zarr store / Reader /
    Log) plus the PLINK-specific sidecar flags ``--no-bim`` and
    ``--no-fam``. The output target is intentionally excluded so consumers
    can declare it themselves (e.g. biofuse ``mount-plink`` accepts a
    mount point rather than a file stem).
    """

    selection: SelectionOptions = dataclasses.field(default_factory=SelectionOptions)
    zarr_store: ZarrStoreOptions = dataclasses.field(default_factory=ZarrStoreOptions)
    reader: ReaderOptions = dataclasses.field(default_factory=ReaderOptions)
    log: LogOptions = dataclasses.field(default_factory=LogOptions)
    no_bim: bool = False
    no_fam: bool = False

    @staticmethod
    def decorator(f):
        """Click decorator: add every view-plink option except ``-o`` to f."""
        f = LogOptions.decorator(f)
        f = ReaderOptions.decorator(f)
        f = ZarrStoreOptions.decorator(f)
        f = SelectionOptions.decorator(f)
        f = click.option(
            "--no-fam",
            is_flag=True,
            default=False,
            help="Skip the .fam sample-info sidecar.",
        )(f)
        f = click.option(
            "--no-bim",
            is_flag=True,
            default=False,
            help="Skip the .bim variant-info sidecar.",
        )(f)
        return f

    @classmethod
    def pop_from_click_kwargs(cls, kwargs: dict) -> "ViewPlinkOptions":
        return cls(
            selection=SelectionOptions.pop_from_click_kwargs(kwargs),
            zarr_store=ZarrStoreOptions.pop_from_click_kwargs(kwargs),
            reader=ReaderOptions.pop_from_click_kwargs(kwargs),
            log=LogOptions.pop_from_click_kwargs(kwargs),
            no_bim=kwargs.pop("no_bim", False),
            no_fam=kwargs.pop("no_fam", False),
        )


@dataclasses.dataclass(frozen=True)
class ViewBgenOptions:
    """Bundled options for ``view-bgen``-shaped commands — every option
    except ``-o/--output``.

    Composes the four category groups (Selection / Zarr store / Reader /
    Log) plus the BGEN-specific sidecar and encoding flags. The output
    target is intentionally excluded so consumers can declare it
    themselves.
    """

    selection: SelectionOptions = dataclasses.field(default_factory=SelectionOptions)
    zarr_store: ZarrStoreOptions = dataclasses.field(default_factory=ZarrStoreOptions)
    reader: ReaderOptions = dataclasses.field(default_factory=ReaderOptions)
    log: LogOptions = dataclasses.field(default_factory=LogOptions)
    no_bgi: bool = False
    no_sample_file: bool = False
    no_header_samples: bool = False
    compression_level: int = 1

    @staticmethod
    def decorator(f):
        """Click decorator: add every view-bgen option except ``-o`` to f."""
        f = LogOptions.decorator(f)
        f = ReaderOptions.decorator(f)
        f = ZarrStoreOptions.decorator(f)
        f = SelectionOptions.decorator(f)
        f = click.option(
            "--compression-level",
            type=click.IntRange(-1, 9),
            default=1,
            show_default=True,
            help=(
                "zlib compression level for the BGEN genotype probability "
                "blocks. 1 (default) is fast and within ~10-30% of the size "
                "of level 6 on hard-call BGEN; -1 = zlib default (~6); 0 = "
                "stored (no compression); 9 = max."
            ),
        )(f)
        f = click.option(
            "--no-header-samples",
            is_flag=True,
            default=False,
            help=(
                "Omit sample IDs from the BGEN header (clears "
                "SAMPLE_IDS_PRESENT). Combine with --no-sample-file at your "
                "peril: most downstream tools require sample IDs from one "
                "source or the other."
            ),
        )(f)
        f = click.option(
            "--no-sample-file",
            is_flag=True,
            default=False,
            help=(
                "Skip the Oxford .sample sidecar. No effect when streaming "
                "to stdout. Distinct from -S/--samples-file, which filters "
                "input."
            ),
        )(f)
        f = click.option(
            "--no-bgi",
            is_flag=True,
            default=False,
            help=(
                "Skip the bgenix .bgen.bgi SQLite index sidecar. No effect "
                "when streaming to stdout."
            ),
        )(f)
        return f

    @classmethod
    def pop_from_click_kwargs(cls, kwargs: dict) -> "ViewBgenOptions":
        return cls(
            selection=SelectionOptions.pop_from_click_kwargs(kwargs),
            zarr_store=ZarrStoreOptions.pop_from_click_kwargs(kwargs),
            reader=ReaderOptions.pop_from_click_kwargs(kwargs),
            log=LogOptions.pop_from_click_kwargs(kwargs),
            no_bgi=kwargs.pop("no_bgi", False),
            no_sample_file=kwargs.pop("no_sample_file", False),
            no_header_samples=kwargs.pop("no_header_samples", False),
            compression_level=kwargs.pop("compression_level", 1),
        )


def make_reader_from_groups(
    path: str,
    *,
    selection: SelectionOptions | None = None,
    zarr_store: ZarrStoreOptions | None = None,
    reader: ReaderOptions | None = None,
    view_semantics: bool = False,
    drop_genotypes: bool = False,
) -> retrieval.VczReader:
    """Construct a :class:`VczReader` from the category-aligned option
    bundles. Single seam between the public dataclasses and
    :func:`make_reader`."""
    if selection is None:
        selection = SelectionOptions()
    if zarr_store is None:
        zarr_store = ZarrStoreOptions()
    if reader is None:
        reader = ReaderOptions()
    return make_reader(
        path,
        view_semantics=view_semantics,
        drop_genotypes=drop_genotypes,
        **dataclasses.asdict(selection),
        **dataclasses.asdict(zarr_store),
        **dataclasses.asdict(reader),
    )


class NaturalOrderGroup(click.Group):
    """
    List commands in the order they are provided in the help text.
    """

    def list_commands(self, ctx):
        return self.commands.keys()


@click.command(cls=GroupedCommand)
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
@ZarrStoreOptions.decorator
@LogOptions.decorator
@handle_exception
def index(path, nrecords, stats, **kwargs):
    """
    Query the number of records in a VCZ dataset. This subcommand only
    implements the --nrecords and --stats options and does not build any
    indexes.
    """
    LogOptions.pop_from_click_kwargs(kwargs).apply()
    zarr_store = ZarrStoreOptions.pop_from_click_kwargs(kwargs)
    assert kwargs == {}, kwargs
    if nrecords and stats:
        raise click.UsageError("Expected only one of --stats or --nrecords options")
    if not nrecords and not stats:
        raise click.UsageError("Building region indexes is not supported")
    root = open_zarr(
        path,
        mode="r",
        backend_storage=zarr_store.backend_storage,
        storage_options=zarr_store.storage_options,
    )
    with retrieval.VczReader(root) as reader:
        if nrecords:
            stats_module.nrecords(reader, sys.stdout)
        else:
            stats_module.stats(reader, sys.stdout)


@click.command(cls=GroupedCommand)
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
@ZarrStoreOptions.decorator
@ReaderOptions.decorator
@LogOptions.decorator
@handle_exception
def query(
    path,
    output,
    list_samples,
    format,
    disable_automatic_newline,
    **kwargs,
):
    """
    Transform VCZ into user-defined formats with efficient subsetting and
    filtering. Intended as a drop-in replacement for bcftools query, where we
    replace the VCF file path with a VCZ dataset URL.

    This is an early version and not feature complete: if you are missing a
    particular piece of functionality please open an issue at
    https://github.com/sgkit-dev/vcztools/issues
    """
    LogOptions.pop_from_click_kwargs(kwargs).apply()
    zarr_store = ZarrStoreOptions.pop_from_click_kwargs(kwargs)
    reader_opts = ReaderOptions.pop_from_click_kwargs(kwargs)
    selection = SelectionOptions.pop_from_click_kwargs(kwargs)
    assert kwargs == {}, kwargs
    if list_samples:
        # bcftools query -l ignores the --output option and always writes to stdout
        output = sys.stdout
        root = open_zarr(
            path,
            mode="r",
            backend_storage=zarr_store.backend_storage,
            storage_options=zarr_store.storage_options,
        )
        with retrieval.VczReader(root) as reader, handle_broken_pipe(output):
            query_module.list_samples(reader, output)
        return

    if format is None:
        raise click.UsageError("Missing option -f / --format")

    reader = make_reader_from_groups(
        path,
        selection=selection,
        zarr_store=zarr_store,
        reader=reader_opts,
    )
    with reader, handle_broken_pipe(output):
        query_module.write_query(
            reader,
            output,
            query_format=format,
            disable_automatic_newline=disable_automatic_newline,
        )


@click.command(cls=GroupedCommand)
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
@click.option(
    "-I",
    "--no-update",
    is_flag=True,
    help="Do not recalculate INFO fields for the sample subset.",
)
@click.option(
    "-G",
    "--drop-genotypes",
    is_flag=True,
    help="Drop genotypes.",
)
@SelectionOptions.decorator
@encode_threads
@ZarrStoreOptions.decorator
@ReaderOptions.decorator
@LogOptions.decorator
@handle_exception
def view(
    path,
    output,
    header_only,
    no_header,
    no_version,
    no_update,
    drop_genotypes,
    encode_threads,
    **kwargs,
):
    """
    Convert VCZ dataset to VCF with efficient subsetting and filtering.
    Intended as a drop-in replacement for bcftools view, where
    we replace the VCF file path with a VCZ dataset URL.

    This is an early version and not feature complete: if you are missing a
    particular piece of functionality please open an issue at
    https://github.com/sgkit-dev/vcztools/issues
    """
    LogOptions.pop_from_click_kwargs(kwargs).apply()
    zarr_store = ZarrStoreOptions.pop_from_click_kwargs(kwargs)
    reader_opts = ReaderOptions.pop_from_click_kwargs(kwargs)
    selection = SelectionOptions.pop_from_click_kwargs(kwargs)
    assert kwargs == {}, kwargs

    suffix = output.name.split(".")[-1]
    # Exclude suffixes which require bgzipped or BCF output:
    # https://github.com/samtools/htslib/blob/329e7943b7ba3f0af15b0eaa00a367a1ac15bd83/vcf.c#L3815
    if suffix in ["gz", "bcf", "bgz"]:
        raise ValueError(
            f"Only uncompressed VCF output supported, suffix .{suffix} not allowed"
        )

    if (selection.samples or selection.samples_file) and drop_genotypes:
        raise ValueError("Cannot select samples and drop genotypes.")

    reader = make_reader_from_groups(
        path,
        selection=selection,
        zarr_store=zarr_store,
        reader=reader_opts,
        view_semantics=True,
        drop_genotypes=drop_genotypes,
    )
    subsetting_samples = (
        selection.samples is not None
        or selection.samples_file is not None
        or drop_genotypes
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
            encode_threads=encode_threads,
        )


@click.command(cls=GroupedCommand)
@click.argument("path", type=click.Path())
@click.option(
    "-o",
    "--output",
    "output",
    required=True,
    type=str,
    help=(
        "Output stem; taken verbatim. -o foo writes foo.bed, plus foo.bim "
        "and foo.fam unless --no-bim/--no-fam is given."
    ),
)
@ViewPlinkOptions.decorator
@handle_exception
def view_plink(path, output, **kwargs):
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
    opts = ViewPlinkOptions.pop_from_click_kwargs(kwargs)
    assert kwargs == {}, kwargs
    opts.log.apply()
    reader = make_reader_from_groups(
        path,
        selection=opts.selection,
        zarr_store=opts.zarr_store,
        reader=opts.reader,
    )
    with reader:
        plink.write_plink(
            reader,
            output,
            write_bim=not opts.no_bim,
            write_fam=not opts.no_fam,
        )


@click.command(cls=GroupedCommand)
@click.argument("path", type=click.Path())
@click.option(
    "-o",
    "--output",
    "output",
    type=str,
    default=None,
    help=(
        "Output stem; taken verbatim. Absent: stream .bgen to stdout "
        "(no sidecars). Present: -o foo writes foo.bgen plus, by default, "
        "foo.bgen.bgi and foo.sample."
    ),
)
@ViewBgenOptions.decorator
@handle_exception
def view_bgen(path, output, **kwargs):
    """
    Generate Oxford BGEN output from a VCZ dataset.

    Default: stream the .bgen payload to stdout (symmetric with `view`).
    With -o STEM, write foo.bgen + foo.bgen.bgi + foo.sample; sidecars
    are individually suppressible (--no-bgi, --no-sample-file).

    Output profile: layout 2, zlib-compressed, 8 bits/probability,
    biallelic, diploid, embedded sample IDs. Hard calls in
    ``call_genotype`` are encoded as 1.0 probabilities (round-trips
    exactly at 8-bit). Phase is propagated per-variant from
    ``call_genotype_phased`` if present. Sample/region/filter selection
    mirrors bcftools view (-s/-S/-r/-R/-t/-T/-i/-e/-v/-V/-m/-M).
    Multi-allelic variants are rejected by default; pass ``-M 2`` (or
    ``--max-alleles 2``) to skip them.

    See the "BGEN output" documentation page for the full reference,
    including downstream-tool compatibility (REGENIE, SAIGE,
    BOLT-LMM, BGENIE, qctool, PLINK 2) and sidecar conventions.
    """
    opts = ViewBgenOptions.pop_from_click_kwargs(kwargs)
    assert kwargs == {}, kwargs
    opts.log.apply()
    embed_header_samples = not opts.no_header_samples
    if output is None:
        # Stream .bgen to stdout. Sidecars have no stem to derive paths
        # from, so they're omitted.
        bgen_dest = sys.stdout.buffer
        sample_path = None
        bgi_path = None
        pipe_context = handle_broken_pipe(sys.stdout.buffer)
    else:
        # Stem is taken verbatim — no suffix stripping. "foo" -> foo.bgen
        # / foo.sample / foo.bgen.bgi (the bgenix filename convention).
        out_stem = str(output)
        bgen_dest = pathlib.Path(out_stem + ".bgen")
        sample_path = (
            None if opts.no_sample_file else pathlib.Path(out_stem + ".sample")
        )
        bgi_path = None if opts.no_bgi else pathlib.Path(str(bgen_dest) + ".bgi")
        pipe_context = contextlib.nullcontext()
    reader = make_reader_from_groups(
        path,
        selection=opts.selection,
        zarr_store=opts.zarr_store,
        reader=opts.reader,
    )
    with reader, pipe_context:
        bgen.write_bgen(
            reader,
            bgen_dest,
            sample_path=sample_path,
            bgi_path=bgi_path,
            embed_header_samples=embed_header_samples,
            compression_level=opts.compression_level,
        )


@version
@click.group(cls=NaturalOrderGroup, name="vcztools")
def vcztools_main():
    pass


vcztools_main.add_command(index)
vcztools_main.add_command(query)
vcztools_main.add_command(view)
vcztools_main.add_command(view_plink)
vcztools_main.add_command(view_bgen)
