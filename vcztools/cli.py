import contextlib
import dataclasses
import enum
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
from . import stats as stats_module
from . import variant_filter as variant_filter_mod
from .utils import open_zarr


def _read_samples_file(path: str) -> list[str]:
    """Read a samples file (one sample ID per line) into a list.

    Blank lines are ignored. The file is decoded as UTF-8 regardless of
    the platform locale, matching the VCF 4.3 spec's encoding for
    sample IDs.
    """
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


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


class OptionGroup(enum.StrEnum):
    """Help-text section labels for grouped ``--help`` rendering.

    ``DEFAULT`` is the bucket for options without an explicit ``help_group``
    (category 4 — command-specific). The other four members form the long-term
    stable CLI groupings exposed to biofuse and other downstream consumers.
    Declaration order is the rendering order used by
    :meth:`GroupedCommand.format_options`.

    The members compare equal to their display strings and ``str(member)``
    returns the display string, so they can be passed straight to
    ``click.formatter.section(...)``.
    """

    DEFAULT = "Options"
    SELECTION = "Selection options"
    ZARR_STORE = "Zarr store options"
    READER = "Reader options"
    LOGGING = "Logging options"


class GroupedOption(click.Option):
    """A click Option tagged with the help-text section it belongs to."""

    def __init__(self, *args, help_group=OptionGroup.DEFAULT, **kwargs):
        super().__init__(*args, **kwargs)
        self.help_group = help_group


class GroupedCommand(click.Command):
    """A click Command that renders options grouped by ``help_group``.

    Public API surface for downstream consumers (e.g. biofuse) that mount
    :meth:`ViewBgenOptions.decorator` / :meth:`ViewPlinkOptions.decorator` on
    their own commands. Pass ``cls=GroupedCommand`` to ``@click.command(...)``
    to get the same grouped ``--help`` sections (Selection options / Zarr
    store options / Reader options / Logging options) as ``vcztools view-bgen
    --help`` and ``vcztools view-plink --help``.
    """

    def format_options(self, ctx, formatter):
        buckets: dict[str, list] = {g: [] for g in OptionGroup}
        for param in self.get_params(ctx):
            record = param.get_help_record(ctx)
            if record is None:
                continue
            group = getattr(param, "help_group", OptionGroup.DEFAULT)
            buckets.setdefault(group, []).append(record)
        for group in OptionGroup:
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
    help_group=OptionGroup.SELECTION,
)
exclude = click.option(
    "-e",
    "--exclude",
    type=str,
    help="Filter expression to exclude variant sites.",
    cls=GroupedOption,
    help_group=OptionGroup.SELECTION,
)
force_samples = click.option(
    "--force-samples",
    is_flag=True,
    help="Only warn about unknown sample subsets.",
    cls=GroupedOption,
    help_group=OptionGroup.SELECTION,
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
    help_group=OptionGroup.SELECTION,
)
regions_file = click.option(
    "-R",
    "--regions-file",
    type=str,
    default=None,
    help="File of regions to include.",
    cls=GroupedOption,
    help_group=OptionGroup.SELECTION,
)
samples = click.option(
    "-s",
    "--samples",
    type=str,
    default=None,
    help="Samples to include.",
    cls=GroupedOption,
    help_group=OptionGroup.SELECTION,
)
samples_file = click.option(
    "-S",
    "--samples-file",
    type=str,
    default=None,
    help="File of sample names to include.",
    cls=GroupedOption,
    help_group=OptionGroup.SELECTION,
)
targets = click.option(
    "-t",
    "--targets",
    type=str,
    default=None,
    help="Target regions to include.",
    cls=GroupedOption,
    help_group=OptionGroup.SELECTION,
)
targets_file = click.option(
    "-T",
    "--targets-file",
    type=str,
    default=None,
    help="File of target regions to include.",
    cls=GroupedOption,
    help_group=OptionGroup.SELECTION,
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
    help_group=OptionGroup.SELECTION,
)
exclude_types_opt = click.option(
    "-V",
    "--exclude-types",
    type=str,
    default=None,
    help="Comma-separated list of variant types to exclude.",
    cls=GroupedOption,
    help_group=OptionGroup.SELECTION,
)
min_alleles_opt = click.option(
    "-m",
    "--min-alleles",
    type=int,
    default=None,
    help="Print sites with at least INT alleles listed in REF and ALT.",
    cls=GroupedOption,
    help_group=OptionGroup.SELECTION,
)
max_alleles_opt = click.option(
    "-M",
    "--max-alleles",
    type=int,
    default=None,
    help="Print sites with at most INT alleles listed in REF and ALT.",
    cls=GroupedOption,
    help_group=OptionGroup.SELECTION,
)
version = click.version_option(version=f"{provenance.__version__}")

log_level = click.option(
    "--log-level",
    type=click.Choice(["WARNING", "INFO", "DEBUG"], case_sensitive=False),
    default="WARNING",
    show_default=True,
    help="Logging verbosity.",
    cls=GroupedOption,
    help_group=OptionGroup.LOGGING,
)
log_file = click.option(
    "--log-file",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Write log output to FILE instead of stderr.",
    cls=GroupedOption,
    help_group=OptionGroup.LOGGING,
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
    help_group=OptionGroup.READER,
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
    help_group=OptionGroup.READER,
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
    help_group=OptionGroup.ZARR_STORE,
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
    help_group=OptionGroup.ZARR_STORE,
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


# Tag-name surface for ``vcztools view --fill-tags``. Matches the
# ``bcftools +fill-tags -t`` accept-list filtered to entries with a
# corresponding virtual field in
# :data:`vcztools.virtual_fields.REGISTRY`. ``MAF``, ``AC_Hom``,
# ``AC_Het``, ``AC_Hemi`` are bcftools-supported but not yet wired in
# vcztools: they parse as known names and surface a dedicated "not
# yet implemented" error so the help text stays honest.
SUPPORTED_FILL_TAGS = frozenset({"AC", "AN", "AF", "NS"})
_DEFERRED_FILL_TAGS = frozenset({"MAF", "AC_Hom", "AC_Het", "AC_Hemi"})
_FILL_TAGS_ISSUE_URL = "https://github.com/sgkit-dev/vcztools/issues/171"
# Tags recomputed by default when ``vcztools view`` is given a sample
# subset and ``--no-update`` / ``--fill-tags`` are absent. Matches the
# AC/AN recompute that ``bcftools view -s`` does implicitly.
_DEFAULT_SUBSET_FILL_TAGS = frozenset({"AC", "AN"})


def _parse_fill_tags_option(value):
    """Validate ``--fill-tags VALUE`` and return the
    ``frozenset[str]`` of accepted tag names. Empty value, unknown
    tag, or a deferred tag each raises ``ValueError`` at parse time
    with a message that names the supported set."""
    if value is None:
        return None
    if value == "":
        raise ValueError(
            "--fill-tags requires a non-empty comma-separated tag list; "
            f"supported tags: {', '.join(sorted(SUPPORTED_FILL_TAGS))}."
        )
    tags = [t for t in value.split(",") if len(t) > 0]
    if len(tags) == 0:
        raise ValueError(
            "--fill-tags requires a non-empty comma-separated tag list; "
            f"supported tags: {', '.join(sorted(SUPPORTED_FILL_TAGS))}."
        )
    deferred = [t for t in tags if t in _DEFERRED_FILL_TAGS]
    if len(deferred) > 0:
        raise ValueError(
            f"--fill-tags tag(s) {deferred} not yet implemented; "
            f"track at {_FILL_TAGS_ISSUE_URL}."
        )
    unknown = [t for t in tags if t not in SUPPORTED_FILL_TAGS]
    if len(unknown) > 0:
        raise ValueError(
            f"--fill-tags: unsupported tag(s) {unknown}. "
            f"Supported: {', '.join(sorted(SUPPORTED_FILL_TAGS))}."
        )
    return frozenset(tags)


def _resolve_fill_tags(
    *,
    fill_tags: frozenset | None,
    no_update: bool,
    drop_genotypes: bool,
    samples_subset: bool,
) -> frozenset:
    """Return the set of *VCF-tag* names whose values should be
    recomputed and emitted by the writer for this ``view`` invocation.

    Three regimes, picked by exactly one of the flags:

    - ``--no-update``: nothing recomputed.
    - ``--fill-tags=X,Y``: those tags (already validated).
    - Default with a sample subset in play: ``AC``/``AN``.

    The caller has already enforced mutex constraints (``--no-update``
    vs ``--fill-tags``, ``--fill-tags`` vs ``--drop-genotypes``) at
    Click-validation time, so this is pure resolution logic.
    """
    if no_update:
        return frozenset()
    if fill_tags is not None:
        return fill_tags
    if samples_subset and not drop_genotypes:
        return _DEFAULT_SUBSET_FILL_TAGS
    return frozenset()


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
    bcftools_semantics=False,
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
        samples = _read_samples_file(samples_file)
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
            reader, include=include, exclude=exclude
        )

    view_expr = _build_view_filter_expression(
        types, exclude_types, min_alleles, max_alleles
    )
    if view_expr is not None:
        synthetic_filter = bcftools_filter.BcftoolsFilter(reader, include=view_expr)
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
        reader.set_samples(
            samples,
            complement=samples_complement,
            ignore_missing_samples=force_samples,
        )
        if len(reader.sample_ids) == 0:
            raise ValueError(
                "Empty sample set is not supported. The bcftools semantics "
                "for this corner case (AC/AN recomputed over the full "
                "non-null sample set while emitting zero sample columns) "
                "would require significant internal complexity. "
                "If you need this feature, please open an issue at "
                "https://github.com/sgkit-dev/vcztools/issues."
            )

    if regions is not None or targets is not None:
        variant_chunk_plan = regions_mod.build_chunk_plan(
            reader,
            regions=regions,
            targets=targets,
            targets_complement=targets_complement,
        )
        reader.set_variants(variant_chunk_plan)

    # Native subset-first semantics are the default (filters evaluate
    # over the selected sample subset). ``view`` and ``query`` opt into
    # bcftools filter / INFO semantics (filters read the stored INFO) by
    # passing bcftools_semantics=True; ``view`` additionally sets
    # full_sample_filter so sample-scope filters use the full pre-subset
    # sample axis. The export commands (view-plink / view-bgen) keep the
    # native default.
    if bcftools_semantics:
        reader.set_bcftools_semantics(full_sample_filter=view_semantics)

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
    def from_click_kwargs(cls, kwargs: dict) -> "ZarrStoreOptions":
        """Read the Zarr store fields from ``kwargs`` non-destructively.

        Click's repeatable ``KEY=VALUE`` tuple is parsed to a ``dict | None``
        at this seam so consumers always see the resolved dict.
        """
        raw_pairs = kwargs.get("storage_options", ())
        return cls(
            backend_storage=kwargs.get("backend_storage"),
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
    def from_click_kwargs(cls, kwargs: dict) -> "ReaderOptions":
        return cls(
            readahead_workers=kwargs.get("readahead_workers"),
            readahead_bytes=kwargs.get("readahead_bytes"),
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
    def from_click_kwargs(cls, kwargs: dict) -> "LogOptions":
        return cls(
            log_level=kwargs.get("log_level", "WARNING"),
            log_file=kwargs.get("log_file"),
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
    def from_click_kwargs(cls, kwargs: dict) -> "SelectionOptions":
        field_names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: kwargs[k] for k in kwargs if k in field_names})


@dataclasses.dataclass(frozen=True)
class ViewPlinkOptions:
    """Public API bundle of every ``view-plink`` option except
    ``-o/--output``.

    Designed for downstream consumers (e.g. biofuse) that re-expose
    ``vcztools view-plink`` under a different output model — a FUSE
    mount, a generated stem, etc. — and therefore want every
    selection / sidecar flag without inheriting the file-output
    contract that ``-o`` implies.

    Stable public surface:

    * The dataclass fields: ``selection`` / ``zarr_store`` / ``reader``
      / ``log`` sub-bundles plus the PLINK-specific sidecar flags
      ``no_bim`` and ``no_fam``. Adding new fields is non-breaking
      (they ship with defaults); renaming or removing existing fields
      is breaking.
    * :meth:`decorator` — Click decorator that attaches every bundled
      option to a command function. Help-text order matches
      ``vcztools view-plink --help``.
    * :meth:`from_click_kwargs` — constructs the bundle from a Click
      handler's ``**kwargs`` dict without mutating it.
    * :meth:`make_reader` — opens ``path`` as a :class:`VczReader`
      configured from the ``selection`` / ``zarr_store`` / ``reader``
      fields.

    Typical use::

        from vcztools import GroupedCommand, ViewPlinkOptions

        @click.command(cls=GroupedCommand)
        @click.argument("path")
        @click.option("--mount-point")
        @ViewPlinkOptions.decorator
        def mount_plink(path, mount_point, **kwargs):
            opts = ViewPlinkOptions.from_click_kwargs(kwargs)
            opts.log.apply()
            with opts.make_reader(path) as reader:
                ...  # use mount_point + reader
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
    def from_click_kwargs(cls, kwargs: dict) -> "ViewPlinkOptions":
        return cls(
            selection=SelectionOptions.from_click_kwargs(kwargs),
            zarr_store=ZarrStoreOptions.from_click_kwargs(kwargs),
            reader=ReaderOptions.from_click_kwargs(kwargs),
            log=LogOptions.from_click_kwargs(kwargs),
            no_bim=kwargs.get("no_bim", False),
            no_fam=kwargs.get("no_fam", False),
        )

    def make_reader(self, path: str) -> retrieval.VczReader:
        """Open ``path`` as a :class:`VczReader` configured with this
        bundle's Selection / Zarr store / Reader options.

        Uses native subset-first semantics (``bcftools_semantics=False``):
        ``-i/-e`` filters evaluate over the selected sample subset, unlike
        ``view`` / ``query``. See the command help for the rationale."""
        return make_reader_from_groups(
            path,
            selection=self.selection,
            zarr_store=self.zarr_store,
            reader=self.reader,
            bcftools_semantics=False,
        )


def _pad_byte_callback(ctx, param, value):
    """Click callback for ``--pad-byte``: convert the user's ASCII
    character to a single-byte ``bytes`` value (or ``None`` if the
    option was not supplied)."""
    if value is None:
        return None
    encoded = value.encode("ascii", errors="strict")
    if len(encoded) != 1:
        raise click.BadParameter(
            f"--pad-byte must be a single ASCII character (got {value!r})"
        )
    return encoded


@dataclasses.dataclass(frozen=True)
class ViewBgenOptions:
    """Public API bundle of every ``view-bgen`` option except
    ``-o/--output``.

    Designed for downstream consumers (e.g. biofuse) that re-expose
    ``vcztools view-bgen`` under a different output model — a FUSE
    mount, a generated stem, etc. — and therefore want every
    selection / encoding / sidecar flag without inheriting the
    file-output contract that ``-o`` implies.

    Stable public surface:

    * The dataclass fields: ``selection`` / ``zarr_store`` / ``reader``
      / ``log`` sub-bundles plus the BGEN-specific sidecar flags
      ``no_bgi``, ``no_sample_file``, ``no_header_samples``. Adding new
      fields is non-breaking (they ship with defaults); renaming or
      removing existing fields is breaking.
    * :meth:`decorator` — Click decorator that attaches every bundled
      option to a command function. Help-text order matches
      ``vcztools view-bgen --help``.
    * :meth:`from_click_kwargs` — constructs the bundle from a Click
      handler's ``**kwargs`` dict without mutating it.
    * :meth:`make_reader` — opens ``path`` as a :class:`VczReader`
      configured from the ``selection`` / ``zarr_store`` / ``reader``
      fields.

    Typical use::

        from vcztools import GroupedCommand, ViewBgenOptions

        @click.command(cls=GroupedCommand)
        @click.argument("path")
        @click.option("--mount-point")
        @ViewBgenOptions.decorator
        def mount_bgen(path, mount_point, **kwargs):
            opts = ViewBgenOptions.from_click_kwargs(kwargs)
            opts.log.apply()
            with opts.make_reader(path) as reader:
                ...  # use mount_point + reader
    """

    selection: SelectionOptions = dataclasses.field(default_factory=SelectionOptions)
    zarr_store: ZarrStoreOptions = dataclasses.field(default_factory=ZarrStoreOptions)
    reader: ReaderOptions = dataclasses.field(default_factory=ReaderOptions)
    log: LogOptions = dataclasses.field(default_factory=LogOptions)
    no_bgi: bool = False
    no_sample_file: bool = False
    no_header_samples: bool = False
    unphased: bool = False
    variant_id_field: str = "rsid"
    total_string_length: int | None = None
    pad_byte: bytes | None = None

    @staticmethod
    def decorator(f):
        """Click decorator: add every view-bgen option except ``-o``,
        ``--compression-level``, and ``--fixed-variant-size`` to f."""
        f = LogOptions.decorator(f)
        f = ReaderOptions.decorator(f)
        f = ZarrStoreOptions.decorator(f)
        f = SelectionOptions.decorator(f)
        f = click.option(
            "--variant-id-field",
            type=click.Choice(["rsid", "varid"]),
            default="rsid",
            show_default=True,
            help=(
                "Which BGEN slot carries the zarr variant_id. The other "
                "slot is written as a literal '.' for every variant."
            ),
        )(f)
        f = click.option(
            "--total-string-length",
            type=int,
            default=None,
            help=("Combined byte budget (default 64) for the five BGEN string slots."),
        )(f)
        f = click.option(
            "--pad-byte",
            type=str,
            default=None,
            callback=_pad_byte_callback,
            help=(
                "Single ASCII char used to fill the padding slot beyond "
                "its leading '.'. Default: '.'."
            ),
        )(f)
        f = click.option(
            "--unphased",
            is_flag=True,
            default=False,
            help=(
                "Force every variant's phased flag to 0, ignoring "
                "call_genotype_phased. Required for qctool -snp-stats, "
                "which rejects per-haplotype-per-allele probabilities."
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
                "Skip the Oxford .sample sidecar. Distinct from "
                "-S/--samples-file, which filters input."
            ),
        )(f)
        f = click.option(
            "--no-bgi",
            is_flag=True,
            default=False,
            help=("Skip the bgenix .bgen.bgi SQLite index sidecar."),
        )(f)
        return f

    @classmethod
    def from_click_kwargs(cls, kwargs: dict) -> "ViewBgenOptions":
        return cls(
            selection=SelectionOptions.from_click_kwargs(kwargs),
            zarr_store=ZarrStoreOptions.from_click_kwargs(kwargs),
            reader=ReaderOptions.from_click_kwargs(kwargs),
            log=LogOptions.from_click_kwargs(kwargs),
            no_bgi=kwargs.get("no_bgi", False),
            no_sample_file=kwargs.get("no_sample_file", False),
            no_header_samples=kwargs.get("no_header_samples", False),
            unphased=kwargs.get("unphased", False),
            variant_id_field=kwargs.get("variant_id_field", "rsid"),
            total_string_length=kwargs.get("total_string_length"),
            pad_byte=kwargs.get("pad_byte"),
        )

    def make_reader(self, path: str) -> retrieval.VczReader:
        """Open ``path`` as a :class:`VczReader` configured with this
        bundle's Selection / Zarr store / Reader options.

        Uses native subset-first semantics (``bcftools_semantics=False``):
        ``-i/-e`` filters evaluate over the selected sample subset, unlike
        ``view`` / ``query``. See the command help for the rationale."""
        return make_reader_from_groups(
            path,
            selection=self.selection,
            zarr_store=self.zarr_store,
            reader=self.reader,
            bcftools_semantics=False,
        )


def make_reader_from_groups(
    path: str,
    *,
    selection: SelectionOptions | None = None,
    zarr_store: ZarrStoreOptions | None = None,
    reader: ReaderOptions | None = None,
    view_semantics: bool = False,
    bcftools_semantics: bool = False,
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
        bcftools_semantics=bcftools_semantics,
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
    LogOptions.from_click_kwargs(kwargs).apply()
    zarr_store = ZarrStoreOptions.from_click_kwargs(kwargs)
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
    LogOptions.from_click_kwargs(kwargs).apply()
    zarr_store = ZarrStoreOptions.from_click_kwargs(kwargs)
    reader_opts = ReaderOptions.from_click_kwargs(kwargs)
    selection = SelectionOptions.from_click_kwargs(kwargs)
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
            print("\n".join(reader.sample_ids), file=output)
        return

    if format is None:
        raise click.UsageError("Missing option -f / --format")

    reader = make_reader_from_groups(
        path,
        selection=selection,
        zarr_store=zarr_store,
        reader=reader_opts,
        bcftools_semantics=True,
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
    help="Do not recalculate INFO fields for the sample subset. "
    "Mutually exclusive with --fill-tags.",
)
@click.option(
    "-G",
    "--drop-genotypes",
    is_flag=True,
    help="Drop genotypes.",
)
@click.option(
    "--fill-tags",
    "fill_tags",
    type=str,
    default=None,
    help=(
        "Comma-separated list of INFO tags to (re)compute and emit, "
        "replacing any source value. Supported: "
        f"{', '.join(sorted(SUPPORTED_FILL_TAGS))}. "
        "Mutually exclusive with --no-update and -G."
    ),
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
    fill_tags,
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
    LogOptions.from_click_kwargs(kwargs).apply()
    zarr_store = ZarrStoreOptions.from_click_kwargs(kwargs)
    reader_opts = ReaderOptions.from_click_kwargs(kwargs)
    selection = SelectionOptions.from_click_kwargs(kwargs)

    suffix = output.name.split(".")[-1]
    # Exclude suffixes which require bgzipped or BCF output:
    # https://github.com/samtools/htslib/blob/329e7943b7ba3f0af15b0eaa00a367a1ac15bd83/vcf.c#L3815
    if suffix in ["gz", "bcf", "bgz"]:
        raise ValueError(
            f"Only uncompressed VCF output supported, suffix .{suffix} not allowed"
        )

    if (selection.samples or selection.samples_file) and drop_genotypes:
        raise ValueError("Cannot select samples and drop genotypes.")

    fill_tags_set = _parse_fill_tags_option(fill_tags)
    if fill_tags_set is not None:
        if no_update:
            raise ValueError("Cannot combine --no-update and --fill-tags.")
        if drop_genotypes:
            raise ValueError(
                "Cannot combine -G/--drop-genotypes and --fill-tags: "
                "tag computation requires call_genotype."
            )

    samples_subset = selection.samples is not None or selection.samples_file is not None
    effective_fill_tags = _resolve_fill_tags(
        fill_tags=fill_tags_set,
        no_update=no_update,
        drop_genotypes=drop_genotypes,
        samples_subset=samples_subset,
    )

    reader = make_reader_from_groups(
        path,
        selection=selection,
        zarr_store=zarr_store,
        reader=reader_opts,
        view_semantics=True,
        bcftools_semantics=True,
        drop_genotypes=drop_genotypes,
    )
    with reader, handle_broken_pipe(output):
        vcf_writer.write_vcf(
            reader,
            output,
            header_only=header_only,
            no_header=no_header,
            no_version=no_version,
            drop_genotypes=drop_genotypes,
            encode_threads=encode_threads,
            fill_tags=effective_fill_tags,
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
    variants. Sample/region selection mirrors bcftools view
    (-s/-S/-r/-R/-t/-T/-v/-V/-m/-M). Multi-allelic variants are
    rejected by default; pass ``-M 2`` (or ``--max-alleles 2``) to
    skip them, matching ``plink2 --vcf X --max-alleles 2 --make-bed``.

    Unlike ``vcztools view`` / ``query``, ``-i/-e`` filters are
    evaluated over the *selected* samples: with a ``-s``/``-S`` subset,
    AC/AN/AF/NS in a filter expression are recomputed for that subset
    rather than read from the file's stored (full-cohort) INFO.

    See the "PLINK 1" documentation page for details.
    """
    opts = ViewPlinkOptions.from_click_kwargs(kwargs)
    opts.log.apply()
    reader = opts.make_reader(path)
    with reader:
        reader.materialise_variant_filter()
        plink.write_plink(
            reader,
            output,
            bim=not opts.no_bim,
            fam=not opts.no_fam,
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
@click.option(
    "--compression-level",
    type=click.IntRange(-1, 9),
    default=None,
    show_default=False,
    help=(
        "zlib compression level for the BGEN genotype probability "
        "blocks. Default: 1 for variable-size output; required to be 0 "
        "when --fixed-variant-size is set. 1 is fast and within ~10-30% "
        "of the size of level 6 on hard-call BGEN; -1 = zlib default "
        "(~6); 0 = stored (no compression); 9 = max."
    ),
)
@click.option(
    "--fixed-variant-size",
    is_flag=True,
    default=False,
    help=(
        "Produce a fixed-stride BGEN where every variant block is "
        "exactly the same number of bytes wide (random-access, "
        "BgenEncoder path). Requires --compression-level=0 (default in "
        "this mode) and uniform ploidy across the store. The "
        "--total-string-length and --pad-byte options feed the "
        "fixed-stride encoder's per-variant string-slot budget and the "
        "filler byte for the padding slot beyond its leading '.'; they "
        "are inert without --fixed-variant-size. See the 'Fixed-size "
        "random-access encoding' docs section."
    ),
)
@ViewBgenOptions.decorator
@handle_exception
def view_bgen(
    path,
    output,
    compression_level,
    fixed_variant_size,
    **kwargs,
):
    """
    Generate BGEN output from a VCZ dataset.

    By default stream the .bgen payload to stdout (symmetric with `view`).
    With -o STEM, write foo.bgen + foo.bgen.bgi + foo.sample; sidecars
    are individually suppressible (--no-bgi, --no-sample-file).

    Phase is propagated per-variant from ``call_genotype_phased`` if present;
    pass ``--unphased`` to force every variant unphased . Sample/region
    selection mirrors bcftools view (-s/-S/-r/-R/-t/-T/-v/-V/-m/-M).
    Multi-allelic variants are rejected by default; pass ``-M 2`` (or
    ``--max-alleles 2``) to skip them.

    Unlike ``vcztools view`` / ``query``, ``-i/-e`` filters are
    evaluated over the *selected* samples: with a ``-s``/``-S`` subset,
    AC/AN/AF/NS in a filter expression are recomputed for that subset
    rather than read from the file's stored (full-cohort) INFO.

    See the "BGEN" documentation page for details.
    """
    opts = ViewBgenOptions.from_click_kwargs(kwargs)
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
    reader = opts.make_reader(path)
    with reader, pipe_context:
        reader.materialise_variant_filter()
        bgen.write_bgen(
            reader,
            bgen_dest,
            sample_path=sample_path,
            bgi_path=bgi_path,
            embed_header_samples=embed_header_samples,
            compression_level=compression_level,
            unphased=opts.unphased,
            variant_id_field=opts.variant_id_field,
            fixed_variant_size=fixed_variant_size,
            total_string_length=opts.total_string_length,
            pad_byte=opts.pad_byte,
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
