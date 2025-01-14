import contextlib
import os
import sys

import click

from . import query as query_module
from . import stats as stats_module
from . import vcf_writer


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


include = click.option(
    "-i", "--include", type=str, help="Filter expression to include variant sites."
)
exclude = click.option(
    "-e", "--exclude", type=str, help="Filter expression to exclude variant sites."
)
output = click.option(
    "-o",
    "--output",
    type=click.File("w"),
    default="-",
    help="File path to write output to (defaults to stdout '-').",
)


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
def index(path, nrecords, stats):
    if nrecords and stats:
        raise click.UsageError("Expected only one of --stats or --nrecords options")
    if nrecords:
        stats_module.nrecords(path, sys.stdout)
    elif stats:
        stats_module.stats(path, sys.stdout)
    else:
        raise click.UsageError("Building region indexes is not supported")


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
@include
@exclude
def query(path, output, list_samples, format, include, exclude):
    if list_samples:
        # bcftools query -l ignores the --output option and always writes to stdout
        output = sys.stdout
        with handle_broken_pipe(output):
            query_module.list_samples(path, output)
        return

    if format is None:
        raise click.UsageError("Missing option -f / --format")
    with handle_broken_pipe(output):
        query_module.write_query(
            path, output, query_format=format, include=include, exclude=exclude
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
@click.option(
    "-r",
    "--regions",
    type=str,
    default=None,
    help="Regions to include.",
)
# @click.option(
#     "--force-samples", is_flag=True, help="Only warn about unknown sample subsets."
# )
# @click.option(
#     "-I",
#     "--no-update",
#     is_flag=True,
#     help="Do not recalculate INFO fields for the sample subset.",
# )
# @click.option(
#     "-s",
#     "--samples",
#     type=str,
#     default=None,
#     help="Samples to include.",
# )
# @click.option(
#     "-S",
#     "--samples-file",
#     type=str,
#     default=None,
#     help="File of sample names to include.",
# )
@click.option(
    "-G",
    "--drop-genotypes",
    type=bool,
    is_flag=True,
    help="Drop genotypes.",
)
@click.option(
    "-t",
    "--targets",
    type=str,
    default=None,
    help="Target regions to include.",
)
@include
@exclude
def view(
    path,
    output,
    header_only,
    no_header,
    no_version,
    regions,
    targets,
    # force_samples,
    # no_update,
    # samples,
    # samples_file,
    drop_genotypes,
    include,
    exclude,
):
    suffix = output.name.split(".")[-1]
    # Exclude suffixes which require bgzipped or BCF output:
    # https://github.com/samtools/htslib/blob/329e7943b7ba3f0af15b0eaa00a367a1ac15bd83/vcf.c#L3815
    if suffix in ["gz", "bcf", "bgz"]:
        raise ValueError(
            f"Only uncompressed VCF output supported, suffix .{suffix} not allowed"
        )

    # Dropping implementation here until it's reimplemented after initial release:
    # https://github.com/sgkit-dev/vcztools/issues/121
    # if samples_file:
    #     assert not samples, "vcztools does not support combining -s and -S"

    #     samples = ""
    #     exclude_samples_file = samples_file.startswith("^")
    #     samples_file = samples_file.lstrip("^")

    #     with open(samples_file) as file:
    #         if exclude_samples_file:
    #             samples = "^" + samples
    #         samples += ",".join(line.strip() for line in file.readlines())

    with handle_broken_pipe(output):
        vcf_writer.write_vcf(
            path,
            output,
            header_only=header_only,
            no_header=no_header,
            no_version=no_version,
            variant_regions=regions,
            variant_targets=targets,
            # no_update=no_update,
            # samples=samples,
            drop_genotypes=drop_genotypes,
            include=include,
            exclude=exclude,
        )


@click.group(cls=NaturalOrderGroup, name="vcztools")
def vcztools_main():
    pass


vcztools_main.add_command(index)
vcztools_main.add_command(query)
vcztools_main.add_command(view)
