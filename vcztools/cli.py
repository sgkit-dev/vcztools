import sys

import click

from . import query as query_module
from . import regions, vcf_writer
from . import stats as stats_module


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
    if nrecords:
        stats_module.nrecords(path, sys.stdout)
    elif stats:
        stats_module.stats(path, sys.stdout)
    else:
        regions.create_index(path)


@click.command
@click.argument("path", type=click.Path())
@click.option(
    "-l",
    "--list-samples",
    is_flag=True,
    help="List the sample IDs and exit.",
)
@click.option("-f", "--format", type=str, help="The format of the output.")
def query(path, list_samples, format):
    if list_samples:
        query_module.list_samples(path)
        return

    query_module.write_query(path, query_format=format)


@click.command
@click.argument("path", type=click.Path())
@click.option(
    "-o", "--output", type=str, default=None, help="File path to write output to."
)
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
@click.option(
    "--force-samples", is_flag=True, help="Only warn about unknown sample subsets."
)
@click.option(
    "-I",
    "--no-update",
    is_flag=True,
    help="Do not recalculate INFO fields for the sample subset.",
)
@click.option(
    "-s",
    "--samples",
    type=str,
    default=None,
    help="Samples to include.",
)
@click.option(
    "-S",
    "--samples-file",
    type=str,
    default=None,
    help="File of sample names to include.",
)
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
@click.option(
    "-i", "--include", type=str, help="Filter expression to include variant sites."
)
@click.option(
    "-e", "--exclude", type=str, help="Filter expression to exclude variant sites."
)
def view(
    path,
    output,
    header_only,
    no_header,
    no_version,
    regions,
    targets,
    force_samples,
    no_update,
    samples,
    samples_file,
    drop_genotypes,
    include,
    exclude,
):
    if output and not output.endswith(".vcf"):
        split = output.split(".")
        raise ValueError(f"Output file extension must be .vcf, got: .{split[-1]}")

    if samples_file:
        assert not samples, "vcztools does not support combining -s and -S"

        samples = ""
        exclude_samples_file = samples_file.startswith("^")
        samples_file = samples_file.lstrip("^")

        with open(samples_file) as file:
            if exclude_samples_file:
                samples = "^" + samples
            samples += ",".join(line.strip() for line in file.readlines())

    # TODO: use no_update when fixing https://github.com/sgkit-dev/vcztools/issues/75

    vcf_writer.write_vcf(
        path,
        output,
        header_only=header_only,
        no_header=no_header,
        no_version=no_version,
        variant_regions=regions,
        variant_targets=targets,
        no_update=no_update,
        samples=samples,
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
