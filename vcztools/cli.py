import sys

import click

from . import query as query_module
from . import regions, stats, vcf_writer


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
def index(path, nrecords):
    if nrecords:
        stats.nrecords(path, sys.stdout)
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
    "-s",
    "--samples",
    type=str,
    default=None,
    help="Samples to include.",
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
    header_only,
    no_header,
    no_version,
    regions,
    targets,
    samples,
    include,
    exclude,
):
    vcf_writer.write_vcf(
        path,
        sys.stdout,
        header_only=header_only,
        no_header=no_header,
        variant_regions=regions,
        variant_targets=targets,
        samples=samples,
        include=include,
        exclude=exclude,
    )


@click.group(cls=NaturalOrderGroup, name="vcztools")
def vcztools_main():
    pass


vcztools_main.add_command(index)
vcztools_main.add_command(query)
vcztools_main.add_command(view)
