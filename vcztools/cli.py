import sys

import click

from . import regions, vcf_writer


class NaturalOrderGroup(click.Group):
    """
    List commands in the order they are provided in the help text.
    """

    def list_commands(self, ctx):
        return self.commands.keys()


@click.command
@click.argument("path", type=click.Path())
def index(path):
    regions.create_index(path)


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
def view(path, header_only, no_header, no_version, regions, targets, samples):
    vcf_writer.write_vcf(
        path,
        sys.stdout,
        header_only=header_only,
        no_header=no_header,
        variant_regions=regions,
        variant_targets=targets,
        samples=samples,
    )


@click.group(cls=NaturalOrderGroup, name="vcztools")
def vcztools_main():
    pass


vcztools_main.add_command(index)
vcztools_main.add_command(view)
