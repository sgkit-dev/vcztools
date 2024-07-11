import sys

import click

from . import vcf_writer



class NaturalOrderGroup(click.Group):
    """
    List commands in the order they are provided in the help text.
    """

    def list_commands(self, ctx):
        return self.commands.keys()


@click.command
@click.argument("path", type=click.Path())
@click.option("-c", is_flag=True, default=False, help="Use C implementation")
@click.option(
    "-t",
    "--targets",
    type=str,
    default=None,
    help="Target regions to include.",
)
def view(path, c, targets):
    implementation = "c" if c else "numba"
    vcf_writer.write_vcf(path, sys.stdout, variant_targets=targets, implementation=implementation)


@click.group(cls=NaturalOrderGroup, name="vcztools")
def vcztools_main():
    pass


vcztools_main.add_command(view)
