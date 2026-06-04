"""Sphinx extension that reproduces vcztools' grouped ``--help`` sections.

sphinx-click renders every option of a command under a single ``Options``
rubric, ignoring the command's ``format_options``. vcztools groups options into
labelled sections (Selection options, Zarr store options, ...) via the
``help_group`` attribute on its options. This extension hooks the
``sphinx-click-process-options`` event and re-groups the generated option lines
to match, inserting a rubric before each non-default group.
"""

import click

from vcztools import cli


def _group_options(app, ctx, lines):
    options = [
        param
        for param in ctx.command.params
        if isinstance(param, click.Option) and not getattr(param, "hidden", False)
    ]

    block_starts = [i for i, line in enumerate(lines) if line.startswith(".. option::")]
    if len(block_starts) != len(options):
        return

    block_bounds = block_starts + [len(lines)]
    blocks = {group: [] for group in cli.OptionGroup}
    for index, option in enumerate(options):
        group = getattr(option, "help_group", cli.OptionGroup.DEFAULT)
        block = lines[block_bounds[index] : block_bounds[index + 1]]
        blocks[group].extend(block)

    rebuilt = list(blocks[cli.OptionGroup.DEFAULT])
    for group in cli.OptionGroup:
        if group == cli.OptionGroup.DEFAULT:
            continue
        group_blocks = blocks[group]
        if len(group_blocks) == 0:
            continue
        rebuilt.append(f".. rubric:: {group}")
        rebuilt.append("")
        rebuilt.extend(group_blocks)

    lines[:] = rebuilt


def setup(app):
    app.connect("sphinx-click-process-options", _group_options)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
