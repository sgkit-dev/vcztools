import functools

import pyparsing as pp
import zarr

from vcztools.utils import open_file_like


def list_samples(vcz_path, output=None):
    root = zarr.open(vcz_path, mode="r")

    with open_file_like(output) as output:
        sample_ids = root["sample_id"][:]
        print("\n".join(sample_ids), file=output)


class QueryFormatParser:
    def __init__(self):
        info_tag_pattern = pp.Combine(
            pp.Literal("%INFO/") + pp.Word(pp.srange("[A-Z]"))
        )
        tag_pattern = info_tag_pattern | pp.Combine(
            pp.Literal("%") + pp.Regex(r"[A-Z]+\d?")
        )
        subscript_pattern = pp.Group(
            tag_pattern
            + pp.Literal("{").suppress()
            + pp.common.integer
            + pp.Literal("}").suppress()
        )
        newline_pattern = pp.Literal("\\n").set_parse_action(pp.replace_with("\n"))
        tab_pattern = pp.Literal("\\t").set_parse_action(pp.replace_with("\t"))
        pattern = pp.ZeroOrMore(
            subscript_pattern
            | tag_pattern
            | newline_pattern
            | tab_pattern
            | pp.White()
            | pp.Word(pp.printables, exclude_chars=r"\{}[]%")
        )
        pattern = pattern.leave_whitespace()

        self._parser = functools.partial(pattern.parse_string, parse_all=True)

    def __call__(self, *args, **kwargs):
        assert len(args) == 1
        assert not kwargs

        return self._parser(args[0])
