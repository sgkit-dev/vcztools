"""Subprocess wrappers shared by validation tests.

The ``run_view_*`` helpers invoke ``vcztools`` via ``CliRunner`` (no
shell) so failures surface inside pytest. The ``run_tool`` helper
invokes external binaries via subprocess; on non-zero exit it raises
``AssertionError`` with the full command line and stderr so failure
output is debuggable.
"""

from __future__ import annotations

import dataclasses
import pathlib
import subprocess

import click.testing as ct

import vcztools.cli as cli


@dataclasses.dataclass
class ToolResult:
    returncode: int
    stdout: str
    stderr: str


def run_tool(cmd: list[str], *, check: bool = True) -> ToolResult:
    """Run an external tool. On non-zero exit, raise AssertionError
    with the failing command and full stderr."""
    completed = subprocess.run(cmd, capture_output=True, check=False, text=True)
    if check and completed.returncode != 0:
        raise AssertionError(
            f"command failed with exit code {completed.returncode}\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return ToolResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def run_tool_capture_binary(cmd: list[str], stdout_path: pathlib.Path) -> ToolResult:
    """Run an external tool that writes binary data on stdout (e.g.
    bgenix subset to a new BGEN), capturing stdout directly into the
    file at ``stdout_path``. Asserts exit 0."""
    with open(stdout_path, "wb") as fh:
        completed = subprocess.run(cmd, stdout=fh, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0:
        raise AssertionError(
            f"command failed with exit code {completed.returncode}\n"
            f"command: {' '.join(cmd)}\n"
            f"stderr:\n{completed.stderr.decode('utf-8', errors='replace')}"
        )
    return ToolResult(
        returncode=completed.returncode,
        stdout="",  # written to file, not memory
        stderr=completed.stderr.decode("utf-8", errors="replace"),
    )


def run_view_plink(
    vcz_path: pathlib.Path,
    out_stem: pathlib.Path,
    *,
    extra_args: str = "",
) -> pathlib.Path:
    """Invoke ``vcztools view-plink``; return the output stem."""
    cmd = (
        f"view-plink {pathlib.Path(vcz_path).as_posix()} "
        f"--output {out_stem.as_posix()} {extra_args}"
    )
    runner = ct.CliRunner()
    result = runner.invoke(cli.vcztools_main, cmd, catch_exceptions=False)
    if result.exit_code != 0:
        raise AssertionError(
            f"view-plink exited {result.exit_code}\n"
            f"command: vcztools {cmd}\nstderr: {result.stderr}"
        )
    return out_stem


def run_view_bgen(
    vcz_path: pathlib.Path,
    out_stem: pathlib.Path,
    *,
    compression_level: int = -1,
    variant_id_field: str | None = None,
    fixed_variant_size: bool = False,
    total_string_length: int | None = None,
    pad_byte: str | None = None,
    extra_args: str = "",
) -> pathlib.Path:
    """Invoke ``vcztools view-bgen``; return the output stem.

    ``variant_id_field`` (``"rsid"`` or ``"varid"``) chooses which BGEN
    string slot carries the zarr ``variant_id``; the other becomes the
    padding field. ``None`` (the default) leaves the CLI default in
    place (``"rsid"``).

    ``fixed_variant_size=True`` switches output to the random-access
    fixed-stride encoding (``BgenEncoder`` path); requires
    ``compression_level=0``. ``total_string_length`` and ``pad_byte``
    override the encoder defaults in that mode.
    """
    parts = [
        "view-bgen",
        pathlib.Path(vcz_path).as_posix(),
        "--output",
        out_stem.as_posix(),
        "--compression-level",
        str(compression_level),
    ]
    if variant_id_field is not None:
        parts += ["--variant-id-field", variant_id_field]
    if fixed_variant_size:
        parts.append("--fixed-variant-size")
    if total_string_length is not None:
        parts += ["--total-string-length", str(total_string_length)]
    if pad_byte is not None:
        parts += ["--pad-byte", pad_byte]
    if extra_args:
        parts.append(extra_args)
    cmd = " ".join(parts)
    runner = ct.CliRunner()
    result = runner.invoke(cli.vcztools_main, cmd, catch_exceptions=False)
    if result.exit_code != 0:
        raise AssertionError(
            f"view-bgen exited {result.exit_code}\n"
            f"command: vcztools {cmd}\nstderr: {result.stderr}"
        )
    return out_stem
