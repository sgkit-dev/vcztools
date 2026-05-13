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
import zarr

import vcztools.bcftools_filter as bcftools_filter
import vcztools.bgen as bgen
import vcztools.cli as cli
import vcztools.retrieval as retrieval


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
    out_prefix: pathlib.Path,
    *,
    extra_args: str = "",
) -> pathlib.Path:
    """Invoke ``vcztools view-plink``; return the output prefix."""
    cmd = (
        f"view-plink {pathlib.Path(vcz_path).as_posix()} "
        f"--out {out_prefix.as_posix()} {extra_args}"
    )
    runner = ct.CliRunner()
    result = runner.invoke(cli.vcztools_main, cmd, catch_exceptions=False)
    if result.exit_code != 0:
        raise AssertionError(
            f"view-plink exited {result.exit_code}\n"
            f"command: vcztools {cmd}\nstderr: {result.stderr}"
        )
    return out_prefix


def run_view_bgen(
    vcz_path: pathlib.Path,
    out_prefix: pathlib.Path,
    *,
    compression_level: int = -1,
    extra_args: str = "",
) -> pathlib.Path:
    """Invoke ``vcztools view-bgen``; return the output prefix."""
    cmd = (
        f"view-bgen {pathlib.Path(vcz_path).as_posix()} "
        f"--out {out_prefix.as_posix()} "
        f"--compression-level {compression_level} {extra_args}"
    )
    runner = ct.CliRunner()
    result = runner.invoke(cli.vcztools_main, cmd, catch_exceptions=False)
    if result.exit_code != 0:
        raise AssertionError(
            f"view-bgen exited {result.exit_code}\n"
            f"command: vcztools {cmd}\nstderr: {result.stderr}"
        )
    return out_prefix


def run_bgen_encoder(vcz_path: pathlib.Path, out_prefix: pathlib.Path) -> pathlib.Path:
    """Encode ``vcz_path`` to a fixed-size BGEN via :class:`vcztools.bgen.BgenEncoder`.

    Writes ``<out_prefix>.bgen``, ``<out_prefix>.sample``, and the
    bgenix ``.bgen.bgi`` sidecar matching the encoder's fixed-size
    layout. The encoder rejects multi-allelic variants, so a
    biallelic-only filter is materialised first (matching what
    ``view-bgen --max-alleles 2`` does for the variable-size path).
    """
    root = zarr.open_group(str(vcz_path), mode="r")
    bgen_path = out_prefix.with_suffix(".bgen")
    sample_path = out_prefix.with_suffix(".sample")
    bgi_path = pathlib.Path(str(bgen_path) + ".bgi")
    with retrieval.VczReader(root) as reader:
        biallelic_filter = bcftools_filter.BcftoolsFilter(
            field_names=reader.field_names, include="N_ALT <= 1"
        )
        reader.set_variant_filter(biallelic_filter)
        reader.materialise_variant_filter()
        with bgen.BgenEncoder(reader) as enc:
            with open(bgen_path, "wb") as f:
                off = 0
                step = 1 << 17
                while True:
                    chunk = enc.read(off, step)
                    if not chunk:
                        break
                    f.write(chunk)
                    off += len(chunk)
            bgen.write_bgen_index(reader, bgi_path, enc.variant_offsets)
        sample_path.write_text(bgen.generate_sample(reader))
    return bgen_path
