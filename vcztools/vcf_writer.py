import concurrent.futures as cf
import contextlib
import io
import logging
import sys
import time
from datetime import datetime

import numpy as np

from vcztools.utils import (
    _as_fixed_length_string,
    _as_fixed_length_unicode,
    open_file_like,
    to_vcf_float32,
    to_vcf_int32,
)

from . import _vcztools
from .constants import FLOAT32_MISSING, RESERVED_VARIABLE_NAMES

logger = logging.getLogger(__name__)

# references to the VCF spec are for https://samtools.github.io/hts-specs/VCFv4.3.pdf

# [Table 1: Reserved INFO keys]
RESERVED_INFO_KEY_DESCRIPTIONS = {
    "AA": "Ancestral allele",
    "AC": "Allele count in genotypes",
    "AD": "Total read depth for each allele",
    "ADF": "Read depth for each allele on the forward strand",
    "ADR": "Read depth for each allele on the reverse strand",
    "AF": "Allele frequency for each ALT allele in the same order as listed",
    "AN": "Total number of alleles in called genotypes",
    "BQ": "RMS base quality",
    "CIGAR": "Cigar string describing how to align an alternate allele to the reference"
    "allele",
    "DB": "dbSNP membership",
    "DP": "Combined depth across samples",
    "END": "End position on CHROM",
    "H2": "HapMap2 membership",
    "H3": "HapMap3 membership",
    "MQ": "RMS mapping quality",
    "MQ0": "Number of MAPQ == 0 reads",
    "NS": "Number of samples with data",
    "SB": "Strand bias",
    "SOMATIC": "Somatic mutation",
    "VALIDATED": "Validated by follow-up experiment",
    "1000G": "1000 Genomes membership",
}

# [Table 2: Reserved genotype keys]
RESERVED_FORMAT_KEY_DESCRIPTIONS = {
    "AD": "Read depth for each allele",
    "ADF": "Read depth for each allele on the forward strand",
    "ADR": "Read depth for each allele on the reverse strand",
    "DP": "Read depth",
    "EC": "Expected alternate allele counts",
    "FT": 'Filter indicating if this genotype was "called"',
    "GL": "Genotype likelihoods",
    "GP": "Genotype posterior probabilities",
    "GQ": "Conditional genotype quality",
    "GT": "Genotype",
    "HQ": "Haplotype quality",
    "MQ": "RMS mapping quality",
    "PL": "Phred-scaled genotype likelihoods rounded to the closest integer",
    "PP": "Phred-scaled genotype posterior probabilities rounded to the closest "
    "integer",
    "PQ": "Phasing quality",
    "PS": "Phase set",
}


class VcfWriter:
    """Context-managed VCF writer that streams a ``VczReader``'s chunks
    to ``output``. Holds the per-write state (reader, output file,
    config flags, encode-thread pool) so chunk encoding doesn't need to
    thread it all through. The output file handle and
    :class:`~concurrent.futures.ThreadPoolExecutor` are owned for the
    duration of the ``with`` block.

    ``fill_tags`` is the set of VCF-tag names (e.g. ``{"AC", "AN"}``)
    that should be emitted as recomputed values, replacing any stored
    counterpart. The CLI populates this from ``--fill-tags`` and from
    the implicit ``view -s X`` default. The corresponding
    ``variant_<TAG>`` virtual fields end up in the per-chunk request
    with ``force_recompute=`` set, and are injected into the header's
    INFO section even when the source store had no header line for
    them.

    ``encode_threads`` sets the size of the per-chunk line encoding
    thread pool (default 4). Each chunk's rows are split into
    ``encode_threads`` contiguous blocks encoded in parallel; completed
    blocks are written to ``output`` in row order.
    """

    def __init__(
        self,
        reader,
        output,
        *,
        drop_genotypes: bool = False,
        encode_threads: int | None = None,
        fill_tags: frozenset | None = None,
    ):
        if encode_threads is None:
            encode_threads = 4
        if encode_threads < 1:
            raise ValueError(f"encode_threads must be >= 1 (got {encode_threads})")
        self.reader = reader
        self._output_arg = output
        self.drop_genotypes = drop_genotypes
        self.encode_threads = encode_threads
        self.fill_tags = frozenset() if fill_tags is None else frozenset(fill_tags)
        # The corresponding VCZ array names (variant_AC, ...) used both
        # to extend the per-chunk read list and to drive
        # force_recompute. A fill-tags name unknown to the reader's
        # virtual-field registry is dropped silently; the CLI validator
        # has already rejected anything unsupported.
        self._fill_field_names = frozenset(
            f"variant_{tag}"
            for tag in self.fill_tags
            if f"variant_{tag}" in reader.virtual_field_names
        )
        # Populated in __enter__
        self.output = None
        self._executor = None
        self._stack = None
        self._bytes_written = 0
        self._start = None

    def __enter__(self):
        with contextlib.ExitStack() as stack:
            self._start = time.perf_counter()
            self.output = stack.enter_context(open_file_like(self._output_arg))
            self._executor = stack.enter_context(
                cf.ThreadPoolExecutor(
                    max_workers=self.encode_threads,
                    thread_name_prefix="vcf-encode",
                )
            )
            # All resources entered cleanly: transfer their cleanup to
            # self._stack so __exit__ can unwind them later.
            self._stack = stack.pop_all()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            return self._stack.__exit__(exc_type, exc, tb)
        finally:
            elapsed = time.perf_counter() - self._start
            mib = self._bytes_written / (1024 * 1024)
            rate = mib / elapsed if elapsed > 0 else 0.0
            logger.info(
                f"VcfWriter: wrote {mib:.1f} MiB in {elapsed:.2f}s ({rate:.1f} MiB/s)"
            )

    def write_header(self, *, no_version: bool = False) -> None:
        header = _generate_header(
            self.reader,
            no_version=no_version,
            extra_info_fields=self._fill_field_names,
        )
        print(header, end="", file=self.output)
        self._bytes_written += len(header)

    def write_chunks(self) -> None:
        if len(self._fill_field_names) == 0:
            iterator = self.reader.variant_chunks()
        else:
            # Pull the stored field set into a list, layer the
            # fill-tag virtual names on top (de-duplicated), and tell
            # the reader to force-recompute precisely those names so
            # any stored counterpart is overridden.
            stored = [
                f
                for f in self.reader.field_names
                if f.startswith("variant_") or f.startswith("call_")
            ]
            fields = list(dict.fromkeys(stored + list(self._fill_field_names)))
            iterator = self.reader.variant_chunks(
                fields=fields, force_recompute=self._fill_field_names
            )
        for chunk_data in iterator:
            self.write_chunk(chunk_data)

    def write_chunk(self, chunk_data) -> None:
        samples_selection = self.reader.samples_selection
        contigs = self.reader.contigs
        filters = self.reader.filters
        drop_genotypes = self.drop_genotypes

        format_fields = {}
        info_fields = {}
        num_samples = len(samples_selection)

        pos = to_vcf_int32(chunk_data["variant_position"], "POS")
        num_variants = len(pos)
        if num_variants == 0:
            return
        # Required fields
        chrom = contigs[chunk_data["variant_contig"]]
        alleles = chunk_data["variant_allele"]

        # Optional fields which we fill in with "all missing" defaults
        if "variant_id" in chunk_data:
            id = _as_fixed_length_string(chunk_data["variant_id"])
        else:
            id = np.array(["."] * num_variants, dtype="S")
        if "variant_quality" in chunk_data:
            qual = to_vcf_float32(chunk_data["variant_quality"])
        else:
            qual = np.full(num_variants, FLOAT32_MISSING, dtype=np.float32)

        # Filter defaults to "PASS" if not present
        if "variant_filter" in chunk_data:
            filter_ = chunk_data["variant_filter"]
        else:
            filter_ = np.ones((num_variants, 1), dtype=bool)

        gt = None
        gt_phased = None

        if "call_genotype" in chunk_data and not drop_genotypes:
            gt = chunk_data["call_genotype"]

            if (
                "call_genotype_phased" in chunk_data
                and not drop_genotypes
                and num_samples != 0
            ):
                gt_phased = chunk_data["call_genotype_phased"]
            else:
                # Default to unphased if call_genotype_phased not present
                gt_phased = np.zeros(gt.shape[:2], dtype=bool)

        # Iterate sorted keys so info_fields / format_fields insertion order
        # is deterministic regardless of how the reader assembled chunk_data.
        # The resulting INFO/FORMAT field ordering in each output line is
        # the insertion order of these dicts.
        for name in sorted(chunk_data):
            array = chunk_data[name]
            if (
                name.startswith("call_")
                and not name.startswith("call_genotype")
                and num_samples != 0
            ):
                vcf_name = name[len("call_") :]
                format_fields[vcf_name] = array
            elif name.startswith("variant_") and name not in RESERVED_VARIABLE_NAMES:
                vcf_name = name[len("variant_") :]
                info_fields[vcf_name] = array

        ref = _as_fixed_length_string(alleles[:, 0])
        alt = _as_fixed_length_string(alleles[:, 1:])

        if len(id.shape) == 1:
            id = id.reshape((-1, 1))
        if num_samples == 0:
            gt = None

        encoder = _vcztools.VcfEncoder(
            num_variants,
            num_samples,
            chrom=chrom,
            pos=pos,
            id=id,
            alt=alt,
            ref=ref,
            qual=qual,
            filter_ids=filters,
            filter=filter_,
        )
        # The C encoder validates NPY_ARRAY_IN_ARRAY on every field and reads
        # raw buffers via PyArray_DATA, so every field handed to add_*_field
        # must be C-contiguous. VczReader emits strided views for sample-axis
        # slices and column gathers — wrap here at the boundary instead of
        # per-chunk in CachedVariantChunk, so non-VCF callers (benchmarks,
        # query.py, Python API) avoid the memcpy.
        if gt is not None:
            gt = np.ascontiguousarray(gt)
            gt_phased = np.ascontiguousarray(gt_phased)
            encoder.add_gt_field(gt, gt_phased)
        for name, zarray in info_fields.items():
            if zarray.dtype.kind in ("O", "U", "T"):
                zarray = _as_fixed_length_string(zarray)
            elif zarray.dtype.kind == "f":
                zarray = to_vcf_float32(zarray)
            elif zarray.dtype.kind == "i" and zarray.dtype.itemsize > 4:
                zarray = to_vcf_int32(zarray, f"INFO/{name}")
            if len(zarray.shape) == 1:
                zarray = zarray.reshape((num_variants, 1))
            zarray = np.ascontiguousarray(zarray)
            encoder.add_info_field(name, zarray)

        if num_samples != 0:
            for name, zarray in format_fields.items():
                if zarray.dtype.kind in ("O", "U", "T"):
                    zarray = _as_fixed_length_string(zarray)
                elif zarray.dtype.kind == "f":
                    zarray = to_vcf_float32(zarray)
                elif zarray.dtype.kind == "i" and zarray.dtype.itemsize > 4:
                    zarray = to_vcf_int32(zarray, f"FORMAT/{name}")
                if len(zarray.shape) == 2:
                    zarray = zarray.reshape((num_variants, num_samples, 1))
                zarray = np.ascontiguousarray(zarray)
                encoder.add_format_field(name, zarray)

        def encode_block(start, end):
            # TODO: (1) make a guess at this based on number of fields and samples,
            # and (2) log a DEBUG message when we have to double.
            buflen = 1024
            parts = []
            for j in range(start, end):
                while True:
                    try:
                        parts.append(encoder.encode(j, buflen))
                        break
                    except _vcztools.VczBufferTooSmall:
                        buflen *= 2
                parts.append("\n")
            return "".join(parts)

        num_blocks = min(self.encode_threads, num_variants)
        block_size = (num_variants + num_blocks - 1) // num_blocks
        futures = []
        for i in range(num_blocks):
            start = i * block_size
            end = min(start + block_size, num_variants)
            if start >= end:
                break
            futures.append(self._executor.submit(encode_block, start, end))

        for fut in futures:
            block_text = fut.result()
            self.output.write(block_text)
            self._bytes_written += len(block_text)


def write_vcf(
    reader,
    output,
    *,
    header_only: bool = False,
    no_header: bool = False,
    no_version: bool = False,
    drop_genotypes: bool = False,
    encode_threads: int | None = None,
    fill_tags: frozenset | None = None,
) -> None:
    """Write the VCF text for ``reader`` to ``output``.

    ``output`` is either a filesystem path (``str`` / ``pathlib.Path``)
    or a writable text file-like object (anything with a ``.write``
    method, including ``sys.stdout``); paths are opened in text mode.
    VCF output is plain, uncompressed text.

    Unlike :func:`write_plink` and :func:`write_bgen`, a configured
    variant filter on the reader does not need to be resolved first:
    records are streamed and filtered on the fly, so both a configured
    variant filter and a sample subset are honoured as-is.

    ``header_only=True`` writes only the VCF header (the ``##`` meta
    lines and the ``#CHROM`` column line) and no variant records.

    ``no_header=True`` suppresses the header entirely, emitting only the
    variant records.

    ``no_version=True`` omits the vcztools version and command line from
    the header.

    ``drop_genotypes=True`` suppresses the ``GT`` genotype values from
    the per-sample output. To omit the sample columns entirely — the
    ``FORMAT`` field and every sample column, in both the header and the
    records — clear the reader's sample selection first with
    ``reader.set_samples([])``.

    ``encode_threads`` sizes the worker pool that encodes each chunk's
    records; ``None`` (default) selects the default (4). Encoded blocks
    are written in record order so the output is deterministic.

    ``fill_tags`` is a set of VCF INFO tag names (e.g. ``{"AC", "AN"}``)
    to emit as recomputed values, overriding any stored counterpart and
    injecting the corresponding INFO header lines. Tag names unknown to
    the reader's virtual-field registry are ignored.
    """
    with VcfWriter(
        reader,
        output,
        drop_genotypes=drop_genotypes,
        encode_threads=encode_threads,
        fill_tags=fill_tags,
    ) as writer:
        if not no_header:
            writer.write_header(no_version=no_version)
        if not header_only:
            writer.write_chunks()


def _generate_header(
    reader,
    *,
    no_version: bool = False,
    extra_info_fields: frozenset | None = None,
):
    output = io.StringIO()

    sample_ids = reader.sample_ids
    contigs = list(reader.contig_ids)
    filters = list(_as_fixed_length_unicode(reader.filters))
    field_names = reader.field_names
    if extra_info_fields is None:
        extra_info_fields = frozenset()
    # Union of stored fields and any --fill-tags-driven virtual names
    # the writer asked for; the latter make sure the header announces
    # INFO lines for tags the source store didn't ship.
    header_field_names = field_names | frozenset(extra_info_fields)
    info_fields = []
    format_fields = []

    if "call_genotype" in field_names and len(sample_ids) > 0:
        # GT must be the first field if present, per the spec (section 1.6.2)
        format_fields.append("GT")

    for name in sorted(header_field_names):
        if (
            name.startswith("variant_")
            and not name.endswith("_fill")
            and not name.endswith("_mask")
            and name not in RESERVED_VARIABLE_NAMES
            and reader.get_field_info(name).dims[0] == "variants"
        ):
            info_fields.append(name[len("variant_") :])
        elif (
            len(sample_ids) > 0
            and name.startswith("call_")
            and not name.endswith("_fill")
            and not name.endswith("_mask")
        ):
            info = reader.get_field_info(name)
            if (
                len(info.dims) >= 2
                and info.dims[0] == "variants"
                and info.dims[1] == "samples"
            ):
                key = name[len("call_") :]
                if key not in ("genotype", "genotype_phased"):
                    format_fields.append(key)

    # [1.4.1 File format]
    print("##fileformat=VCFv4.3", file=output)

    if reader.source is not None:
        print(f"##source={reader.source}", file=output)

    # [1.4.2 Information field format]
    for key in info_fields:
        info = reader.get_field_info(f"variant_{key}")
        vcf_number = _array_to_vcf_number("INFO", key, info)
        vcf_type = _array_to_vcf_type(info)
        vcf_description = info.attrs.get(
            "description", RESERVED_INFO_KEY_DESCRIPTIONS.get(key, "")
        )
        print(
            f'##INFO=<ID={key},Number={vcf_number},Type={vcf_type},Description="{vcf_description}">',
            file=output,
        )

    # [1.4.3 Filter field format]
    filter_descriptions = reader.filter_descriptions
    for i, filter in enumerate(filters):
        filter_description = (
            "" if filter_descriptions is None else filter_descriptions[i]
        )
        print(
            f'##FILTER=<ID={filter},Description="{filter_description}">',
            file=output,
        )

    # [1.4.4 Individual format field format]
    for key in format_fields:
        if key == "GT":
            print(
                '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
                file=output,
            )
        else:
            info = reader.get_field_info(f"call_{key}")
            vcf_number = _array_to_vcf_number("FORMAT", key, info)
            vcf_type = _array_to_vcf_type(info)
            vcf_description = info.attrs.get(
                "description", RESERVED_FORMAT_KEY_DESCRIPTIONS.get(key, "")
            )
            print(
                f'##FORMAT=<ID={key},Number={vcf_number},Type={vcf_type},Description="{vcf_description}">',
                file=output,
            )

    # [1.4.7 Contig field format]
    contig_lengths = reader.contig_lengths
    for i, contig in enumerate(contigs):
        if contig_lengths is None:
            print(f"##contig=<ID={contig}>", file=output)
        else:
            print(f"##contig=<ID={contig},length={contig_lengths[i]}>", file=output)

    if not no_version:
        print(
            f"##vcztools_viewCommand={' '.join(sys.argv[1:])}; Date={datetime.now()}",
            file=output,
        )

    # Other meta information lines not covered above
    if reader.vcf_meta_information is not None:
        for key, value in reader.vcf_meta_information:
            if key not in ("fileformat", "source"):
                print(f"##{key}={value}", file=output)

    # [1.5 Header line syntax]
    print(
        "#CHROM",
        "POS",
        "ID",
        "REF",
        "ALT",
        "QUAL",
        "FILTER",
        "INFO",
        sep="\t",
        end="",
        file=output,
    )

    if len(sample_ids) > 0:
        print(end="\t", file=output)
        print("FORMAT", *sample_ids, sep="\t", file=output)
    else:
        print(file=output)

    return output.getvalue()


def _array_to_vcf_number(category, key, info):
    # reverse of vcf_number_to_dimension_and_size
    if info.dtype == bool:
        return 0
    elif category == "INFO" and len(info.dims) == 1:
        return 1
    elif category == "FORMAT" and len(info.dims) == 2:
        return 1

    last_dim = info.dims[-1]
    if last_dim == "alt_alleles":
        return "A"
    elif last_dim == "alleles":
        return "R"
    elif last_dim == "genotypes":
        return "G"
    elif last_dim == f"{category}_{key}_dim":
        return info.shape[-1]
    else:
        raise ValueError(
            f"Cannot determine VCF Number for dimension name "
            f"'{last_dim}' in field {info.name}"
        )


def _array_to_vcf_type(info):
    if info.dtype == bool:
        return "Flag"
    elif np.issubdtype(info.dtype, np.integer):
        return "Integer"
    elif np.issubdtype(info.dtype, np.floating):
        return "Float"
    elif info.dtype.str[1:] in ("S1", "U1"):
        return "Character"
    elif info.dtype.kind in ("O", "S", "U", "T"):
        return "String"
    else:
        raise ValueError(f"Unsupported dtype: {info.dtype}")
