import io
import logging
import sys
from datetime import datetime

import numpy as np

from vcztools.utils import (
    _as_fixed_length_string,
    _as_fixed_length_unicode,
    open_file_like,
)

from . import _vcztools, constants
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


def write_vcf(
    reader,
    output,
    *,
    header_only: bool = False,
    no_header: bool = False,
    no_version: bool = False,
    no_update=None,
    drop_genotypes: bool = False,
) -> None:
    with open_file_like(output) as output:
        if not no_header:
            force_ac_an_header = not drop_genotypes and reader.subsetting_samples
            vcf_header = _generate_header(
                reader,
                no_version=no_version,
                force_ac_an=force_ac_an_header,
            )
            print(vcf_header, end="", file=output)

        if header_only:
            return

        for chunk_data in reader.variant_chunks():
            c_chunk_to_vcf(
                chunk_data,
                reader.samples_selection,
                reader.contigs,
                reader.filters,
                output,
                drop_genotypes=drop_genotypes,
                no_update=no_update,
                subsetting_samples=reader.subsetting_samples,
            )


def c_chunk_to_vcf(
    chunk_data,
    samples_selection,
    contigs,
    filters,
    output,
    *,
    drop_genotypes,
    no_update,
    subsetting_samples,
):
    format_fields = {}
    info_fields = {}
    num_samples = len(samples_selection)

    # TODO check we don't truncate silently by doing this
    pos = chunk_data["variant_position"].astype(np.int32)
    num_variants = len(pos)
    if num_variants == 0:
        return ""
    # Required fields
    chrom = contigs[chunk_data["variant_contig"]]
    alleles = chunk_data["variant_allele"]

    # Optional fields which we fill in with "all missing" defaults
    if "variant_id" in chunk_data:
        id = _as_fixed_length_string(chunk_data["variant_id"])
    else:
        id = np.array(["."] * num_variants, dtype="S")
    if "variant_quality" in chunk_data:
        qual = chunk_data["variant_quality"]
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

    for name, array in chunk_data.items():
        if (
            name.startswith("call_")
            and not name == "call_mask"
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
    if (
        not no_update
        and subsetting_samples
        and "call_genotype" in chunk_data
        and not drop_genotypes
    ):
        # Recompute INFO/AC and INFO/AN. When the effective subset is empty
        # (num_samples == 0), ``gt`` still contains all samples (see the
        # bypass in ``VariantChunkReader.get_chunk_data``), so AC/AN are
        # recomputed over the full genotype set to match bcftools.
        info_fields |= _compute_info_fields(gt, alt)
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
    # print(encoder.arrays)
    if gt is not None:
        encoder.add_gt_field(gt, gt_phased)
    for name, zarray in info_fields.items():
        # print(array.dtype.kind)
        if zarray.dtype.kind in ("O", "U", "T"):
            zarray = _as_fixed_length_string(zarray)
        if len(zarray.shape) == 1:
            zarray = zarray.reshape((num_variants, 1))
        encoder.add_info_field(name, zarray)

    if num_samples != 0:
        for name, zarray in format_fields.items():
            if zarray.dtype.kind in ("O", "U", "T"):
                zarray = _as_fixed_length_string(zarray)
            if len(zarray.shape) == 2:
                zarray = zarray.reshape((num_variants, num_samples, 1))
            encoder.add_format_field(name, zarray)

    # TODO: (1) make a guess at this based on number of fields and samples,
    # and (2) log a DEBUG message when we have to double.
    buflen = 1024
    for j in range(num_variants):
        failed = True
        while failed:
            try:
                line = encoder.encode(j, buflen)
                failed = False
            except _vcztools.VczBufferTooSmall:
                buflen *= 2
                # print("Bumping buflen to", buflen)
        print(line, file=output)


def _generate_header(
    reader,
    *,
    no_version: bool = False,
    force_ac_an: bool = False,
):
    output = io.StringIO()

    sample_ids = reader.sample_ids
    contigs = list(reader.contig_ids)
    filters = list(_as_fixed_length_unicode(reader.filters))
    field_names = reader.field_names
    info_fields = []
    format_fields = []

    if "call_genotype" in field_names and len(sample_ids) > 0:
        # GT must be the first field if present, per the spec (section 1.6.2)
        format_fields.append("GT")

    for name in sorted(field_names):
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

    if force_ac_an:
        # bcftools always recomputes the AC and AN fields when samples are specified,
        # even if these fields don't exist before
        for key, number in [("AC", "A"), ("AN", "1")]:
            if key not in info_fields:
                print(
                    f"##INFO=<ID={key},Number={number},Type=Integer,"
                    f'Description="{RESERVED_INFO_KEY_DESCRIPTIONS[key]}">',
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
    elif np.issubdtype(info.dtype, np.float32):
        return "Float"
    elif info.dtype.str[1:] in ("S1", "U1"):
        return "Character"
    elif info.dtype.kind in ("O", "S", "U", "T"):
        return "String"
    else:
        raise ValueError(f"Unsupported dtype: {info.dtype}")


def _compute_info_fields(gt: np.ndarray, alt: np.ndarray):
    flatter_gt = gt.reshape((gt.shape[0], -1))
    allele_count = alt.shape[1] + 1

    def filter_and_bincount(values: np.ndarray):
        positive = values[values > 0]
        return np.bincount(positive, minlength=allele_count)[1:]

    computed_ac = np.apply_along_axis(filter_and_bincount, 1, flatter_gt).astype(
        np.int32
    )
    computed_ac[alt == b""] = constants.INT_FILL
    computed_an = np.sum(flatter_gt >= 0, axis=1, dtype=np.int32)

    return {
        "AC": computed_ac,
        "AN": computed_an,
    }
