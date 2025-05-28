import io
import logging
import sys
from datetime import datetime

import numpy as np
import zarr

from vcztools.samples import parse_samples
from vcztools.utils import (
    open_file_like,
)

from . import _vcztools, constants, retrieval
from . import filter as filter_mod
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


def dims(arr):
    return arr.attrs["_ARRAY_DIMENSIONS"]


def write_vcf(
    vcz,
    output,
    *,
    header_only: bool = False,
    no_header: bool = False,
    no_version: bool = False,
    variant_regions=None,
    variant_targets=None,
    no_update=None,
    samples=None,
    force_samples: bool = False,
    drop_genotypes: bool = False,
    include: str | None = None,
    exclude: str | None = None,
) -> None:
    """Convert a dataset to a VCF file.

    If a VCF header is required and ``vcf_header`` attribute is present in the dataset,
    it will be used to generate the new VCF header. In this case, any variables in the
    dataset that are not specified in this header will have corresponding header lines
    added, and any lines in the header without a corresponding variable in the dataset
    will be omitted.

    In the case of no ``vcf_header`` attribute, a VCF header will
    be generated, and will include all variables in the dataset.

    Float fields are written with up to 3 decimal places of precision.
    Exponent/scientific notation is *not* supported, so values less than
    ``5e-4`` will be rounded to zero.

    Data is written sequentially to VCF, using C to optimize the write
    throughput speed.

    Data is loaded into memory in chunks sized according to the chunking along
    the variants dimension. Chunking in other dimensions (such as samples) is
    ignored for the purposes of writing VCF. If the dataset is not chunked
    (because it does not originate from Zarr or Dask, for example), then it
    will all be loaded into memory at once.

    The output is *not* compressed or indexed. It is therefore recommended to
    post-process the output using external tools such as ``bgzip(1)``,
    ``bcftools(1)``, or ``tabix(1)``.

    Parameters
    ----------
    input
        Dataset to convert to VCF.
    output
        A path or text file object that the output VCF should be written to.
    """

    root = zarr.open(vcz, mode="r")

    with open_file_like(output) as output:
        if samples and drop_genotypes:
            raise ValueError("Cannot select samples and drop genotypes.")
        elif drop_genotypes:
            sample_ids = []
            samples_selection = np.array([])
        else:
            all_samples = root["sample_id"][:]
            sample_ids, samples_selection = parse_samples(
                samples, all_samples, force_samples=force_samples
            )

        # Need to try parsing filter expressions before writing header
        filter_mod.FilterExpression(
            field_names=set(root), include=include, exclude=exclude
        )

        if not no_header:
            force_ac_an_header = not drop_genotypes and samples_selection is not None
            vcf_header = _generate_header(
                root,
                sample_ids,
                no_version=no_version,
                force_ac_an=force_ac_an_header,
            )
            print(vcf_header, end="", file=output)

        if header_only:
            return

        contigs = root["contig_id"][:].astype("S")
        filters = get_filter_ids(root)

        for chunk_data in retrieval.variant_chunk_iter(
            root,
            variant_regions=variant_regions,
            variant_targets=variant_targets,
            include=include,
            exclude=exclude,
            samples_selection=samples_selection,
        ):
            c_chunk_to_vcf(
                chunk_data,
                samples_selection,
                contigs,
                filters,
                output,
                drop_genotypes=drop_genotypes,
                no_update=no_update,
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
):
    format_fields = {}
    info_fields = {}
    num_samples = len(samples_selection) if samples_selection is not None else None

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
        id = chunk_data["variant_id"].astype("S")
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
            and (samples_selection is None or num_samples != 0)
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
            if num_samples is None:
                num_samples = array.shape[1]
        elif name.startswith("variant_") and name not in RESERVED_VARIABLE_NAMES:
            vcf_name = name[len("variant_") :]
            info_fields[vcf_name] = array

    ref = alleles[:, 0].astype("S")
    alt = alleles[:, 1:].astype("S")

    if len(id.shape) == 1:
        id = id.reshape((-1, 1))
    if (
        not no_update
        and samples_selection is not None
        and "call_genotype" in chunk_data
        and not drop_genotypes
    ):
        # Recompute INFO/AC and INFO/AN
        info_fields |= _compute_info_fields(gt, alt)
    if num_samples == 0:
        gt = None
    if gt is not None and num_samples is None:
        num_samples = gt.shape[1]

    encoder = _vcztools.VcfEncoder(
        num_variants,
        num_samples if num_samples is not None else 0,
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
        if zarray.dtype.kind in ("O", "U"):
            zarray = zarray.astype("S")
        if len(zarray.shape) == 1:
            zarray = zarray.reshape((num_variants, 1))
        encoder.add_info_field(name, zarray)

    if num_samples != 0:
        for name, zarray in format_fields.items():
            if zarray.dtype.kind in ("O", "U"):
                zarray = zarray.astype("S")
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


def get_filter_ids(root):
    """
    Returns the filter IDs from the specified Zarr store. If the array
    does not exist, return a single filter "PASS" by default.
    """
    if "filter_id" in root:
        filters = root["filter_id"][:].astype("S")
    else:
        filters = np.array(["PASS"], dtype="S")
    return filters


def _generate_header(
    ds,
    sample_ids,
    *,
    no_version: bool = False,
    force_ac_an: bool = False,
):
    output = io.StringIO()

    contigs = list(ds["contig_id"][:])
    filters = list(get_filter_ids(ds).astype("U"))
    info_fields = []
    format_fields = []

    if "call_genotype" in ds and len(sample_ids) > 0:
        # GT must be the first field if present, per the spec (section 1.6.2)
        format_fields.append("GT")

    for var in sorted(ds.keys()):
        arr = ds[var]
        if (
            var.startswith("variant_")
            and not var.endswith("_fill")
            and not var.endswith("_mask")
            and var not in RESERVED_VARIABLE_NAMES
            and dims(arr)[0] == "variants"
        ):
            key = var[len("variant_") :]
            info_fields.append(key)
        elif (
            len(sample_ids) > 0
            and var.startswith("call_")
            and not var.endswith("_fill")
            and not var.endswith("_mask")
            and dims(arr)[0] == "variants"
            and dims(arr)[1] == "samples"
        ):
            key = var[len("call_") :]
            if key in ("genotype", "genotype_phased"):
                continue
            format_fields.append(key)

    # [1.4.1 File format]
    print("##fileformat=VCFv4.3", file=output)

    if "source" in ds.attrs:
        print(f'##source={ds.attrs["source"]}', file=output)

    # [1.4.2 Information field format]
    for key in info_fields:
        arr = ds[f"variant_{key}"]
        category = "INFO"
        vcf_number = _array_to_vcf_number(category, key, arr)
        vcf_type = _array_to_vcf_type(arr)
        vcf_description = arr.attrs.get(
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
    filter_descriptions = (
        ds["filter_description"] if "filter_description" in ds else None
    )
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
            arr = ds[f"call_{key}"]
            category = "FORMAT"
            vcf_number = _array_to_vcf_number(category, key, arr)
            vcf_type = _array_to_vcf_type(arr)
            vcf_description = arr.attrs.get(
                "description", RESERVED_FORMAT_KEY_DESCRIPTIONS.get(key, "")
            )
            print(
                f'##FORMAT=<ID={key},Number={vcf_number},Type={vcf_type},Description="{vcf_description}">',
                file=output,
            )

    # [1.4.7 Contig field format]
    contig_lengths = ds["contig_length"] if "contig_length" in ds else None
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
    if "vcf_meta_information" in ds.attrs:
        for key, value in ds.attrs["vcf_meta_information"]:
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


def _array_to_vcf_number(category, key, a):
    # reverse of vcf_number_to_dimension_and_size
    if a.dtype == bool:
        return 0
    elif category == "INFO" and len(dims(a)) == 1:
        return 1
    elif category == "FORMAT" and len(dims(a)) == 2:
        return 1

    last_dim = dims(a)[-1]
    if last_dim == "alt_alleles":
        return "A"
    elif last_dim == "alleles":
        return "R"
    elif last_dim == "genotypes":
        return "G"
    elif last_dim == f"{category}_{key}_dim":
        return a.shape[-1]
    else:
        raise ValueError(
            f"Cannot determine VCF Number for dimension name '{last_dim}' in {a}"
        )


def _array_to_vcf_type(a):
    if a.dtype == bool:
        return "Flag"
    elif np.issubdtype(a.dtype, np.integer):
        return "Integer"
    elif np.issubdtype(a.dtype, np.float32):
        return "Float"
    elif a.dtype.str[1:] in ("S1", "U1"):
        return "Character"
    elif a.dtype.kind in ("O", "S", "U"):
        return "String"
    else:
        raise ValueError(f"Unsupported dtype: {a.dtype}")


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
