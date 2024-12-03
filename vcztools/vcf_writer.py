import concurrent.futures
import functools
import io
import re
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import zarr

from vcztools.regions import (
    parse_regions,
    parse_targets,
    regions_to_chunk_indexes,
    regions_to_selection,
)
from vcztools.utils import (
    open_file_like,
    search,
)

from . import _vcztools, constants
from .constants import RESERVED_VARIABLE_NAMES
from .filter import FilterExpressionEvaluator, FilterExpressionParser

# references to the VCF spec are for https://samtools.github.io/hts-specs/VCFv4.3.pdf

# [Table 1: Reserved INFO keys]
RESERVED_INFO_KEY_DESCRIPTIONS = {
    "AA": "Ancestral allele",
    "AC": "Allele count in genotypes, for each ALT allele, in the same order as listed",
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
    drop_genotypes: bool = False,
    include: Optional[str] = None,
    exclude: Optional[str] = None,
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
        elif samples is None:
            sample_ids = root["sample_id"][:]
            samples_selection = None
        else:
            all_samples = root["sample_id"][:]
            exclude_samples = samples.startswith("^")
            samples = samples.lstrip("^")
            sample_ids = np.array(samples.split(","))
            if np.all(sample_ids == np.array("")):
                sample_ids = np.empty((0,))

            samples_selection = search(all_samples, sample_ids)
            if exclude_samples:
                samples_selection = np.setdiff1d(
                    np.arange(all_samples.size), samples_selection
                )
                sample_ids = all_samples[samples_selection]

        if not no_header:
            original_header = root.attrs.get("vcf_header", None)
            vcf_header = _generate_header(
                root, original_header, sample_ids, no_version=no_version
            )
            print(vcf_header, end="", file=output)

        if header_only:
            return

        pos = root["variant_position"]
        num_variants = pos.shape[0]

        if num_variants == 0:
            return

        contigs = root["contig_id"][:].astype("S")
        filters = root["filter_id"][:].astype("S")

        if include and exclude:
            raise ValueError(
                "Cannot handle both an include expression and an exclude expression."
            )
        elif include or exclude:
            parser = FilterExpressionParser()
            parse_results = parser(include or exclude)[0]
            filter_evaluator = FilterExpressionEvaluator(
                parse_results, invert=bool(exclude)
            )
            filter_evaluator = functools.partial(filter_evaluator, root)
        else:
            filter_evaluator = None

        if variant_regions is None and variant_targets is None:
            # no regions or targets selected
            with concurrent.futures.ThreadPoolExecutor() as executor:
                preceding_future = None
                for v_chunk in range(pos.cdata_shape[0]):
                    v_mask_chunk = (
                        filter_evaluator(v_chunk) if filter_evaluator else None
                    )
                    future = executor.submit(
                        c_chunk_to_vcf,
                        root,
                        v_chunk,
                        v_mask_chunk,
                        samples_selection,
                        contigs,
                        filters,
                        output,
                        drop_genotypes=drop_genotypes,
                        no_update=no_update,
                        preceding_future=preceding_future,
                    )
                    if preceding_future:
                        concurrent.futures.wait((preceding_future,))
                    preceding_future = future
        else:
            contigs_u = root["contig_id"][:].astype("U").tolist()
            regions = parse_regions(variant_regions, contigs_u)
            targets, complement = parse_targets(variant_targets, contigs_u)

            # Use the region index to find the chunks that overlap specfied regions or
            # targets
            region_index = root["region_index"][:]
            chunk_indexes = regions_to_chunk_indexes(
                regions,
                targets,
                complement,
                region_index,
            )

            # Then use only load required variant_contig/position chunks
            if len(chunk_indexes) == 0:
                # no chunks - no variants to write
                return
            elif len(chunk_indexes) == 1:
                # single chunk
                block_sel = chunk_indexes[0]
            else:
                # zarr.blocks doesn't support int array indexing - use that when it does
                block_sel = slice(chunk_indexes[0], chunk_indexes[-1] + 1)

            region_variant_contig = root["variant_contig"].blocks[block_sel][:]
            region_variant_position = root["variant_position"].blocks[block_sel][:]
            region_variant_length = root["variant_length"].blocks[block_sel][:]

            # Find the final variant selection
            variant_selection = regions_to_selection(
                regions,
                targets,
                complement,
                region_variant_contig,
                region_variant_position,
                region_variant_length,
            )
            variant_mask = np.zeros(region_variant_position.shape[0], dtype=bool)
            variant_mask[variant_selection] = 1
            # Use zarr arrays to get mask chunks aligned with the main data
            # for convenience.
            z_variant_mask = zarr.array(variant_mask, chunks=pos.chunks[0])
            with concurrent.futures.ThreadPoolExecutor() as executor:
                preceding_future = None
                for i, v_chunk in enumerate(chunk_indexes):
                    v_mask_chunk = z_variant_mask.blocks[i]

                    if filter_evaluator and np.any(v_mask_chunk):
                        v_mask_chunk = np.logical_and(
                            v_mask_chunk, filter_evaluator(v_chunk)
                        )
                    if np.any(v_mask_chunk):
                        future = executor.submit(
                            c_chunk_to_vcf,
                            root,
                            v_chunk,
                            v_mask_chunk,
                            samples_selection,
                            contigs,
                            filters,
                            output,
                            drop_genotypes=drop_genotypes,
                            no_update=no_update,
                            preceding_future=preceding_future,
                        )
                        if preceding_future:
                            concurrent.futures.wait((preceding_future,))
                        preceding_future = future


def get_vchunk_array(zarray, v_chunk, mask, samples_selection=None):
    v_chunksize = zarray.chunks[0]
    start = v_chunksize * v_chunk
    end = v_chunksize * (v_chunk + 1)
    if samples_selection is None:
        result = zarray[start:end]
    else:
        result = zarray.oindex[start:end, samples_selection]
    if mask is not None:
        result = result[mask]
    return result


def c_chunk_to_vcf(
    root,
    v_chunk,
    v_mask_chunk,
    samples_selection,
    contigs,
    filters,
    output,
    *,
    drop_genotypes,
    no_update,
    preceding_future=None,
):
    chrom = contigs[get_vchunk_array(root["variant_contig"], v_chunk, v_mask_chunk)]
    # TODO check we don't truncate silently by doing this
    pos = get_vchunk_array(root["variant_position"], v_chunk, v_mask_chunk).astype(
        np.int32
    )
    id = get_vchunk_array(root["variant_id"], v_chunk, v_mask_chunk).astype("S")
    alleles = get_vchunk_array(root["variant_allele"], v_chunk, v_mask_chunk)
    qual = get_vchunk_array(root["variant_quality"], v_chunk, v_mask_chunk)
    filter_ = get_vchunk_array(root["variant_filter"], v_chunk, v_mask_chunk)
    format_fields = {}
    info_fields = {}
    num_samples = len(samples_selection) if samples_selection is not None else None
    gt = None
    gt_phased = None

    if "call_genotype" in root and not drop_genotypes:
        if samples_selection is not None and num_samples != 0:
            gt = get_vchunk_array(
                root["call_genotype"], v_chunk, v_mask_chunk, samples_selection
            )
        else:
            gt = get_vchunk_array(root["call_genotype"], v_chunk, v_mask_chunk)

        if (
            "call_genotype_phased" in root
            and not drop_genotypes
            and (samples_selection is None or num_samples > 0)
        ):
            gt_phased = get_vchunk_array(
                root["call_genotype_phased"],
                v_chunk,
                v_mask_chunk,
                samples_selection,
            )
        else:
            gt_phased = np.zeros_like(gt, dtype=bool)

    for name, zarray in root.arrays():
        if (
            name.startswith("call_")
            and not name.startswith("call_genotype")
            and num_samples != 0
        ):
            vcf_name = name[len("call_") :]
            format_fields[vcf_name] = get_vchunk_array(
                zarray, v_chunk, v_mask_chunk, samples_selection
            )
            if num_samples is None:
                num_samples = zarray.shape[1]
        elif name.startswith("variant_") and name not in RESERVED_VARIABLE_NAMES:
            vcf_name = name[len("variant_") :]
            info_fields[vcf_name] = get_vchunk_array(zarray, v_chunk, v_mask_chunk)

    ref = alleles[:, 0].astype("S")
    alt = alleles[:, 1:].astype("S")

    if len(id.shape) == 1:
        id = id.reshape((-1, 1))
    if (
        not no_update
        and samples_selection is not None
        and "call_genotype" in root
        and not drop_genotypes
    ):
        # Recompute INFO/AC and INFO/AN
        info_fields |= _compute_info_fields(gt, alt)
    if num_samples == 0:
        gt = None
    if gt is not None and num_samples is None:
        num_samples = gt.shape[1]

    num_variants = len(pos)
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

    if preceding_future:
        concurrent.futures.wait((preceding_future,))

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


def _generate_header(ds, original_header, sample_ids, *, no_version: bool = False):
    output = io.StringIO()

    contigs = list(ds["contig_id"][:])
    filters = list(ds["filter_id"][:])
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

    if original_header is None:  # generate entire header
        # [1.4.1 File format]
        print("##fileformat=VCFv4.3", file=output)

        if "source" in ds.attrs:
            print(f'##source={ds.attrs["source"]}', file=output)

    else:  # use original header fields where appropriate
        unstructured_pattern = re.compile("##([^=]+)=([^<].*)")
        structured_pattern = re.compile("##([^=]+)=(<.*)")

        for line in original_header.split("\n"):
            if re.fullmatch(unstructured_pattern, line):
                print(line, file=output)
            else:
                match = re.fullmatch(structured_pattern, line)
                if match:
                    category = match.group(1)
                    id_pattern = re.compile("ID=([^,>]+)")
                    key = id_pattern.findall(line)[0]
                    if category not in ("contig", "FILTER", "INFO", "FORMAT"):
                        # output other structured fields
                        print(line, file=output)
                    # only output certain categories if in dataset
                    elif category == "contig" and key in contigs:
                        contigs.remove(key)
                        print(line, file=output)
                    elif category == "FILTER" and key in filters:
                        filters.remove(key)
                        print(line, file=output)
                    elif category == "INFO" and key in info_fields:
                        info_fields.remove(key)
                        print(line, file=output)
                    elif category == "FORMAT" and key in format_fields:
                        format_fields.remove(key)
                        print(line, file=output)

    # add all fields that are not in the original header
    # or all fields if there was no original header

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

    # [1.4.3 Filter field format]
    for filter in filters:
        print(f'##FILTER=<ID={filter},Description="">', file=output)

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
    contig_lengths = (
        ds.attrs["contig_lengths"] if "contig_lengths" in ds.attrs else None
    )
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
    # reverse of _vcf_type_to_numpy
    if a.dtype == bool:
        return "Flag"
    elif np.issubdtype(a.dtype, np.integer):
        return "Integer"
    elif np.issubdtype(a.dtype, np.float32):
        return "Float"
    elif a.dtype.str == "|S1":
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
