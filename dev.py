import sys

import _vcztools
import numpy as np
import zarr

# From VCF fixed fields
RESERVED_VARIABLE_NAMES = [
    "variant_contig",
    "variant_position",
    "variant_id",
    "variant_id_mask",
    "variant_allele",
    "variant_quality",
    "variant_filter",
]


def copy_to_memory(group):
    mem_group = zarr.group()
    for name, array in group.items():
        if array.dtype == "O":
            copy = mem_group.empty_like(name, array, compressor={})
            # FIXME: this is the best I've been able to come up with here.
            # Something very weird with object arrays in v2
            for j, row in enumerate(array):
                copy[j] = row
        else:
            copy = mem_group.empty_like(name, array, compressor={})
            copy[:] = array[:]
        # print("copy = ", copy[:])
    return mem_group


def main(root):
    v_chunk = 0
    contigs = root["contig_id"][:].astype("S")
    # filters = root["filter_id"][:].astype("S")
    # print("contigs = ", contigs)
    # print("filters = ", contigs)

    chrom = contigs[root.variant_contig.blocks[v_chunk]]
    pos = root.variant_position.blocks[v_chunk]
    id = root.variant_id.blocks[v_chunk].astype("S")
    alleles = root.variant_allele.blocks[v_chunk]
    ref = alleles[:, 0].astype("S")
    alt = alleles[:, 1:].astype("S")
    # qual = root.variant_quality.blocks[v_chunk]
    # filter_ = filters[root.variant_filter.blocks[v_chunk]]

    num_variants = len(pos)
    if len(id.shape) == 1:
        id = id.reshape((num_variants, 1))

    # TODO gathering fields and doing IO will be done separately later so that
    # we avoid retrieving stuff we don't need.
    format_fields = {}
    info_fields = {}
    for name, array in root.arrays():
        if name.startswith("call_") and not name.startswith("call_genotype"):
            vcf_name = name[len("call_") :]
            format_fields[vcf_name] = array.blocks[v_chunk]
        elif name.startswith("variant_") and name not in RESERVED_VARIABLE_NAMES:
            vcf_name = name[len("variant_") :]
            info_fields[vcf_name] = array.blocks[v_chunk]

    gt = None
    gt_phased = None
    if "call_genotype" in root:
        array = root["call_genotype"]
        gt = array.blocks[v_chunk]
        if "call_genotype_phased" in root:
            array = root["call_genotype_phased"]
            gt_phased = array.blocks[v_chunk]
        else:
            gt_phased = np.zeros_like(gt, dtype=bool)

    # print(gt, gt_phased)
    # print(list(format_fields.keys()))
    # print(list(info_fields.keys()))

    # print(contigs[chrom])
    # print(bytes(contigs[chrom]))
    # print(pos)
    # print(alleles)
    # print(alleles.dtype)
    # print(chrom)
    # print(pos)
    # print(id)
    # print(ref)
    # print(alt)

    num_samples = 0
    if gt is not None:
        num_samples = gt.shape[1]

    encoder = _vcztools.VcfEncoder(
        num_variants, num_samples, chrom=chrom, pos=pos, id=id, alt=alt, ref=ref
    )
    print(gt.shape)
    print(gt_phased.shape)
    encoder.add_gt_field(gt.astype("int32"), gt_phased)
    # # print(encoder.arrays)
    # # print(encoder)
    for name, array in info_fields.items():
        if array.dtype.kind == "O":
            array = array.astype("S")
        if len(array.shape) == 1:
            array = array.reshape((num_variants, 1))
        if array.dtype.kind == "i":
            array = array.astype("int32")  # tmp
        if array.dtype.kind == "f":
            continue  # tmp
        if array.dtype.kind == "b":
            continue  # tmp
            # array = array.astype("int32") # tmp

        print(name, array.dtype, array.dtype.kind)
        encoder.add_info_field(name, array)

    for name, array in format_fields.items():
        if array.dtype.kind == "O":
            array = array.astype("S")
        if len(array.shape) == 2:
            array = array.reshape((num_variants, num_samples, 1))
        if array.dtype.kind == "i":
            array = array.astype("int32")  # tmp
        if array.dtype.kind == "f":
            continue  # tmp
            # array = array.astype("int32") # tmp

        print(name, array.dtype, array.dtype.kind)
        encoder.add_format_field(name, array)

    # d = encoder.arrays
    # pos = encoder.arrays["POS"]
    # print(pos)
    # # print(d)
    # pos[0] = 123457
    # print(pos.flags)
    # pos.resize(0, refcheck=False)
    # print(pos)

    encoder.print_state(sys.stdout)
    for k, v in encoder.arrays.items():
        print(k, "\t", v.shape)
    for j in range(num_variants):
        line = encoder.encode_row(j, 2**30)
        print(line)


if __name__ == "__main__":
    root = zarr.open(sys.argv[1], mode="r")
    # root = copy_to_memory(root)
    # print("pos = ", root["variant_position"].info)
    # print(root.tree())
    main(root)
    # for _ in range(10000):
    # import tqdm

    # for _ in tqdm.tqdm(range(10000)):
    #     main(root)
