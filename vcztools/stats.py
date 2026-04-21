import numpy as np

from vcztools.utils import _as_fixed_length_unicode, open_file_like


def nrecords(reader, output):
    with open_file_like(output) as output:
        print(reader.num_variants, file=output)


def stats(reader, output):
    # reader.region_index raises with the user-facing message if absent.
    region_index = reader.region_index

    with open_file_like(output) as output:
        contigs = _as_fixed_length_unicode(reader.contig_ids).tolist()
        reader_contig_lengths = reader.contig_lengths
        if reader_contig_lengths is None:
            contig_lengths = ["."] * len(contigs)
        else:
            contig_lengths = reader_contig_lengths

        contig_indexes = region_index[:, 1]
        num_records = region_index[:, 5]

        num_records_per_contig = np.bincount(
            contig_indexes, weights=num_records
        ).astype(np.int64)

        for contig, contig_length, nr in zip(
            contigs, contig_lengths, num_records_per_contig
        ):
            if nr > 0:
                print(f"{contig}\t{contig_length}\t{nr}", file=output)
