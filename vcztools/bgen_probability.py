"""VCZ validation and chunked orchestration for native layout-2 encoding."""

import concurrent.futures as cf
import itertools
import logging
import math
import struct
import zlib

import numpy as np

from vcztools import utils

logger = logging.getLogger(__name__)


def _string(value, width):
    encoded = str(value).encode("utf-8")
    if len(encoded) >= 1 << (8 * width):
        raise ValueError(f"BGEN string exceeds {width}-byte length prefix")
    return len(encoded).to_bytes(width, "little") + encoded


def _ploidy(chunk, row, samples):
    if "call_PLOIDY" in chunk:
        p = np.asarray(chunk["call_PLOIDY"][row])
        if p.shape != (samples,) or p.dtype.kind not in "iu":
            raise ValueError("call_PLOIDY must be an integer (variants, samples) array")
        if ((p < 0) | (p > 63)).any():
            raise ValueError("BGEN ploidy must be in 0..63")
        return p.astype(np.uint8)
    if "call_genotype" not in chunk:
        raise ValueError("Probability export requires call_PLOIDY or call_genotype")
    g = chunk["call_genotype"][row]
    if g.ndim != 2 or g.dtype.kind not in "iu":
        raise ValueError("call_genotype must be a 3D integer array")
    padding = g == -2
    p = (~padding).sum(axis=-1)
    if (p > 63).any() or (padding & (np.arange(g.shape[-1]) < p[:, None])).any():
        raise ValueError("Genotype fill must be trailing and ploidy must be in 0..63")
    return p.astype(np.uint8)


def _phase(chunk, row, probability_source, unphased, *, warn_mixed=True):
    if "variant_bgen_phased" in chunk:
        flag = chunk["variant_bgen_phased"][row]
        if flag not in (0, 1):
            raise ValueError("variant_bgen_phased must contain 0/1 flags")
        phased = bool(flag)
    elif probability_source:
        if "call_GP" in chunk and "call_HP" in chunk:
            raise ValueError("GP plus HP export requires variant_bgen_phased")
        phased = "call_HP" in chunk
    elif "call_genotype_phased" in chunk:
        flags = chunk["call_genotype_phased"][row]
        if flags.any() and not flags.all() and not unphased and warn_mixed:
            logger.warning(
                "Mixed phase across samples; hard calls emitted as unphased "
                "(BGEN has one phase flag per variant)."
            )
        phased = bool(flags.all())
    else:
        phased = False
    if unphased and phased and probability_source:
        raise ValueError("Cannot convert HP to GP with unphased=True without data loss")
    return phased and not unphased


def _encode_genotype_blocks(chunk, allele_counts, samples, bits, unphased, backend):
    """Validate VCZ representation, then cross the native numerical seam once."""
    probability_source = "call_GP" in chunk or "call_HP" in chunk
    variants = len(allele_counts)
    ploidy = np.empty((variants, samples), dtype=np.uint8)
    missing = np.zeros((variants, samples), dtype=bool)
    phased = np.empty(variants, dtype=bool)
    for row, k in enumerate(allele_counts):
        k = int(k)
        ploidy[row] = _ploidy(chunk, row, samples)
        phased[row] = _phase(chunk, row, probability_source, unphased)
        explicit_missing = chunk.get("call_BGEN_MISSING")
        if explicit_missing is not None:
            explicit_missing = explicit_missing[row]
            if (
                explicit_missing.shape != (samples,)
                or not np.isin(explicit_missing, [0, 1]).all()
            ):
                raise ValueError(
                    "call_BGEN_MISSING must contain one 0/1 flag per sample"
                )
            missing[row] = explicit_missing
        if probability_source:
            field = "call_HP" if phased[row] else "call_GP"
            if field not in chunk:
                raise ValueError(
                    f"BGEN phase requires {field}; refusing hard-call fallback"
                )
            inactive = "call_GP" if phased[row] else "call_HP"
            if inactive in chunk:
                vector = chunk[inactive][row]
                if (
                    vector.ndim != 2
                    or vector.shape[0] != samples
                    or vector.shape[1] < 1
                    or not utils.is_missing(vector[:, 0]).all()
                    or not utils.is_fill(vector[:, 1:]).all()
                ):
                    raise ValueError(
                        f"Inactive {inactive} must contain missing then fill sentinels"
                    )
            vector = chunk[field][row]
            if (
                vector.ndim != 2
                or vector.shape[0] != samples
                or vector.dtype.kind != "f"
                or vector.dtype.itemsize not in (4, 8)
            ):
                raise ValueError(f"{field} must be a float32/float64 probability array")
            counts = np.empty(samples, dtype=np.intp)
            for p in np.unique(ploidy[row]):
                p = int(p)
                count = p * k if phased[row] else math.comb(p + k - 1, p)
                if count > vector.shape[1]:
                    raise ValueError(f"{field} has insufficient probability states")
                counts[ploidy[row] == p] = count
            active = np.arange(vector.shape[1]) < counts[:, None]
            if not (utils.is_fill(vector) | active).all():
                raise ValueError(f"{field} has non-fill values beyond active states")
            if explicit_missing is None:
                missing[row] = np.all(np.isnan(vector), axis=1, where=active) & (
                    counts > 0
                )
            if not utils.is_missing(vector[active & missing[row, :, None]]).all():
                raise ValueError("Missing probability calls must contain missing NaNs")
        else:
            g = chunk["call_genotype"][row]
            if g.ndim != 2 or g.shape[0] != samples or g.dtype.kind not in "iu":
                raise ValueError("call_genotype must be a 3D integer array")
            active = np.arange(g.shape[-1]) < ploidy[row, :, None]
            if (
                (ploidy[row] > g.shape[1]).any()
                or ((g != -2) & ~active).any()
                or (((g < -1) | (g >= k)) & active).any()
            ):
                raise ValueError("Invalid genotype allele or non-trailing ploidy fill")
            if explicit_missing is None:
                missing[row] = ((g == -1) & active).any(axis=1)
            if ((g < 0) & active & ~missing[row, :, None]).any():
                raise ValueError("Nonmissing genotype contains missing alleles")
            if not phased[row]:
                for p in np.unique(ploidy[row]):
                    if math.comb(int(p) + k - 1, int(p)) - 1 > 1_000_000:
                        raise ValueError(
                            "Hard-call BGEN probability vector exceeds 1000000 states"
                        )
    kwargs = {
        "allele_counts": np.asarray(allele_counts, dtype=np.uint16),
        "missing": missing,
        "phased": phased,
        "bits": bits,
    }
    if probability_source:
        kwargs["ploidy"] = ploidy
        for field, argument in (("call_GP", "gp"), ("call_HP", "hp")):
            if field in chunk:
                vector = chunk[field]
                kwargs[argument] = np.ascontiguousarray(
                    vector, dtype=vector.dtype.newbyteorder("=")
                )
        return backend.encode_layout2(**kwargs)
    return backend.encode_layout2_hardcalls(
        genotypes=np.ascontiguousarray(chunk["call_genotype"], dtype=np.int32),
        **kwargs,
    )


def _genotype_block(chunk, row, k, samples, bits, unphased):
    from vcztools import bgen  # noqa: PLC0415

    backend = bgen._native_backend(encoding=True)
    selected = {name: value[row : row + 1] for name, value in chunk.items()}
    result, _ = _encode_genotype_blocks(selected, [k], samples, bits, unphased, backend)
    return result


def _variant_header(chunk, row, contigs, variant_id_field, minimum_alleles):
    alleles = utils.trim_fill(chunk["variant_allele"][row])
    if len(alleles) == 1 and minimum_alleles == 2:
        alleles = np.append(alleles, ".")
    k = len(alleles)
    if not 1 <= k <= 65535 or (alleles == "").any():
        raise ValueError("BGEN requires 1..65535 nonempty, trailing-fill alleles")
    position = int(chunk["variant_position"][row])
    if not 1 <= position <= 0xFFFFFFFF:
        raise ValueError("BGEN positions must be in 1..4294967295")
    identifier = str(chunk["variant_id"][row]) or "." if "variant_id" in chunk else "."
    varid = str(chunk["variant_bgen_id"][row]) if "variant_bgen_id" in chunk else "."
    rsid = identifier
    if variant_id_field == "varid":
        if "variant_bgen_id" in chunk:
            raise ValueError("variant_id_field='varid' conflicts with variant_bgen_id")
        varid, rsid = identifier, "."
    return (
        _string(varid, 2)
        + _string(rsid, 2)
        + _string(contigs[chunk["variant_contig"][row]], 2)
        + struct.pack("<IH", position, k)
        + b"".join(_string(a, 4) for a in alleles),
        k,
    )


def _slice_ranges(chunk, allele_counts, samples, bits, unphased, threads):
    """Bound each native output batch, including ragged hard-call state spaces."""
    probability_source = "call_GP" in chunk or "call_HP" in chunk
    variants = len(allele_counts)
    rows_per_slice = max(1, (variants + threads - 1) // threads)
    start = 0
    size = 0
    for row, k in enumerate(allele_counts):
        ploidy = _ploidy(chunk, row, samples)
        phased = _phase(chunk, row, probability_source, unphased, warn_mixed=False)
        p, counts = np.unique(ploidy, return_counts=True)
        stored = 0
        for value, count in zip(p, counts):
            value = int(value)
            states = value * (k - 1) if phased else math.comb(value + k - 1, value) - 1
            if not probability_source and not phased and states > 1_000_000:
                raise ValueError(
                    "Hard-call BGEN probability vector exceeds 1000000 states"
                )
            stored += states * int(count)
        row_size = 10 + samples + (stored * bits + 7) // 8
        if row_size > 256 * 1024**2:
            raise ValueError(
                "A BGEN genotype block exceeds the 256 MiB native batch limit"
            )
        if row > start and (
            size + row_size > 8 * 1024**2 or row - start >= rows_per_slice
        ):
            yield start, row
            start = row
            size = 0
        size += row_size
    if start < variants:
        yield start, variants


def _encode_slice(
    chunk,
    headers,
    allele_counts,
    start,
    end,
    samples,
    bits,
    unphased,
    backend,
    compression_level,
):
    selected = {name: value[start:end] for name, value in chunk.items()}
    encoded, boundaries = _encode_genotype_blocks(
        selected, allele_counts[start:end], samples, bits, unphased, backend
    )
    raw = memoryview(encoded)
    output = bytearray()
    lengths = []
    for row in range(end - start):
        genotype = raw[int(boundaries[row]) : int(boundaries[row + 1])]
        compressed = zlib.compress(genotype, compression_level)
        block = (
            headers[start + row]
            + struct.pack("<II", len(compressed) + 4, len(genotype))
            + compressed
        )
        output.extend(block)
        lengths.append(len(block))
    return output, lengths


def write_payload(
    reader,
    output,
    *,
    header,
    compression_level,
    bits,
    unphased,
    variant_id_field,
    minimum_alleles=1,
    backend,
    encode_threads,
):
    """Write one reader chunk at a time and return absolute variant boundaries."""
    fields = ["variant_allele", "variant_contig", "variant_position"]
    fields += [
        name
        for name in (
            "variant_id",
            "variant_bgen_id",
            "variant_bgen_phased",
            "call_GP",
            "call_HP",
            "call_PLOIDY",
            "call_BGEN_MISSING",
            "call_genotype",
            "call_genotype_phased",
        )
        if name in reader.field_names
    ]
    if (
        {"call_GP", "call_HP"} & reader.field_names
        and "call_PLOIDY" in reader.field_names
        and "call_genotype" in fields
    ):
        fields.remove("call_genotype")
    if {"call_GP", "call_HP"} & reader.field_names and "call_genotype_phased" in fields:
        fields.remove("call_genotype_phased")
    contigs = reader.contig_ids
    offsets = [len(header)]
    with (
        utils.open_file_like(output, mode="wb") as stream,
        cf.ThreadPoolExecutor(max_workers=encode_threads) as executor,
    ):
        stream.write(header)
        for chunk in reader.variant_chunks(fields=fields):
            metadata = [
                _variant_header(chunk, row, contigs, variant_id_field, minimum_alleles)
                for row in range(len(chunk["variant_allele"]))
            ]
            headers = [value[0] for value in metadata]
            allele_counts = [value[1] for value in metadata]
            samples = len(reader.sample_ids)
            ranges = iter(
                _slice_ranges(
                    chunk, allele_counts, samples, bits, unphased, encode_threads
                )
            )
            while group := list(itertools.islice(ranges, encode_threads)):
                futures = [
                    executor.submit(
                        _encode_slice,
                        chunk,
                        headers,
                        allele_counts,
                        start,
                        end,
                        samples,
                        bits,
                        unphased,
                        backend,
                        compression_level,
                    )
                    for start, end in group
                ]
                for future in futures:
                    block, lengths = future.result()
                    stream.write(block)
                    offsets.extend(offsets[-1] + np.cumsum(lengths, dtype=np.int64))
    return np.asarray(offsets, dtype=np.int64)
