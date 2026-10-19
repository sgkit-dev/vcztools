"""Chunked layout-2 encoding for probability and general-ploidy VCZ stores."""

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


def _quantize(probabilities, scale):
    p = np.asarray(probabilities, dtype=np.float64)
    if (
        not np.isfinite(p).all()
        or (p < 0).any()
        or (p > 1).any()
        or not np.isclose(p.sum(), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("BGEN probabilities must be finite, in [0, 1], and sum to 1")
    # Largest-remainder rounding keeps each simplex's integer sum exact.
    scaled = p / p.sum() * scale
    q = np.floor(scaled).astype(np.uint64)
    remainder = scale - int(q.sum())
    order = np.argsort(-(scaled - q), kind="stable")
    q[order[:remainder]] += 1
    return q[:-1]


def _pack(values, bits):
    values = np.asarray(values, dtype=np.uint64)
    if bits in (8, 16, 32):
        return values.astype(f"<u{bits // 8}").tobytes()
    result = bytearray()
    buffer = 0
    available = 0
    for value in values:
        buffer |= int(value) << available
        available += bits
        while available >= 8:
            result.append(buffer & 255)
            buffer >>= 8
            available -= 8
    if available:
        result.append(buffer)
    return bytes(result)


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


def _phase(chunk, row, probability_source, unphased):
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
        if flags.any() and not flags.all() and not unphased:
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


def _genotype_block(chunk, row, k, samples, bits, unphased):
    probability_source = "call_GP" in chunk or "call_HP" in chunk
    ploidy = _ploidy(chunk, row, samples)
    phased = _phase(chunk, row, probability_source, unphased)
    field = "call_HP" if phased else "call_GP"
    if probability_source and field not in chunk:
        raise ValueError(f"BGEN phase requires {field}; refusing hard-call fallback")
    inactive = "call_GP" if phased else "call_HP"
    if probability_source and inactive in chunk:
        vector = chunk[inactive][row]
        if (
            vector.ndim != 2
            or vector.shape[1] < 1
            or not utils.is_missing(vector[:, 0]).all()
            or not utils.is_fill(vector[:, 1:]).all()
        ):
            raise ValueError(
                f"Inactive {inactive} must contain missing then fill sentinels"
            )
    explicit_missing = chunk.get("call_BGEN_MISSING")
    if explicit_missing is not None:
        explicit_missing = explicit_missing[row]
        if (
            explicit_missing.shape != (samples,)
            or not np.isin(explicit_missing, [0, 1]).all()
        ):
            raise ValueError("call_BGEN_MISSING must contain one 0/1 flag per sample")
    missing = np.zeros(samples, dtype=bool)
    values = []
    scale = (1 << bits) - 1
    for sample, p in enumerate(ploidy):
        p = int(p)
        states = math.comb(p + k - 1, p)
        count = p * k if phased else states
        stored = p * (k - 1) if phased else states - 1
        if probability_source:
            vector = chunk[field][row, sample]
            if vector.ndim != 1 or vector.dtype.kind != "f" or vector.size < count:
                raise ValueError(f"{field} has insufficient probability states")
            if not utils.is_fill(vector[count:]).all():
                raise ValueError(f"{field} has non-fill values beyond active states")
            active = vector[:count]
            missing[sample] = (
                bool(explicit_missing[sample])
                if explicit_missing is not None
                else bool(np.isnan(active).all()) and count > 0
            )
            if missing[sample]:
                if not utils.is_missing(active).all():
                    raise ValueError(
                        "Missing probability calls must contain missing NaNs"
                    )
                values.extend([0] * stored)
            elif phased:
                for simplex in active.reshape(p, k):
                    values.extend(_quantize(simplex, scale))
            else:
                values.extend(_quantize(active, scale))
        else:
            g = chunk["call_genotype"][row, sample]
            if (
                g.ndim != 1
                or g.dtype.kind not in "iu"
                or len(g) < p
                or (g[p:] != -2).any()
                or (g[:p] < -1).any()
                or (g[:p] >= k).any()
            ):
                raise ValueError("Invalid genotype allele or non-trailing ploidy fill")
            missing[sample] = (
                bool(explicit_missing[sample])
                if explicit_missing is not None
                else bool((g[:p] == -1).any())
            )
            if not missing[sample] and (g[:p] < 0).any():
                raise ValueError("Nonmissing genotype contains missing alleles")
            if stored > 1_000_000:
                raise ValueError(
                    "Hard-call BGEN probability vector exceeds 1000000 states"
                )
            encoded = np.zeros(stored, dtype=np.uint64)
            if not missing[sample]:
                if phased:
                    for h, allele in enumerate(g[:p]):
                        if allele < k - 1:
                            encoded[h * (k - 1) + int(allele)] = scale
                else:
                    rank = sum(
                        math.comb(int(a) + i, i + 1)
                        for i, a in enumerate(sorted(g[:p]))
                    )
                    if rank < stored:
                        encoded[rank] = scale
            values.extend(encoded)
    flags = ploidy | (missing.astype(np.uint8) << 7)
    return (
        struct.pack(
            "<IHBB",
            samples,
            k,
            int(ploidy.min()) if samples else 0,
            int(ploidy.max()) if samples else 0,
        )
        + flags.tobytes()
        + bytes([phased, bits])
        + _pack(values, bits)
    )


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
    with utils.open_file_like(output, mode="wb") as stream:
        stream.write(header)
        for chunk in reader.variant_chunks(fields=fields):
            for row, padded_alleles in enumerate(chunk["variant_allele"]):
                alleles = utils.trim_fill(padded_alleles)
                if len(alleles) == 1 and minimum_alleles == 2:
                    alleles = np.append(alleles, ".")
                k = len(alleles)
                if not 1 <= k <= 65535 or (alleles == "").any():
                    raise ValueError(
                        "BGEN requires 1..65535 nonempty, trailing-fill alleles"
                    )
                position = int(chunk["variant_position"][row])
                if not 1 <= position <= 0xFFFFFFFF:
                    raise ValueError("BGEN positions must be in 1..4294967295")
                identifier = (
                    str(chunk["variant_id"][row]) or "."
                    if "variant_id" in chunk
                    else "."
                )
                varid = (
                    str(chunk["variant_bgen_id"][row])
                    if "variant_bgen_id" in chunk
                    else "."
                )
                rsid = identifier
                if variant_id_field == "varid":
                    if "variant_bgen_id" in chunk:
                        raise ValueError(
                            "variant_id_field='varid' conflicts with variant_bgen_id"
                        )
                    varid, rsid = identifier, "."
                raw = _genotype_block(
                    chunk, row, k, len(reader.sample_ids), bits, unphased
                )
                compressed = zlib.compress(raw, compression_level)
                block = (
                    _string(varid, 2)
                    + _string(rsid, 2)
                    + _string(contigs[chunk["variant_contig"][row]], 2)
                    + struct.pack("<IH", position, k)
                    + b"".join(_string(a, 4) for a in alleles)
                    + struct.pack("<II", len(compressed) + 4, len(raw))
                    + compressed
                )
                stream.write(block)
                offsets.append(offsets[-1] + len(block))
    return np.asarray(offsets, dtype=np.int64)
