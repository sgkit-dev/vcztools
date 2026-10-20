"""Slow numerical oracle retained from the original Python BGEN encoder."""

import math
import struct

import numpy as np

from vcztools import bgen_probability


def quantize(probabilities, scale):
    p = np.asarray(probabilities, dtype=np.float64)
    scaled = p / p.sum() * scale
    q = np.floor(scaled).astype(np.uint64)
    remainder = scale - int(q.sum())
    order = np.argsort(-(scaled - q), kind="stable")
    q[order[:remainder]] += 1
    return q[:-1]


def genotype_block(chunk, row, k, samples, bits, unphased=False):
    probability_source = "call_GP" in chunk or "call_HP" in chunk
    ploidy = bgen_probability._ploidy(chunk, row, samples)
    phased = bgen_probability._phase(chunk, row, probability_source, unphased)
    field = "call_HP" if phased else "call_GP"
    explicit_missing = chunk.get("call_BGEN_MISSING")
    missing = np.zeros(samples, dtype=bool)
    values = []
    scale = (1 << bits) - 1
    for sample, p in enumerate(ploidy):
        p = int(p)
        states = math.comb(p + k - 1, p)
        count = p * k if phased else states
        stored = p * (k - 1) if phased else states - 1
        if probability_source:
            active = chunk[field][row, sample, :count]
            missing[sample] = (
                bool(explicit_missing[row, sample])
                if explicit_missing is not None
                else bool(np.isnan(active).all()) and count > 0
            )
            if missing[sample]:
                values.extend([0] * stored)
            elif phased:
                for simplex in active.reshape(p, k):
                    values.extend(quantize(simplex, scale))
            else:
                values.extend(quantize(active, scale))
        else:
            g = chunk["call_genotype"][row, sample]
            missing[sample] = (
                bool(explicit_missing[row, sample])
                if explicit_missing is not None
                else bool((g[:p] == -1).any())
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
    output = bytearray()
    buffer = 0
    available = 0
    for value in values:
        buffer |= int(value) << available
        available += bits
        while available >= 8:
            output.append(buffer & 255)
            buffer >>= 8
            available -= 8
    if available:
        output.append(buffer)
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
        + bytes(output)
    )
