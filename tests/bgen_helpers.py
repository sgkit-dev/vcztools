"""Small in-memory test snapshots decoded through bgen-limix's public API."""

import dataclasses
import math
import pathlib
import tempfile

import cbgen
import numpy as np


@dataclasses.dataclass
class BgenSnapshot:
    sample_count: int
    variant_count: int
    samples: list[str]
    variants: list[cbgen.VariantInfo]
    genotypes: list[cbgen.GenotypeResult]

    def metadata(self, name):
        return [getattr(variant, name) for variant in self.variants]

    @property
    def phased(self):
        return np.array([genotype.phased for genotype in self.genotypes], dtype=bool)

    @property
    def missing(self):
        return (
            np.array([genotype.missing for genotype in self.genotypes], dtype=bool)
            .reshape(self.variant_count, self.sample_count)
            .T
        )

    @property
    def probabilities(self):
        lengths = [
            [
                int(p) * variant.nalleles
                if genotype.phased
                else math.comb(int(p) + variant.nalleles - 1, int(p))
                for p in genotype.ploidy
            ]
            for variant, genotype in zip(self.variants, self.genotypes)
        ]
        width = max((max(row, default=0) for row in lengths), default=0)
        result = np.full((self.sample_count, self.variant_count, width), np.nan)
        for row, genotype in enumerate(self.genotypes):
            for sample, length in enumerate(lengths[row]):
                if not genotype.missing[sample]:
                    result[sample, row, :length] = genotype.probability[sample, :length]
        return result


def read_bgen(path):
    """Read native variant records and genotype results, closing every handle.

    Temporary native indexes avoid leaving stale sidecars when a test rewrites
    a payload at the same path. Probability matrices are padded with NaN only
    for missing calls and states outside each call's actual ploidy.
    """
    with cbgen.BgenFile(str(path)) as source:
        variants = []
        genotypes = []
        if source.nvariants:
            with tempfile.TemporaryDirectory() as directory:
                index_path = pathlib.Path(directory) / "index.metafile"
                source.create_metafile(str(index_path))
                with cbgen.BgenMetafile(str(index_path)) as index:
                    for partition in range(index.npartitions):
                        for variant in index.read_partition(partition).variants:
                            variants.append(variant)
                            genotypes.append(source.read_genotype(int(variant.offset)))
        return BgenSnapshot(
            source.nsamples,
            source.nvariants,
            source.read_samples() if source.contain_samples else [],
            variants,
            genotypes,
        )
