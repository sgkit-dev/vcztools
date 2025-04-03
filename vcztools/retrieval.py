import collections.abc


# NOTE:  this class is just a skeleton for now. The idea is that this
# will provide readahead, caching etc, and will be the central location
# for fetching bulk Zarr data.
class VariantChunkReader(collections.abc.Sequence):
    """
    Retrieve data from a Zarr store and return chunk-by-chunk in the
    variants dimension.
    """

    def __init__(self, root, *, fields=None):
        self.root = root
        if fields is None:
            fields = [
                key
                for key in root.keys()
                if key.startswith("variant_") or key.startswith("call_")
            ]
        self.arrays = {key: self.root[key] for key in fields}
        # TODO validate the arrays have the correct shapes setc
        self.num_chunks = next(iter(self.arrays.values())).cdata_shape[0]

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, chunk):
        return {key: array.blocks[chunk] for key, array in self.arrays.items()}


def variant_chunk_iter(root, fields=None, variant_select=None):
    chunk_reader = VariantChunkReader(root, fields=fields)
    for chunk in range(len(chunk_reader)):
        yield chunk_reader[chunk]
