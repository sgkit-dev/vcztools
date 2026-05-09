"""Direct chunk fetch + decode without Zarr's high-level Array layer.

Owns the path from ``(zarr.Array, chunk_coords)`` to ``np.ndarray``:

    raw bytes from store.get  →  codec pipeline.decode  →  np.ndarray

Lets the caller drive concurrency with anyio (async store gets) and a
bounded decode pool (``anyio.CapacityLimiter``) instead of going through
``arr.blocks[idx]``, which spins up a fresh asyncio loop per call.

Sharded arrays are explicitly unsupported. ``BlockReader`` rejects them
at construction with a clear error.
"""

import logging

import anyio
import numpy as np
import zarr
from zarr.abc.buffer import Buffer
from zarr.codecs.sharding import ShardingCodec
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import default_buffer_prototype

logger = logging.getLogger(__name__)


class BlockReader:
    """Resolves a single Zarr array to its store, key encoding, codec
    pipeline, and chunk-spec factory. Constructed once per field and
    reused across chunk reads.

    The output of :meth:`read_chunk` matches ``array.blocks[coords]``
    byte-for-byte: boundary chunks are clamped to the actual data
    shape, missing chunks materialise as ``fill_value``.
    """

    def __init__(self, array: zarr.Array):
        async_array = array.async_array
        codec_pipeline = async_array.codec_pipeline
        if isinstance(codec_pipeline.array_bytes_codec, ShardingCodec):
            raise NotImplementedError(
                f"Array {array.path!r} uses ShardingCodec, which is not "
                "supported by vcztools. Re-encode the VCZ without sharding "
                "(bio2zarr writes non-sharded VCZ by default)."
            )

        self._store = array.store_path.store
        self._path = array.path
        self._metadata = array.metadata
        self._codec_pipeline = codec_pipeline
        self._array_config = async_array.config
        self._prototype = default_buffer_prototype()
        self._shape = array.shape
        self._chunk_shape = array.chunks
        self._cdata_shape = tuple(
            (s + c - 1) // c for s, c in zip(array.shape, array.chunks)
        )
        self._dtype = array.dtype
        # Zarr metadata may record fill_value=None (especially in v2);
        # fall back to the dtype's default scalar (0 for ints, "" for
        # strings, etc.) so missing-chunk decode matches arr.blocks[].
        fill = array.metadata.fill_value
        if fill is None:
            fill = array.metadata.dtype.default_scalar()
        self._fill_value = fill

    @property
    def cdata_shape(self) -> tuple[int, ...]:
        """Number of chunks per axis (the size of the chunk grid)."""
        return self._cdata_shape

    def chunk_key(self, coords: tuple[int, ...]) -> str:
        """Store key for the chunk at ``coords``."""
        suffix = self._metadata.encode_chunk_key(coords)
        if self._path == "":
            return suffix
        return f"{self._path}/{suffix}"

    def chunk_spec(self, coords: tuple[int, ...]) -> ArraySpec:
        return self._metadata.get_chunk_spec(
            coords, self._array_config, self._prototype
        )

    def actual_chunk_shape(self, coords: tuple[int, ...]) -> tuple[int, ...]:
        """Boundary-clamped chunk shape — what ``arr.blocks[coords]`` returns.

        Boundary chunks are stored at the nominal chunk shape (padded with
        fill values); ``arr.blocks[]`` slices them back to the actual
        data extent. We mirror that here.
        """
        result = []
        for d, coord in enumerate(coords):
            stored = self._chunk_shape[d]
            remaining = self._shape[d] - coord * self._chunk_shape[d]
            result.append(max(0, min(stored, remaining)))
        return tuple(result)

    async def fetch_chunk_bytes(self, coords: tuple[int, ...]) -> Buffer | None:
        """Fetch raw chunk bytes from the store; ``None`` if absent."""
        key = self.chunk_key(coords)
        return await self._store.get(key, prototype=self._prototype)

    async def decode_chunk(
        self, raw: Buffer | None, coords: tuple[int, ...]
    ) -> np.ndarray:
        """Decode raw chunk bytes to a numpy array.

        Returns the boundary-clamped shape. ``None`` raw materialises
        fill values at the actual shape.
        """
        actual_shape = self.actual_chunk_shape(coords)
        if raw is None:
            return np.full(actual_shape, self._fill_value, dtype=self._dtype)
        spec = self.chunk_spec(coords)
        decoded = list(await self._codec_pipeline.decode([(raw, spec)]))
        nd = decoded[0].as_numpy_array()
        if nd.shape != actual_shape:
            nd = nd[tuple(slice(0, s) for s in actual_shape)]
        return nd

    async def read_chunk(
        self,
        coords: tuple[int, ...],
        decode_limiter: anyio.CapacityLimiter | None = None,
    ) -> np.ndarray:
        """Fetch and decode a single chunk.

        ``decode_limiter`` bounds concurrent decodes when many readers
        share a thread budget; pass ``None`` for unbounded.
        """
        raw = await self.fetch_chunk_bytes(coords)
        if decode_limiter is None:
            return await self.decode_chunk(raw, coords)
        async with decode_limiter:
            return await self.decode_chunk(raw, coords)
