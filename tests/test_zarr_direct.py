"""Tests for vcztools.zarr_direct.BlockReader.

The contract: ``BlockReader.read_chunk(coords)`` returns the same
ndarray as ``zarr.Array.blocks[coords]`` for every chunk position,
across both Zarr v2 and v3 metadata, every codec we encounter in
real fixtures, missing chunks (fill values), and boundary chunks.
Sharded arrays are refused at construction.
"""

import itertools
import warnings

import anyio
import numpy as np
import numpy.testing as nt
import pytest
import zarr
import zarr.codecs
import zarr.storage

from vcztools.zarr_direct import BlockReader


def _all_chunk_coords(arr: zarr.Array):
    """Iterate every chunk index tuple for ``arr``."""
    return itertools.product(*[range(n) for n in arr.cdata_shape])


def _assert_array_parity(arr: zarr.Array):
    """Every chunk read via BlockReader equals arr.blocks[coords]."""
    reader = BlockReader(arr)
    for coords in _all_chunk_coords(arr):
        expected = arr.blocks[coords]
        got = anyio.run(reader.read_chunk, coords)
        nt.assert_array_equal(got, expected, err_msg=f"mismatch at {coords}")
        assert got.shape == expected.shape, f"shape mismatch at {coords}"
        assert got.dtype == expected.dtype, f"dtype mismatch at {coords}"


@pytest.fixture
def synthetic_v3_group():
    """Tiny v3 group with one int array, default codecs (Bytes + Zstd)."""
    store = zarr.storage.MemoryStore()
    g = zarr.group(store=store, zarr_format=3)
    arr = g.create_array(name="ints", shape=(7, 4), chunks=(3, 2), dtype=np.int32)
    arr[:] = np.arange(28, dtype=np.int32).reshape(7, 4)
    return g


@pytest.fixture
def synthetic_v2_group():
    """v2 group with default Blosc compressor."""
    store = zarr.storage.MemoryStore()
    g = zarr.group(store=store, zarr_format=2)
    arr = g.create_array(name="ints", shape=(7, 4), chunks=(3, 2), dtype=np.int32)
    arr[:] = np.arange(28, dtype=np.int32).reshape(7, 4)
    return g


class TestBlockReaderSynthetic:
    """Parity + boundary + fill-value tests on tiny in-memory arrays."""

    def test_v3_default_codecs(self, synthetic_v3_group):
        _assert_array_parity(synthetic_v3_group["ints"])

    def test_v2_default_codecs(self, synthetic_v2_group):
        _assert_array_parity(synthetic_v2_group["ints"])

    def test_uncompressed_v3(self):
        store = zarr.storage.MemoryStore()
        g = zarr.group(store=store, zarr_format=3)
        arr = g.create_array(
            name="x",
            shape=(7,),
            chunks=(3,),
            dtype=np.int32,
            compressors=None,
            filters=None,
        )
        arr[:] = np.arange(7, dtype=np.int32)
        _assert_array_parity(arr)

    def test_vlen_strings_v3(self):
        store = zarr.storage.MemoryStore()
        g = zarr.group(store=store, zarr_format=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = np.array(
                [["hello", "world"], ["foo", ""], ["", "bar"]], dtype="<U16"
            )
            arr = g.create_array(
                name="s", shape=data.shape, chunks=(2, 2), dtype=data.dtype
            )
            arr[:] = data
        _assert_array_parity(arr)

    def test_floats(self):
        """Float dtype round-trips; NaN handling is exercised by the
        real-fixture suite (which uses ``equal_nan``-aware comparison)."""
        store = zarr.storage.MemoryStore()
        g = zarr.group(store=store, zarr_format=3)
        arr = g.create_array(name="f", shape=(5,), chunks=(2,), dtype=np.float32)
        arr[:] = np.array([1.0, 2.5, -3.0, 0.0, 7.7], dtype=np.float32)
        _assert_array_parity(arr)

    def test_boundary_chunk_trim(self):
        """Boundary chunks return the actual (trimmed) shape, not the
        nominal padded shape that's stored on disk."""
        store = zarr.storage.MemoryStore()
        g = zarr.group(store=store, zarr_format=3)
        arr = g.create_array(name="x", shape=(7,), chunks=(3,), dtype=np.int32)
        arr[:] = np.arange(7, dtype=np.int32)
        reader = BlockReader(arr)
        last = anyio.run(reader.read_chunk, (2,))
        assert last.shape == (1,)
        nt.assert_array_equal(last, [6])

    def test_missing_chunk_returns_fill_value(self):
        """A chunk key absent from the store decodes to fill-value-filled
        ndarray at the boundary-clamped shape."""
        store = zarr.storage.MemoryStore()
        g = zarr.group(store=store, zarr_format=3)
        arr = g.create_array(
            name="x", shape=(6,), chunks=(3,), dtype=np.int32, fill_value=99
        )
        # Don't write any data — every chunk key is absent.
        reader = BlockReader(arr)
        for coords in [(0,), (1,)]:
            got = anyio.run(reader.read_chunk, coords)
            expected = arr.blocks[coords]
            nt.assert_array_equal(got, expected)
            nt.assert_array_equal(got, [99, 99, 99])

    def test_chunk_key_at_root_path(self):
        """An array created at the group root has empty ``array.path``;
        the chunk key has no ``"<path>/"`` prefix."""
        store = zarr.storage.MemoryStore()
        # Create a v3 array directly under the store root (no group).
        arr = zarr.create_array(
            store=store, shape=(5,), chunks=(2,), dtype=np.int32, zarr_format=3
        )
        arr[:] = np.arange(5, dtype=np.int32)
        reader = BlockReader(arr)
        assert reader.chunk_key((0,)) == "c/0"
        nt.assert_array_equal(anyio.run(reader.read_chunk, (0,)), [0, 1])

    def test_decode_limiter_is_honoured(self):
        """A CapacityLimiter with capacity 1 must let read_chunk complete;
        we don't assert serialisation here, just that the path doesn't
        deadlock or error when a limiter is supplied."""
        store = zarr.storage.MemoryStore()
        g = zarr.group(store=store, zarr_format=3)
        arr = g.create_array(name="x", shape=(5,), chunks=(2,), dtype=np.int32)
        arr[:] = np.arange(5, dtype=np.int32)
        reader = BlockReader(arr)

        async def run():
            limiter = anyio.CapacityLimiter(1)
            return await reader.read_chunk((1,), decode_limiter=limiter)

        got = anyio.run(run)
        nt.assert_array_equal(got, [2, 3])


class TestBlockReaderShardingRefusal:
    def test_sharded_array_refused(self):
        store = zarr.storage.MemoryStore()
        g = zarr.group(store=store, zarr_format=3)
        arr = g.create_array(
            name="x",
            shape=(8, 8),
            chunks=(8, 8),
            shards=(8, 8),
            dtype=np.int32,
        )
        arr[:] = np.arange(64, dtype=np.int32).reshape(8, 8)
        with pytest.raises(NotImplementedError, match="ShardingCodec"):
            BlockReader(arr)


class TestBlockReaderRealFixtures:
    """Parity against committed VCZ fixtures — exercises the codecs and
    metadata produced by bio2zarr in real workloads (vlen UTF-8, blosc,
    zstd, transpose, NaN-bearing floats, etc.)."""

    @pytest.fixture(autouse=True)
    def _silence_zarr_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield

    def _check_full_group_parity(self, root: zarr.Group):
        """Read every chunk of every array in the group through both paths
        and assert they match. Uses ``equal_nan=True`` since real fixtures
        contain NaN-encoded missing values."""
        for name in sorted(root.array_keys()):
            arr = root[name]
            reader = BlockReader(arr)
            for coords in _all_chunk_coords(arr):
                expected = arr.blocks[coords]
                got = anyio.run(reader.read_chunk, coords)
                assert got.shape == expected.shape, (
                    f"{name}{coords}: shape {got.shape} vs {expected.shape}"
                )
                assert got.dtype == expected.dtype, (
                    f"{name}{coords}: dtype {got.dtype} vs {expected.dtype}"
                )
                if np.issubdtype(expected.dtype, np.floating):
                    nt.assert_array_equal(
                        np.where(np.isnan(expected), 0, expected),
                        np.where(np.isnan(got), 0, got),
                        err_msg=f"{name}{coords}",
                    )
                    nt.assert_array_equal(
                        np.isnan(expected),
                        np.isnan(got),
                        err_msg=f"{name}{coords} NaN mask",
                    )
                else:
                    nt.assert_array_equal(got, expected, err_msg=f"{name}{coords}")

    def test_sample_v2(self, fx_sample_vcz):
        self._check_full_group_parity(fx_sample_vcz.group)

    def test_sample_v3(self, fx_sample_vcz3):
        self._check_full_group_parity(fx_sample_vcz3.group)

    def test_field_type_combos(self, fx_field_type_combos_vcz):
        self._check_full_group_parity(fx_field_type_combos_vcz.group)
