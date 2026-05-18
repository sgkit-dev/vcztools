import sys

import numpy as np
import pytest

from vcztools import _vcztools

FIXED_FIELD_NAMES = ["chrom", "pos", "id", "qual", "ref", "alt", "filter"]


def example_fixed_data(num_variants, num_samples=0):
    chrom = np.array(["X"] * num_variants, dtype="S")
    pos = np.arange(num_variants, dtype=np.int32)
    id = np.array(["."] * num_variants, dtype="S").reshape((num_variants, 1))
    ref = np.array(["A"] * num_variants, dtype="S")
    alt = np.array(["T"] * num_variants, dtype="S").reshape((num_variants, 1))
    qual = np.arange(num_variants, dtype=np.float32)
    filter_ = np.ones(num_variants, dtype=bool).reshape((num_variants, 1))
    filter_id = np.array(["PASS"], dtype="S")
    return {
        "chrom": chrom,
        "pos": pos,
        "id": id,
        "qual": qual,
        "ref": ref,
        "alt": alt,
        "filter": filter_,
        "filter_ids": filter_id,
    }


def example_info_data(num_variants):
    d = {}
    for num_columns in range(1, 3):
        for dtype in ["i1", "i2", "i4", "f4", "S1"]:
            name = f"I{dtype}_{num_columns}"
            data = np.arange(num_variants * num_columns).astype(dtype)
            d[name] = data.reshape((num_variants, num_columns))
    a = np.arange(num_variants, dtype=np.int8).reshape((num_variants, 1)) % 2
    d["FLAG"] = a.astype(bool)
    return d


def example_format_data(num_variants, num_samples):
    d = {}
    for num_columns in range(1, 3):
        for dtype in ["i1", "i2", "i4", "f4", "S1"]:
            name = f"F{dtype}_{num_columns}"
            data = np.arange(num_variants * num_samples * num_columns).astype(dtype)
            d[name] = data.reshape((num_variants, num_samples, num_columns))
    return d


def example_gt_data(num_variants, num_samples, ploidy=2):
    return {
        "gt": np.arange(num_variants * num_samples * ploidy)
        .astype(np.int32)
        .reshape(num_variants, num_samples, ploidy),
        "gt_phased": (np.arange(num_variants * num_samples) % 2)
        .reshape(num_variants, num_samples)
        .astype(bool),
    }


def example_encoder(num_variants=1, num_samples=0, add_info=True):
    encoder = _vcztools.VcfEncoder(
        num_variants, num_samples, **example_fixed_data(num_variants, num_samples)
    )
    if add_info:
        for name, data in example_info_data(num_variants).items():
            encoder.add_info_field(name, data)
    if num_samples > 0:
        for name, data in example_format_data(num_variants, num_samples).items():
            encoder.add_format_field(name, data)

        gt_data = example_gt_data(num_variants, num_samples)
        encoder.add_gt_field(gt_data["gt"], gt_data["gt_phased"])
    # import sys
    # encoder.print_state(sys.stdout)
    return encoder


@pytest.mark.skipif(sys.platform == "win32", reason="Not implemented on Windows")
class TestPrintState:
    def test_nomimal_case(self, tmp_path):
        encoder = example_encoder()
        filename = tmp_path / "debug.txt"
        with open(filename, "w") as f:
            encoder.print_state(f)
        with open(filename) as f:
            s = f.read()
        assert "CHROM" in s

    def test_read_only_file(self, tmp_path):
        encoder = example_encoder()
        filename = tmp_path / "debug.txt"
        with open(filename, "w") as f:
            f.write("x")
        with open(filename) as f:
            with pytest.raises(OSError, match="22"):
                encoder.print_state(f)

    def test_no_arg(self):
        encoder = example_encoder()
        with pytest.raises(TypeError, match="function takes"):
            encoder.print_state()

    @pytest.mark.parametrize("fileobj", [None, "path"])
    def test_bad_file_arg(self, fileobj):
        encoder = example_encoder()
        with pytest.raises(TypeError):
            encoder.print_state(fileobj)

    @pytest.mark.parametrize("fileobj", [1024, 123])
    def test_bad_fileno(self, fileobj):
        encoder = example_encoder()
        with pytest.raises(OSError, match="9"):
            encoder.print_state(fileobj)


class TestEncode:
    @pytest.mark.parametrize("bad_arg", [None, {}, "0"])
    def test_bad_index_arg(self, bad_arg):
        encoder = example_encoder()
        with pytest.raises(TypeError, match="int"):
            encoder.encode(bad_arg, 1)

    @pytest.mark.parametrize("bad_arg", [None, {}, "0"])
    def test_bad_len_arg(self, bad_arg):
        encoder = example_encoder()
        with pytest.raises(TypeError, match="int"):
            encoder.encode(0, bad_arg)

    @pytest.mark.parametrize("bad_row", [2, 3, 100, -1, 2**32, 2**63])
    def test_bad_variant_arg(self, bad_row):
        encoder = example_encoder(2)
        with pytest.raises(ValueError, match="102"):
            encoder.encode(bad_row, 1)

    def test_buffer_malloc_fail(self):
        encoder = example_encoder(1)
        with pytest.raises(MemoryError):
            encoder.encode(0, 2**63)

    def test_small_example_overrun(self):
        encoder = example_encoder(1, 2)
        s = encoder.encode(0, 1024)
        minlen = len(s)
        for length in range(minlen):
            with pytest.raises(_vcztools.VczBufferTooSmall, match="-101"):
                encoder.encode(0, length)
        assert s == encoder.encode(0, minlen)

    @pytest.mark.parametrize("variant", [0, 999, 333, 501])
    def test_large_example_overrun(self, variant):
        encoder = example_encoder(1000, 100)
        s = encoder.encode(variant, 1024 * 1024)
        minlen = len(s)
        for length in range(minlen):
            with pytest.raises(_vcztools.VczBufferTooSmall, match="-101"):
                encoder.encode(variant, length)
        assert s == encoder.encode(variant, minlen)


class TestFixedFieldInputChecking:
    @pytest.mark.parametrize("name", FIXED_FIELD_NAMES)
    def test_field_incorrect_length(self, name):
        num_variants = 5
        data = example_fixed_data(num_variants)
        data[name] = data[name][1:]
        with pytest.raises(ValueError, match=f"Array {name} must have "):
            _vcztools.VcfEncoder(num_variants, 0, **data)

    @pytest.mark.parametrize("name", [*FIXED_FIELD_NAMES, "filter_ids"])
    def test_field_incorrect_dtype(self, name):
        num_variants = 5
        data = example_fixed_data(num_variants)
        data[name] = np.zeros(data[name].shape, dtype=np.int64)
        with pytest.raises(ValueError, match=f"Wrong dtype for {name}"):
            _vcztools.VcfEncoder(num_variants, 0, **data)

    @pytest.mark.parametrize("name", [*FIXED_FIELD_NAMES, "filter_ids"])
    def test_field_incorrect_dimension(self, name):
        num_variants = 2
        data = example_fixed_data(num_variants)
        data[name] = np.expand_dims(data[name], -1)
        with pytest.raises(ValueError, match=f"{name} has wrong dimension"):
            _vcztools.VcfEncoder(num_variants, 0, **data)

    @pytest.mark.parametrize("name", [*FIXED_FIELD_NAMES, "filter_ids"])
    def test_field_incorrect_type(self, name):
        num_variants = 5
        data = example_fixed_data(num_variants)
        data[name] = "A Python string"
        with pytest.raises(TypeError, match="must be numpy.ndarray"):
            _vcztools.VcfEncoder(num_variants, 0, **data)

    @pytest.mark.parametrize("name", ["id", "alt"])
    def test_zero_columns(self, name):
        data = example_fixed_data(1)
        arr = np.frombuffer(b"", dtype="S1").reshape(1, 0)
        data[name] = arr
        with pytest.raises(ValueError, match="-204"):
            _vcztools.VcfEncoder(1, 0, **data)

    def test_zero_filter_ids(self):
        data = example_fixed_data(1)
        data["filter_ids"] = np.array([], dtype="S")
        data["filter"] = np.array([], dtype=bool).reshape(1, 0)
        with pytest.raises(ValueError, match="-204"):
            _vcztools.VcfEncoder(1, 0, **data)

    def test_incorrect_num_filter_ids(self):
        data = example_fixed_data(1)
        data["filter_ids"] = np.array(["PASS", "1"], dtype="S")
        with pytest.raises(ValueError, match="filters dimension must be"):
            _vcztools.VcfEncoder(1, 0, **data)


class TestAddFields:
    def test_add_info_field_bad_num_args(self):
        encoder = example_encoder()
        with pytest.raises(TypeError):
            encoder.add_info_field()
        with pytest.raises(TypeError):
            encoder.add_info_field("name")

    def test_add_format_field_bad_num_args(self):
        encoder = example_encoder()
        with pytest.raises(TypeError):
            encoder.add_format_field()
        with pytest.raises(TypeError):
            encoder.add_format_field("name")

    def test_add_gt_field_bad_num_args(self):
        encoder = example_encoder()
        with pytest.raises(TypeError):
            encoder.add_gt_field()
        with pytest.raises(TypeError):
            encoder.add_gt_field(np.zeros((1), dtype=bool))

    @pytest.mark.parametrize("name", [None, 1234, {}])
    def test_add_info_field_bad_name_type(self, name):
        encoder = example_encoder()
        with pytest.raises(TypeError, match="must be str"):
            encoder.add_info_field(name, np.array([1]))

    @pytest.mark.parametrize("name", [None, 1234, {}])
    def test_add_format_field_bad_name_type(self, name):
        encoder = example_encoder()
        with pytest.raises(TypeError, match="must be str"):
            encoder.add_format_field(name, np.array([1]))

    @pytest.mark.parametrize("array", [None, 1234, {}])
    def test_add_info_field_bad_array_type(self, array):
        encoder = example_encoder()
        with pytest.raises(TypeError, match="ndarray"):
            encoder.add_info_field("name", array)

    @pytest.mark.parametrize("array", [None, 1234, {}])
    def test_add_format_field_bad_array_type(self, array):
        encoder = example_encoder()
        with pytest.raises(TypeError, match="ndarray"):
            encoder.add_format_field("name", array)

    @pytest.mark.parametrize("array", [None, 1234, {}])
    def test_add_gt_phased_field_bad_array_type(self, array):
        encoder = example_encoder(1, 1)
        with pytest.raises(TypeError, match="ndarray"):
            encoder.add_gt_field(np.zeros((1, 1, 1), dtype=np.int8), array)

    @pytest.mark.parametrize("array", [None, 1234, {}])
    def test_add_gt_field_bad_array_type(self, array):
        encoder = example_encoder(1, 1)
        with pytest.raises(TypeError, match="ndarray"):
            encoder.add_gt_field(array, np.zeros((1, 1), dtype=bool))

    def test_add_info_field_bad_array_num_variants(self):
        num_variants = 10
        encoder = example_encoder(num_variants)
        with pytest.raises(ValueError, match="number of variants"):
            encoder.add_info_field("name", np.zeros((3, 1), dtype=bool))

    def test_add_format_field_bad_array_num_variants(self):
        num_variants = 10
        encoder = example_encoder(num_variants)
        with pytest.raises(ValueError, match="number of variants"):
            encoder.add_format_field("name", np.zeros((3, 1, 1), dtype=bool))

    def test_add_gt_field_bad_gt_array_num_variants(self):
        num_variants = 10
        encoder = example_encoder(num_variants, 1)
        with pytest.raises(ValueError, match="number of variants"):
            encoder.add_gt_field(
                np.zeros((2, 1, 1), dtype=np.int8), np.zeros((10, 1), dtype=bool)
            )

    def test_add_gt_field_bad_gt_phased_array_num_variants(self):
        num_variants = 10
        encoder = example_encoder(num_variants, 1)
        with pytest.raises(ValueError, match="number of variants"):
            encoder.add_gt_field(
                np.zeros((10, 1, 1), dtype=np.int8), np.zeros((2, 1), dtype=bool)
            )

    def test_add_format_field_bad_array_num_samples(self):
        encoder = example_encoder(5, 2)
        with pytest.raises(ValueError, match="number of samples"):
            encoder.add_format_field("name", np.zeros((5, 1, 1), dtype=bool))

    def test_add_gt_field_bad_gt_array_num_samples(self):
        encoder = example_encoder(5, 2)
        with pytest.raises(ValueError, match="number of samples"):
            encoder.add_gt_field(
                np.zeros((5, 1, 1), dtype=np.int8), np.zeros((5, 2), dtype=bool)
            )

    def test_add_gt_field_bad_gt_phased_array_num_samples(self):
        encoder = example_encoder(5, 2)
        with pytest.raises(ValueError, match="number of samples"):
            encoder.add_gt_field(
                np.zeros((5, 2, 1), dtype=np.int8), np.zeros((5, 1), dtype=bool)
            )

    @pytest.mark.parametrize("shape", [(), (5,), (5, 1, 1)])
    def test_add_info_field_wrong_dimension(self, shape):
        encoder = example_encoder(5)
        with pytest.raises(ValueError, match="wrong dimension"):
            encoder.add_info_field("name", np.zeros(shape, dtype=bool))

    @pytest.mark.parametrize("shape", [(), (5,), (5, 1), (5, 1, 1, 1)])
    def test_add_format_field_wrong_dimension(self, shape):
        encoder = example_encoder(5)
        with pytest.raises(ValueError, match="wrong dimension"):
            encoder.add_format_field("name", np.zeros(shape, dtype=bool))

    @pytest.mark.parametrize("shape", [(), (5,), (5, 2), (5, 2, 1, 1)])
    def test_add_gt_field_gt_wrong_dimension(self, shape):
        encoder = example_encoder(5, 2)
        with pytest.raises(ValueError, match="wrong dimension"):
            encoder.add_gt_field(
                np.zeros(shape, dtype=np.int8), np.zeros((5, 2), dtype=bool)
            )

    @pytest.mark.parametrize("shape", [(), (5,), (5, 2, 1), (5, 2, 1, 1)])
    def test_add_gt_field_gt_phased_wrong_dimension(self, shape):
        encoder = example_encoder(5, 2)
        with pytest.raises(ValueError, match="wrong dimension"):
            encoder.add_gt_field(
                np.zeros((5, 2, 1), dtype=np.int8), np.zeros(shape, dtype=bool)
            )

    @pytest.mark.parametrize("dtype", ["m", "c16", "O", "u1"])
    def test_add_info_field_unsupported_dtype(self, dtype):
        encoder = example_encoder(1)
        with pytest.raises(ValueError, match="unsupported array dtype"):
            encoder.add_info_field("name", np.zeros((1, 1), dtype=dtype))

    @pytest.mark.parametrize("dtype", ["m", "c16", "O", "u1"])
    def test_add_format_field_unsupported_dtype(self, dtype):
        encoder = example_encoder(1, 1)
        with pytest.raises(ValueError, match="unsupported array dtype"):
            encoder.add_format_field("name", np.zeros((1, 1, 1), dtype=dtype))

    @pytest.mark.parametrize("dtype", ["m", "c16", "O", "u1"])
    def test_add_gt_field_gt_unsupported_dtype(self, dtype):
        encoder = example_encoder(1, 1)
        with pytest.raises(ValueError, match="unsupported array dtype"):
            encoder.add_gt_field(
                np.zeros((1, 1, 1), dtype=dtype), np.zeros((1, 1), dtype=bool)
            )

    @pytest.mark.parametrize("dtype", ["m", "c16", "O", "u1"])
    def test_add_gt_field_gt_phased_unsupported_dtype(self, dtype):
        encoder = example_encoder(1, 1)
        with pytest.raises(ValueError, match="Wrong dtype for gt_phased"):
            encoder.add_gt_field(
                np.zeros((1, 1, 1), dtype=np.int8), np.zeros((1, 1), dtype=dtype)
            )

    @pytest.mark.parametrize("dtype", ["i8", "f8"])
    def test_add_info_field_unsupported_width(self, dtype):
        encoder = example_encoder(1)
        with pytest.raises(ValueError, match="-203"):
            encoder.add_info_field("name", np.zeros((1, 1), dtype=dtype))

    @pytest.mark.parametrize("dtype", ["i8", "f8"])
    def test_add_format_field_unsupported_width(self, dtype):
        encoder = example_encoder(1, 1)
        with pytest.raises(ValueError, match="-203"):
            encoder.add_format_field("name", np.zeros((1, 1, 1), dtype=dtype))

    def test_add_gt_field_unsupported_width(self):
        encoder = example_encoder(1, 1)
        with pytest.raises(ValueError, match="-203"):
            encoder.add_gt_field(
                np.zeros((1, 1, 1), dtype=np.int64), np.zeros((1, 1), dtype=bool)
            )

    def test_add_gt_field_zero_ploidy(self):
        encoder = example_encoder(1, 1)
        with pytest.raises(ValueError, match="-204"):
            encoder.add_gt_field(
                np.zeros((1, 1, 0), dtype=np.int64), np.zeros((1, 1), dtype=bool)
            )


class TestArrays:
    def test_stored_data_equal(self):
        num_variants = 20
        num_samples = 10
        fixed_data = example_fixed_data(num_variants, num_samples)
        gt_data = example_gt_data(num_variants, num_samples)
        encoder = _vcztools.VcfEncoder(num_variants, num_samples, **fixed_data)
        encoder.add_gt_field(gt_data["gt"], gt_data["gt_phased"])
        info_data = example_info_data(num_variants)
        format_data = example_format_data(num_variants, num_samples)
        for name, data in info_data.items():
            encoder.add_info_field(name, data)
        for name, data in format_data.items():
            encoder.add_format_field(name, data)
        all_data = {**fixed_data}
        for name, array in info_data.items():
            all_data[f"INFO/{name}"] = array
        for name, array in {**format_data, **gt_data}.items():
            all_data[f"FORMAT/{name}"] = array
        encoder_arrays = encoder.arrays
        assert set(encoder.arrays.keys()) == set(all_data.keys())
        for name in all_data.keys():
            # Strong identity assertion here for now, but this may change
            assert all_data[name] is encoder_arrays[name]

    def test_array_bad_alignment(self):
        fixed_data = example_fixed_data(1)
        a = fixed_data["pos"]
        # Value is not aligned
        a = np.frombuffer(b"\0\0\0\0\0", dtype=np.int32, offset=1)
        fixed_data["pos"] = a
        with pytest.raises(ValueError, match="NPY_ARRAY_IN_ARRAY"):
            _vcztools.VcfEncoder(1, 0, **fixed_data)


class TestUninitialised:
    @pytest.mark.parametrize(
        "name",
        [
            "add_info_field",
            "add_gt_field",
            "add_format_field",
            pytest.param(
                "print_state",
                marks=pytest.mark.skipif(
                    sys.platform == "win32", reason="Not implemented on Windows"
                ),
            ),
            "encode",
        ],
    )
    def test_methods(self, name):
        cls = _vcztools.VcfEncoder
        encoder = cls.__new__(cls)
        method = getattr(encoder, name)
        with pytest.raises(SystemError, match="not initialised"):
            method()

    @pytest.mark.parametrize("name", ["arrays"])
    def test_attrs(self, name):
        cls = _vcztools.VcfEncoder
        encoder = cls.__new__(cls)
        with pytest.raises(SystemError, match="not initialised"):
            getattr(encoder, name)


class TestEncodePlink:
    def _out_buf(self, num_variants, num_samples):
        return np.zeros(((num_samples + 3) // 4) * num_variants, dtype=np.uint8)

    def test_bad_num_arguments(self):
        with pytest.raises(TypeError):
            _vcztools.encode_plink()
        with pytest.raises(TypeError):
            _vcztools.encode_plink(np.zeros((1, 1, 2), dtype=np.int8))

    @pytest.mark.parametrize("bad_type", [[], {}, "string", 4])
    def test_bad_types(self, bad_type):
        with pytest.raises(TypeError):
            _vcztools.encode_plink(bad_type, self._out_buf(1, 1))

    def test_wrong_genotype_dims(self):
        with pytest.raises(ValueError, match="genotypes has wrong dimension"):
            _vcztools.encode_plink(np.zeros((1, 1), dtype=np.int8), self._out_buf(1, 1))

    @pytest.mark.parametrize("bad_dtype", [np.int16, np.int64, np.float64, "S1"])
    def test_bad_genotype_dtype(self, bad_dtype):
        with pytest.raises(ValueError, match="Wrong dtype for genotypes"):
            _vcztools.encode_plink(
                np.zeros((1, 1, 2), dtype=bad_dtype), self._out_buf(1, 1)
            )

    @pytest.mark.parametrize("bad_ploidy", [1, 3])
    def test_bad_ploidy(self, bad_ploidy):
        with pytest.raises(ValueError, match="Only diploid genotypes"):
            _vcztools.encode_plink(
                np.zeros((1, 1, bad_ploidy), dtype=np.int8), self._out_buf(1, 1)
            )

    def test_buffer_too_small(self):
        G = np.zeros((1, 4, 2), dtype=np.int8)
        with pytest.raises(_vcztools.VczBufferTooSmall):
            _vcztools.encode_plink(G, np.zeros(0, dtype=np.uint8))

    @pytest.mark.parametrize("bad_type", [[], {}, "string", 4])
    def test_out_buf_bad_type(self, bad_type):
        with pytest.raises(TypeError):
            _vcztools.encode_plink(np.zeros((1, 1, 2), dtype=np.int8), bad_type)

    def test_out_buf_wrong_dim(self):
        with pytest.raises(ValueError, match="out_buf has wrong dimension"):
            _vcztools.encode_plink(
                np.zeros((1, 1, 2), dtype=np.int8), np.zeros((1, 1), dtype=np.uint8)
            )

    @pytest.mark.parametrize("bad_dtype", [np.int8, np.int32, np.float32])
    def test_out_buf_wrong_dtype(self, bad_dtype):
        with pytest.raises(ValueError, match="Wrong dtype for out_buf"):
            _vcztools.encode_plink(
                np.zeros((1, 4, 2), dtype=np.int8), np.zeros(1, dtype=bad_dtype)
            )

    def test_non_contiguous_out_buf(self):
        full = np.zeros(16, dtype=np.uint8)
        with pytest.raises(ValueError, match="NPY_ARRAY_IN_ARRAY"):
            _vcztools.encode_plink(np.zeros((1, 4, 2), dtype=np.int8), full[::2])

    def test_read_only_out_buf(self):
        # NPY_ARRAY_IN_ARRAY = C_CONTIGUOUS | ALIGNED; it does NOT include
        # WRITEABLE. A read-only buffer would otherwise pass check_array
        # and have the kernel write to read-only pages.
        out = self._out_buf(1, 4)
        out.setflags(write=False)
        with pytest.raises(ValueError, match="out_buf must be writeable"):
            _vcztools.encode_plink(np.zeros((1, 4, 2), dtype=np.int8), out)

    def test_non_contiguous_genotypes(self):
        full = np.zeros((2, 4, 2), dtype=np.int8)
        view = full[:, ::2, :]
        with pytest.raises(ValueError, match="NPY_ARRAY_IN_ARRAY"):
            _vcztools.encode_plink(view, self._out_buf(2, 2))

    def test_returns_none(self):
        # Wrapper now returns None (the caller owns out_buf).
        G = np.zeros((1, 4, 2), dtype=np.int8)
        assert _vcztools.encode_plink(G, self._out_buf(1, 4)) is None

    def test_example(self):
        G = np.array(
            [
                [[0, 0], [0, 1], [0, 0]],
                [[1, 0], [1, 1], [0, -2]],
                [[1, 1], [0, 1], [-1, -1]],
            ],
            dtype=np.int8,
        )
        enc = self._out_buf(3, 3)
        _vcztools.encode_plink(G, enc)
        assert list(enc) == [59, 50, 24]


class TestEncodeBgenGenoBlocks:
    def _phased(self, n, value=False):
        return np.full(n, value, dtype=bool)

    def test_bad_num_arguments(self):
        with pytest.raises(TypeError):
            _vcztools.encode_bgen_geno_blocks()
        with pytest.raises(TypeError):
            _vcztools.encode_bgen_geno_blocks(
                np.zeros((1, 1, 2), dtype=np.int8),
            )
        with pytest.raises(TypeError):
            _vcztools.encode_bgen_geno_blocks(
                np.zeros((1, 1, 2), dtype=np.int8),
                self._phased(1),
                np.zeros(1),
            )

    @pytest.mark.parametrize("bad_type", [[], {}, "string", 4])
    def test_bad_genotype_type(self, bad_type):
        with pytest.raises(TypeError):
            _vcztools.encode_bgen_geno_blocks(bad_type, self._phased(1))

    @pytest.mark.parametrize("bad_type", [[], {}, "string", 4])
    def test_bad_phased_type(self, bad_type):
        with pytest.raises(TypeError):
            _vcztools.encode_bgen_geno_blocks(
                np.zeros((1, 1, 2), dtype=np.int8), bad_type
            )

    def test_wrong_genotype_dims(self):
        with pytest.raises(ValueError, match="genotypes has wrong dimension"):
            _vcztools.encode_bgen_geno_blocks(
                np.zeros((1, 1), dtype=np.int8), self._phased(1)
            )

    def test_wrong_phased_dims(self):
        with pytest.raises(ValueError, match="phased has wrong dimension"):
            _vcztools.encode_bgen_geno_blocks(
                np.zeros((1, 1, 2), dtype=np.int8),
                np.zeros((1, 1), dtype=bool),
            )

    @pytest.mark.parametrize("bad_dtype", [np.int16, np.int64, np.float64, "S1"])
    def test_bad_genotype_dtype(self, bad_dtype):
        with pytest.raises(ValueError, match="Wrong dtype for genotypes"):
            _vcztools.encode_bgen_geno_blocks(
                np.zeros((1, 1, 2), dtype=bad_dtype), self._phased(1)
            )

    @pytest.mark.parametrize("bad_dtype", [np.int8, np.uint8, np.int32, np.float32])
    def test_bad_phased_dtype(self, bad_dtype):
        with pytest.raises(ValueError, match="Wrong dtype for phased"):
            _vcztools.encode_bgen_geno_blocks(
                np.zeros((1, 1, 2), dtype=np.int8),
                np.zeros(1, dtype=bad_dtype),
            )

    @pytest.mark.parametrize("bad_ploidy", [1, 3])
    def test_bad_ploidy(self, bad_ploidy):
        # The C wrapper only accepts (V, S, 2); shape (V, S, 1) is
        # promoted to (V, S, 2) with -2 padding by the Python layer.
        with pytest.raises(ValueError, match=r"\(V, S, 2\)"):
            _vcztools.encode_bgen_geno_blocks(
                np.zeros((1, 1, bad_ploidy), dtype=np.int8),
                self._phased(1),
            )

    def test_zero_ploidy_in_slot_0_raises(self):
        # -2 in slot 0 has no BGEN representation; the kernel returns
        # an error code mapped to ValueError.
        G = np.array([[[-2, 0]]], dtype=np.int8)
        with pytest.raises(ValueError, match="zero-ploidy"):
            _vcztools.encode_bgen_geno_blocks(G, self._phased(1))

    @pytest.mark.parametrize(
        "bad",
        [
            (2, 0),
            (0, 2),
            (127, -2),
            (-3, 0),
            (0, -3),
            (-128, 1),
        ],
    )
    def test_invalid_allele_raises(self, bad):
        # Anything outside {-2, -1, 0, 1} is a biallelic data-quality
        # error; the kernel returns VCZ_ERR_BGEN_INVALID_ALLELE.
        G = np.array([[[bad[0], bad[1]]]], dtype=np.int8)
        with pytest.raises(ValueError, match="out of range"):
            _vcztools.encode_bgen_geno_blocks(G, self._phased(1))

    def test_phased_variant_mismatch(self):
        with pytest.raises(
            ValueError,
            match=r"phased\.shape\[0\] must equal genotypes\.shape\[0\]",
        ):
            _vcztools.encode_bgen_geno_blocks(
                np.zeros((3, 2, 2), dtype=np.int8), self._phased(2)
            )

    def test_non_contiguous_genotypes(self):
        full = np.zeros((2, 4, 2), dtype=np.int8)
        view = full[:, ::2, :]
        with pytest.raises(ValueError, match="NPY_ARRAY_IN_ARRAY"):
            _vcztools.encode_bgen_geno_blocks(view, self._phased(2))

    def test_non_contiguous_phased(self):
        G = np.zeros((2, 1, 2), dtype=np.int8)
        full_phased = np.zeros(4, dtype=bool)
        view = full_phased[::2]
        with pytest.raises(ValueError, match="NPY_ARRAY_IN_ARRAY"):
            _vcztools.encode_bgen_geno_blocks(G, view)

    def test_output_shape_and_dtype(self):
        G = np.zeros((3, 4, 2), dtype=np.int8)
        buf, lens = _vcztools.encode_bgen_geno_blocks(G, self._phased(3))
        assert buf.dtype == np.uint8
        assert buf.shape == (3, 10 + 3 * 4)
        assert lens.dtype == np.uint32
        assert lens.shape == (3,)
        # All-zero diploid: every variant fills the worst-case row.
        assert (lens == 10 + 3 * 4).all()

    def test_example_unphased(self):
        # Mirrors tests/test_bgen.py::test_unphased_basic.
        G = np.array([[[0, 0], [0, 1], [1, 1]]], dtype=np.int8)
        buf, lens = _vcztools.encode_bgen_geno_blocks(G, self._phased(1))
        assert lens[0] == 19
        assert bytes(buf[0, 0:8]) == bytes([3, 0, 0, 0, 2, 0, 2, 2])
        assert bytes(buf[0, 8:11]) == bytes([0x02, 0x02, 0x02])
        assert buf[0, 11] == 0
        assert buf[0, 12] == 8
        assert bytes(buf[0, 13:19]) == bytes([0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00])

    def test_example_phased(self):
        G = np.array([[[0, 0], [0, 1], [1, 0], [1, 1]]], dtype=np.int8)
        buf, lens = _vcztools.encode_bgen_geno_blocks(G, self._phased(1, True))
        assert lens[0] == 22
        assert buf[0, 12] == 1
        assert bytes(buf[0, 14:22]) == bytes(
            [0xFF, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00]
        )

    def test_missing_pattern(self):
        G = np.array([[[-1, -1], [0, -1], [0, 1]]], dtype=np.int8)
        buf, lens = _vcztools.encode_bgen_geno_blocks(G, self._phased(1))
        assert lens[0] == 19
        assert bytes(buf[0, 8:11]) == bytes([0x82, 0x82, 0x02])
        assert bytes(buf[0, 13:19]) == bytes([0, 0, 0, 0, 0, 0xFF])

    def test_mixed_ploidy(self):
        # 2 samples: diploid (0, 0), haploid (1, -2). Block length:
        # 8 + 2 ploidy + 2 flags + (2 + 1) probs = 15.
        G = np.array([[[0, 0], [1, -2]]], dtype=np.int8)
        buf, lens = _vcztools.encode_bgen_geno_blocks(G, self._phased(1))
        assert lens[0] == 15
        # Pmin=1, Pmax=2.
        assert bytes(buf[0, 0:8]) == bytes([2, 0, 0, 0, 2, 0, 1, 2])
        assert bytes(buf[0, 8:10]) == bytes([0x02, 0x01])
        assert bytes(buf[0, 12:15]) == bytes([0xFF, 0x00, 0x00])

    def test_zero_samples(self):
        G = np.zeros((2, 0, 2), dtype=np.int8)
        buf, lens = _vcztools.encode_bgen_geno_blocks(G, self._phased(2))
        assert buf.shape == (2, 10)
        assert (lens == 10).all()
        for row in buf:
            assert bytes(row[0:8]) == bytes([0, 0, 0, 0, 2, 0, 2, 2])
            assert row[8] == 0  # phased flag
            assert row[9] == 8  # B

    def test_zero_variants(self):
        G = np.zeros((0, 5, 2), dtype=np.int8)
        buf, lens = _vcztools.encode_bgen_geno_blocks(G, self._phased(0))
        assert buf.shape == (0, 25)
        assert lens.shape == (0,)


# ---------------------------------------------------------------------------
# encode_bgen_chunk_slice_level0
# ---------------------------------------------------------------------------


class _ChunkSliceArgs:
    """Builder for valid encode_bgen_chunk_slice_level0 args.

    Each attribute can be overridden via the constructor; the
    ``as_tuple()`` method returns the positional argument tuple the C
    wrapper expects. Tests that exercise a single error path build a
    valid baseline and mutate exactly one field.

    The five string fields are passed as ``(N, item_size)`` uint8 rows
    (S-dtype views). The C kernel scans each row to find the actual
    UTF-8 byte length, sums them, and verifies the per-variant sum
    matches ``total_string_length``.
    """

    def __init__(
        self,
        num_variants=1,
        num_samples=1,
        uniform_ploidy=2,
        # Default: each string field is one "." byte per variant ⇒
        # per-variant string sum = 5 bytes.
        total_string_length=5,
        varid=None,
        rsid=None,
        chrom=None,
        allele1=None,
        allele2=None,
        position=None,
        genotypes=None,
        phased=None,
        out_buf=None,
    ):
        self.num_variants = num_variants
        self.num_samples = num_samples
        self.uniform_ploidy = uniform_ploidy
        self.total_string_length = total_string_length
        nv, ns = num_variants, num_samples
        dot = np.full((nv, 1), ord("."), dtype=np.uint8)
        self.varid = dot.copy() if varid is None else varid
        self.rsid = dot.copy() if rsid is None else rsid
        self.chrom = dot.copy() if chrom is None else chrom
        self.allele1 = dot.copy() if allele1 is None else allele1
        self.allele2 = dot.copy() if allele2 is None else allele2
        self.position = np.zeros(nv, dtype=np.int32) if position is None else position
        self.genotypes = (
            np.zeros((nv, ns, 2), dtype=np.int8) if genotypes is None else genotypes
        )
        self.phased = np.zeros(nv, dtype=bool) if phased is None else phased
        if out_buf is None:
            geno_size = 10 + (uniform_ploidy + 1) * ns
            payload = 2 + 5 * max(1, (geno_size + 65534) // 65535) + geno_size + 4
            bpv = 28 + total_string_length + payload
            out_buf = np.zeros(nv * bpv, dtype=np.uint8)
        self.out_buf = out_buf

    def as_tuple(self):
        return (
            self.varid,
            self.rsid,
            self.chrom,
            self.allele1,
            self.allele2,
            self.position,
            self.genotypes,
            self.phased,
            self.out_buf,
            self.uniform_ploidy,
            self.total_string_length,
        )


def _call_chunk_slice(**kwargs):
    return _vcztools.encode_bgen_chunk_slice_level0(
        *_ChunkSliceArgs(**kwargs).as_tuple()
    )


class TestEncodeBgenChunkSliceLevel0:
    """Argument-parsing and error-path coverage for the
    ``encode_bgen_chunk_slice_level0`` C wrapper. Functional/byte-level
    behaviour is tested in ``lib/tests.c`` (C suite) and in
    ``tests/test_bgen.py::TestBgenChunkSliceLevel0Kernel`` (end-to-end
    via BgenEncoder)."""

    # --- arg-count / arg-type from PyArg_ParseTuple ---

    def test_bad_num_arguments(self):
        with pytest.raises(TypeError):
            _vcztools.encode_bgen_chunk_slice_level0()
        # 12 args (one too many — the kernel takes 11)
        a = _ChunkSliceArgs()
        with pytest.raises(TypeError):
            _vcztools.encode_bgen_chunk_slice_level0(*a.as_tuple(), 0)

    @pytest.mark.parametrize(
        "field",
        [
            "varid",
            "rsid",
            "chrom",
            "allele1",
            "allele2",
            "position",
            "genotypes",
            "phased",
            "out_buf",
        ],
    )
    @pytest.mark.parametrize("bad_type", [[], {}, "string", 4])
    def test_array_arg_bad_type(self, field, bad_type):
        # None is excluded: the args builder treats it as "use default",
        # so it can't be smuggled past the builder into the wrapper.
        with pytest.raises(TypeError):
            _call_chunk_slice(**{field: bad_type})

    @pytest.mark.parametrize("bad_ploidy_type", [None, "string", [], {}, 1.5])
    def test_uniform_ploidy_bad_type(self, bad_ploidy_type):
        a = _ChunkSliceArgs()
        with pytest.raises(TypeError):
            _vcztools.encode_bgen_chunk_slice_level0(
                a.varid,
                a.rsid,
                a.chrom,
                a.allele1,
                a.allele2,
                a.position,
                a.genotypes,
                a.phased,
                a.out_buf,
                bad_ploidy_type,
                a.total_string_length,
            )

    @pytest.mark.parametrize("bad_tsl_type", [None, "string", [], {}, 1.5])
    def test_total_string_length_bad_type(self, bad_tsl_type):
        a = _ChunkSliceArgs()
        with pytest.raises(TypeError):
            _vcztools.encode_bgen_chunk_slice_level0(
                a.varid,
                a.rsid,
                a.chrom,
                a.allele1,
                a.allele2,
                a.position,
                a.genotypes,
                a.phased,
                a.out_buf,
                a.uniform_ploidy,
                bad_tsl_type,
            )

    # --- check_array: wrong dimension for each array ---

    @pytest.mark.parametrize("field", ["varid", "rsid", "chrom", "allele1", "allele2"])
    def test_string_field_wrong_dim(self, field):
        with pytest.raises(ValueError, match=f"{field} has wrong dimension"):
            _call_chunk_slice(**{field: np.zeros(3, dtype=np.uint8)})

    def test_position_wrong_dim(self):
        with pytest.raises(ValueError, match="position has wrong dimension"):
            _call_chunk_slice(position=np.zeros((1, 1), dtype=np.int32))

    def test_genotypes_wrong_dim(self):
        with pytest.raises(ValueError, match="genotypes has wrong dimension"):
            _call_chunk_slice(genotypes=np.zeros((1, 2), dtype=np.int8))

    def test_phased_wrong_dim(self):
        with pytest.raises(ValueError, match="phased has wrong dimension"):
            _call_chunk_slice(phased=np.zeros((1, 1), dtype=bool))

    def test_out_buf_wrong_dim(self):
        with pytest.raises(ValueError, match="out_buf has wrong dimension"):
            _call_chunk_slice(out_buf=np.zeros((4, 4), dtype=np.uint8))

    # --- NPY_ARRAY_IN_ARRAY (non-contiguous views) ---

    def test_non_contiguous_genotypes(self):
        full = np.zeros((2, 4, 2), dtype=np.int8)
        view = full[:, ::2, :]
        with pytest.raises(ValueError, match="NPY_ARRAY_IN_ARRAY"):
            _call_chunk_slice(num_variants=2, num_samples=2, genotypes=view)

    def test_non_contiguous_out_buf(self):
        full = np.zeros(8192, dtype=np.uint8)
        with pytest.raises(ValueError, match="NPY_ARRAY_IN_ARRAY"):
            _call_chunk_slice(out_buf=full[::2])

    def test_read_only_out_buf(self):
        # NPY_ARRAY_IN_ARRAY doesn't include WRITEABLE; a read-only
        # buffer must be rejected so the kernel never writes to
        # read-only pages.
        out = np.zeros(8192, dtype=np.uint8)
        out.setflags(write=False)
        with pytest.raises(ValueError, match="out_buf must be writeable"):
            _call_chunk_slice(out_buf=out)

    # --- check_dtype: wrong dtype for each array ---

    @pytest.mark.parametrize("field", ["varid", "rsid", "chrom", "allele1", "allele2"])
    @pytest.mark.parametrize("bad_dtype", [np.int8, np.int32, np.float32, "S1"])
    def test_string_field_wrong_dtype(self, field, bad_dtype):
        a = _ChunkSliceArgs()
        arr_shape = getattr(a, field).shape
        with pytest.raises(ValueError, match=f"Wrong dtype for {field}"):
            _call_chunk_slice(**{field: np.zeros(arr_shape, dtype=bad_dtype)})

    @pytest.mark.parametrize("bad_dtype", [np.int8, np.int64, np.uint32, np.float32])
    def test_position_wrong_dtype(self, bad_dtype):
        with pytest.raises(ValueError, match="Wrong dtype for position"):
            _call_chunk_slice(position=np.zeros(1, dtype=bad_dtype))

    @pytest.mark.parametrize("bad_dtype", [np.uint8, np.int16, np.float32])
    def test_genotypes_wrong_dtype(self, bad_dtype):
        with pytest.raises(ValueError, match="Wrong dtype for genotypes"):
            _call_chunk_slice(genotypes=np.zeros((1, 1, 2), dtype=bad_dtype))

    @pytest.mark.parametrize("bad_dtype", [np.int8, np.uint8, np.float32])
    def test_phased_wrong_dtype(self, bad_dtype):
        with pytest.raises(ValueError, match="Wrong dtype for phased"):
            _call_chunk_slice(phased=np.zeros(1, dtype=bad_dtype))

    @pytest.mark.parametrize("bad_dtype", [np.int8, np.int32, np.float32])
    def test_out_buf_wrong_dtype(self, bad_dtype):
        with pytest.raises(ValueError, match="Wrong dtype for out_buf"):
            _call_chunk_slice(out_buf=np.zeros(8192, dtype=bad_dtype))

    # --- shape consistency between arrays ---

    @pytest.mark.parametrize("bad_ploidy_dim", [1, 3])
    def test_genotypes_ploidy_dim_not_two(self, bad_ploidy_dim):
        with pytest.raises(ValueError, match=r"\(V, S, 2\)"):
            _call_chunk_slice(genotypes=np.zeros((1, 1, bad_ploidy_dim), dtype=np.int8))

    @pytest.mark.parametrize(
        "field",
        ["rsid", "chrom", "allele1", "allele2", "position", "genotypes", "phased"],
    )
    def test_per_variant_axis_mismatch(self, field):
        # varid sets num_variants=2; bumping one other array to 3 must
        # surface the cross-axis check.
        a = _ChunkSliceArgs(num_variants=2)
        arr = getattr(a, field)
        if arr.ndim == 1:
            mutated = np.zeros(arr.shape[0] + 1, dtype=arr.dtype)
        else:
            mutated = np.zeros((arr.shape[0] + 1,) + arr.shape[1:], dtype=arr.dtype)
        with pytest.raises(
            ValueError, match=r"per-variant inputs must share num_variants axis"
        ):
            _vcztools.encode_bgen_chunk_slice_level0(
                a.varid,
                a.rsid if field != "rsid" else mutated,
                a.chrom if field != "chrom" else mutated,
                a.allele1 if field != "allele1" else mutated,
                a.allele2 if field != "allele2" else mutated,
                a.position if field != "position" else mutated,
                a.genotypes if field != "genotypes" else mutated,
                a.phased if field != "phased" else mutated,
                a.out_buf,
                a.uniform_ploidy,
                a.total_string_length,
            )

    def test_total_string_length_mismatch(self):
        # Per-variant actual sum is 5 (1 byte each); declaring 7 means
        # the kernel's invariant fails on the first variant.
        with pytest.raises(ValueError, match="string byte sum does not match"):
            _call_chunk_slice(total_string_length=7)

    # --- uniform_ploidy range ---

    @pytest.mark.parametrize("bad_ploidy", [0, 3, -1, 100])
    def test_uniform_ploidy_out_of_range(self, bad_ploidy):
        with pytest.raises(ValueError, match="uniform_ploidy must be 1 or 2"):
            _call_chunk_slice(uniform_ploidy=bad_ploidy)

    # --- buffer-too-small ---

    def test_out_buf_too_small(self):
        with pytest.raises(_vcztools.VczBufferTooSmall, match="out_buf is too small"):
            _call_chunk_slice(out_buf=np.zeros(8, dtype=np.uint8))

    # --- kernel error returns (handle_library_error → Python) ---

    def test_invalid_ploidy_error(self):
        # -2 in slot 0 → VCZ_ERR_BGEN_INVALID_PLOIDY → ValueError("zero-ploidy").
        G = np.array([[[-2, 0]]], dtype=np.int8)
        with pytest.raises(ValueError, match="zero-ploidy"):
            _call_chunk_slice(genotypes=G)

    def test_invalid_allele_error(self):
        # Allele 2 → VCZ_ERR_BGEN_INVALID_ALLELE → ValueError("out of range").
        G = np.array([[[0, 2]]], dtype=np.int8)
        with pytest.raises(ValueError, match="out of range"):
            _call_chunk_slice(genotypes=G)

    def test_mixed_ploidy_error(self):
        # uniform_ploidy=2 with one haploid sample (b == -2) →
        # VCZ_ERR_BGEN_MIXED_PLOIDY → NotImplementedError pointing at
        # write_bgen.
        G = np.array([[[0, 0], [0, -2]]], dtype=np.int8)
        with pytest.raises(NotImplementedError, match="write_bgen"):
            _call_chunk_slice(
                num_variants=1, num_samples=2, uniform_ploidy=2, genotypes=G
            )

    # --- success path ---

    def test_success_returns_none_and_writes_buffer(self):
        # Sentinel-fill the out_buf; after the call it must differ.
        a = _ChunkSliceArgs(num_variants=1, num_samples=2, uniform_ploidy=2)
        a.out_buf[:] = 0xAB
        result = _vcztools.encode_bgen_chunk_slice_level0(*a.as_tuple())
        assert result is None
        assert (a.out_buf != 0xAB).any()

    def test_zero_variants_no_write(self):
        # num_variants=0: kernel must not touch out_buf.
        a = _ChunkSliceArgs(num_variants=0, num_samples=2, uniform_ploidy=2)
        a.out_buf = np.full(64, 0xCC, dtype=np.uint8)
        result = _vcztools.encode_bgen_chunk_slice_level0(*a.as_tuple())
        assert result is None
        assert (a.out_buf == 0xCC).all()

    def test_uniform_haploid_succeeds(self):
        # uniform_ploidy=1, every sample has -2 in slot 1.
        G = np.array([[[0, -2], [1, -2]]], dtype=np.int8)
        a = _ChunkSliceArgs(
            num_variants=1, num_samples=2, uniform_ploidy=1, genotypes=G
        )
        assert _vcztools.encode_bgen_chunk_slice_level0(*a.as_tuple()) is None


_BVBS_PARAM_NAMES = (
    "num_samples",
    "uniform_ploidy",
    "total_string_length",
)
_BVBS_VALID_KWARGS = dict(
    num_samples=4,
    uniform_ploidy=2,
    total_string_length=5,
)
_BVBS_VALID_POSITIONAL = tuple(_BVBS_VALID_KWARGS[name] for name in _BVBS_PARAM_NAMES)


class TestBgenVariantBlockSize:
    """Argument-parsing and error-path coverage for the
    ``bgen_variant_block_size`` C wrapper. Functional/byte-level behaviour
    is exercised via ``BgenEncoder`` end-to-end in ``tests/test_bgen.py``."""

    # --- happy path / positional-vs-keyword equivalence ---

    def test_positional_call_returns_size(self):
        result = _vcztools.bgen_variant_block_size(*_BVBS_VALID_POSITIONAL)
        assert isinstance(result, int)
        assert result > 0

    def test_keyword_call_returns_size(self):
        shuffled = dict(
            total_string_length=5,
            uniform_ploidy=2,
            num_samples=4,
        )
        result = _vcztools.bgen_variant_block_size(**shuffled)
        expected = _vcztools.bgen_variant_block_size(*_BVBS_VALID_POSITIONAL)
        assert result == expected

    def test_mixed_positional_keyword(self):
        result = _vcztools.bgen_variant_block_size(
            4,
            2,
            total_string_length=5,
        )
        expected = _vcztools.bgen_variant_block_size(*_BVBS_VALID_POSITIONAL)
        assert result == expected

    # --- arg-count errors (PyArg_ParseTupleAndKeywords) ---

    def test_no_args_raises_type_error(self):
        with pytest.raises(TypeError):
            _vcztools.bgen_variant_block_size()

    def test_too_few_positional_raises_type_error(self):
        with pytest.raises(TypeError):
            _vcztools.bgen_variant_block_size(*_BVBS_VALID_POSITIONAL[:2])

    def test_too_many_positional_raises_type_error(self):
        with pytest.raises(TypeError):
            _vcztools.bgen_variant_block_size(*_BVBS_VALID_POSITIONAL, 1)

    @pytest.mark.parametrize("missing", _BVBS_PARAM_NAMES)
    def test_missing_required_kwarg(self, missing):
        kwargs = {k: v for k, v in _BVBS_VALID_KWARGS.items() if k != missing}
        with pytest.raises(TypeError):
            _vcztools.bgen_variant_block_size(**kwargs)

    # --- unknown keyword ---

    def test_extra_kwarg_raises_type_error(self):
        with pytest.raises(TypeError, match="takes at most 3 keyword arguments"):
            _vcztools.bgen_variant_block_size(**_BVBS_VALID_KWARGS, extra=1)

    # --- bad arg type (PyArg `n` format expects integer-like) ---

    @pytest.mark.parametrize("name", _BVBS_PARAM_NAMES)
    @pytest.mark.parametrize("bad_value", [None, "string", [], {}, 1.5])
    def test_bad_arg_type(self, name, bad_value):
        kwargs = dict(_BVBS_VALID_KWARGS)
        kwargs[name] = bad_value
        with pytest.raises(TypeError):
            _vcztools.bgen_variant_block_size(**kwargs)

    # --- validation errors raised inside the C wrapper ---

    @pytest.mark.parametrize("bad_ploidy", [0, 3, -1, 100])
    def test_invalid_uniform_ploidy(self, bad_ploidy):
        kwargs = dict(_BVBS_VALID_KWARGS, uniform_ploidy=bad_ploidy)
        with pytest.raises(ValueError, match="uniform_ploidy must be 1 or 2"):
            _vcztools.bgen_variant_block_size(**kwargs)

    @pytest.mark.parametrize("name", ["num_samples", "total_string_length"])
    def test_negative_size_arg(self, name):
        kwargs = dict(_BVBS_VALID_KWARGS)
        kwargs[name] = -1
        with pytest.raises(ValueError, match="size arguments must be non-negative"):
            _vcztools.bgen_variant_block_size(**kwargs)

    def test_zero_sizes_allowed(self):
        result = _vcztools.bgen_variant_block_size(
            num_samples=0,
            uniform_ploidy=2,
            total_string_length=0,
        )
        assert isinstance(result, int)
        assert result > 0
