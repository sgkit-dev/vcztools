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


class TestComputeAcAn:
    def _ac(self, num_variants, max_num_alt):
        return np.zeros((num_variants, max_num_alt), dtype=np.int32)

    def _an(self, num_variants):
        return np.zeros(num_variants, dtype=np.int32)

    def _na(self, num_variants, value=2):
        return np.full(num_variants, value, dtype=np.int32)

    def test_bad_num_arguments(self):
        with pytest.raises(TypeError):
            _vcztools.compute_ac_an()
        with pytest.raises(TypeError):
            _vcztools.compute_ac_an(np.zeros((1, 1, 2), dtype=np.int8))
        with pytest.raises(TypeError):
            _vcztools.compute_ac_an(np.zeros((1, 1, 2), dtype=np.int8), self._na(1))
        with pytest.raises(TypeError):
            _vcztools.compute_ac_an(
                np.zeros((1, 1, 2), dtype=np.int8), self._na(1), self._ac(1, 1)
            )

    @pytest.mark.parametrize("bad_type", [[], {}, "string", 4])
    def test_genotypes_bad_type(self, bad_type):
        with pytest.raises(TypeError):
            _vcztools.compute_ac_an(bad_type, self._na(1), self._ac(1, 1), self._an(1))

    def test_genotypes_wrong_dim(self):
        with pytest.raises(ValueError, match="genotypes has wrong dimension"):
            _vcztools.compute_ac_an(
                np.zeros((1, 1), dtype=np.int8),
                self._na(1),
                self._ac(1, 1),
                self._an(1),
            )

    @pytest.mark.parametrize("bad_dtype", [np.int16, np.int32, np.int64, np.float64])
    def test_genotypes_bad_dtype(self, bad_dtype):
        with pytest.raises(ValueError, match="Wrong dtype for genotypes"):
            _vcztools.compute_ac_an(
                np.zeros((1, 1, 2), dtype=bad_dtype),
                self._na(1),
                self._ac(1, 1),
                self._an(1),
            )

    def test_num_alleles_wrong_dim(self):
        with pytest.raises(ValueError, match="num_alleles has wrong dimension"):
            _vcztools.compute_ac_an(
                np.zeros((1, 1, 2), dtype=np.int8),
                np.zeros((1, 1), dtype=np.int32),
                self._ac(1, 1),
                self._an(1),
            )

    @pytest.mark.parametrize("bad_dtype", [np.int8, np.int16, np.int64, np.float32])
    def test_num_alleles_bad_dtype(self, bad_dtype):
        with pytest.raises(ValueError, match="Wrong dtype for num_alleles"):
            _vcztools.compute_ac_an(
                np.zeros((1, 1, 2), dtype=np.int8),
                np.full(1, 2, dtype=bad_dtype),
                self._ac(1, 1),
                self._an(1),
            )

    def test_num_alleles_wrong_num_variants(self):
        with pytest.raises(ValueError, match="num_alleles.shape\\[0\\] must equal"):
            _vcztools.compute_ac_an(
                np.zeros((2, 1, 2), dtype=np.int8),
                self._na(1),
                self._ac(2, 1),
                self._an(2),
            )

    def test_ac_out_wrong_dim(self):
        with pytest.raises(ValueError, match="ac_out has wrong dimension"):
            _vcztools.compute_ac_an(
                np.zeros((1, 1, 2), dtype=np.int8),
                self._na(1),
                np.zeros(1, dtype=np.int32),
                self._an(1),
            )

    @pytest.mark.parametrize("bad_dtype", [np.int8, np.int16, np.int64, np.float32])
    def test_ac_out_bad_dtype(self, bad_dtype):
        with pytest.raises(ValueError, match="Wrong dtype for ac_out"):
            _vcztools.compute_ac_an(
                np.zeros((1, 1, 2), dtype=np.int8),
                self._na(1),
                np.zeros((1, 1), dtype=bad_dtype),
                self._an(1),
            )

    def test_ac_out_read_only(self):
        ac = self._ac(1, 1)
        ac.setflags(write=False)
        with pytest.raises(ValueError, match="ac_out must be writeable"):
            _vcztools.compute_ac_an(
                np.zeros((1, 1, 2), dtype=np.int8), self._na(1), ac, self._an(1)
            )

    def test_an_out_wrong_dim(self):
        with pytest.raises(ValueError, match="an_out has wrong dimension"):
            _vcztools.compute_ac_an(
                np.zeros((1, 1, 2), dtype=np.int8),
                self._na(1),
                self._ac(1, 1),
                np.zeros((1, 1), dtype=np.int32),
            )

    @pytest.mark.parametrize("bad_dtype", [np.int8, np.int16, np.int64, np.float32])
    def test_an_out_bad_dtype(self, bad_dtype):
        with pytest.raises(ValueError, match="Wrong dtype for an_out"):
            _vcztools.compute_ac_an(
                np.zeros((1, 1, 2), dtype=np.int8),
                self._na(1),
                self._ac(1, 1),
                np.zeros(1, dtype=bad_dtype),
            )

    def test_an_out_read_only(self):
        an = self._an(1)
        an.setflags(write=False)
        with pytest.raises(ValueError, match="an_out must be writeable"):
            _vcztools.compute_ac_an(
                np.zeros((1, 1, 2), dtype=np.int8),
                self._na(1),
                self._ac(1, 1),
                an,
            )

    def test_ac_out_wrong_num_variants(self):
        with pytest.raises(ValueError, match="ac_out.shape\\[0\\] must equal"):
            _vcztools.compute_ac_an(
                np.zeros((2, 1, 2), dtype=np.int8),
                self._na(2),
                self._ac(1, 1),
                self._an(2),
            )

    def test_an_out_wrong_num_variants(self):
        with pytest.raises(ValueError, match="an_out.shape\\[0\\] must equal"):
            _vcztools.compute_ac_an(
                np.zeros((2, 1, 2), dtype=np.int8),
                self._na(2),
                self._ac(2, 1),
                self._an(1),
            )

    def test_non_contiguous_genotypes(self):
        full = np.zeros((2, 4, 2), dtype=np.int8)
        view = full[:, ::2, :]
        with pytest.raises(ValueError, match="NPY_ARRAY_IN_ARRAY"):
            _vcztools.compute_ac_an(view, self._na(2), self._ac(2, 1), self._an(2))

    def test_non_contiguous_ac_out(self):
        full = np.zeros((4, 2), dtype=np.int32)
        view = full[::2]
        with pytest.raises(ValueError, match="NPY_ARRAY_IN_ARRAY"):
            _vcztools.compute_ac_an(
                np.zeros((2, 1, 2), dtype=np.int8), self._na(2), view, self._an(2)
            )

    def test_returns_none(self):
        G = np.zeros((1, 1, 2), dtype=np.int8)
        assert (
            _vcztools.compute_ac_an(G, self._na(1), self._ac(1, 1), self._an(1)) is None
        )

    def test_example(self):
        # Mirrors the C-level test_compute_ac_an_ploidy2 CUnit test.
        G = np.array(
            [
                [[0, 0], [0, 1], [1, 1]],
                [[0, 0], [0, 2], [1, 2]],
                [[0, 1], [1, 2], [3, -2]],
                [[-1, -1], [-1, -2], [1, -2]],
                [[0, 2], [-1, -2], [1, -2]],
            ],
            dtype=np.int8,
        )
        num_alleles = np.array([2, 3, 4, 2, 3], dtype=np.int32)
        ac = self._ac(5, 3)
        an = self._an(5)
        FILL = -2
        _vcztools.compute_ac_an(G, num_alleles, ac, an)
        np.testing.assert_array_equal(
            ac,
            [
                [3, FILL, FILL],
                [1, 2, FILL],
                [2, 1, 1],
                [1, FILL, FILL],
                [1, 1, FILL],
            ],
        )
        np.testing.assert_array_equal(an, [6, 6, 5, 1, 3])

    def test_an_only(self):
        # max_num_alt == 0 -> AN populated, AC has zero-width second dim.
        G = np.array(
            [
                [[0, 0], [0, 1]],
                [[-1, -1], [0, -2]],
            ],
            dtype=np.int8,
        )
        num_alleles = np.array([2, 2], dtype=np.int32)
        ac = self._ac(2, 0)
        an = self._an(2)
        _vcztools.compute_ac_an(G, num_alleles, ac, an)
        np.testing.assert_array_equal(an, [4, 1])

    def test_zero_variants(self):
        G = np.zeros((0, 3, 2), dtype=np.int8)
        num_alleles = np.zeros(0, dtype=np.int32)
        ac = self._ac(0, 2)
        an = self._an(0)
        _vcztools.compute_ac_an(G, num_alleles, ac, an)
        assert ac.shape == (0, 2)
        assert an.shape == (0,)

    @pytest.mark.parametrize(
        ("bad_value", "num_alleles_value"),
        [
            (-3, 2),  # below the missing/fill range
            (2, 2),  # one past the upper bound
            (5, 2),  # well above the upper bound
        ],
    )
    def test_invalid_genotype_raises(self, bad_value, num_alleles_value):
        G = np.array([[[0, bad_value]]], dtype=np.int8)
        num_alleles = np.array([num_alleles_value], dtype=np.int32)
        with pytest.raises(ValueError, match="outside accepted range"):
            _vcztools.compute_ac_an(G, num_alleles, self._ac(1, 1), self._an(1))

    @pytest.mark.parametrize("num_alleles_value", [0, -1])
    def test_num_alleles_below_one_raises(self, num_alleles_value):
        G = np.array([[[0, 0]]], dtype=np.int8)
        num_alleles = np.array([num_alleles_value], dtype=np.int32)
        with pytest.raises(ValueError, match="num_alleles values must be >= 1"):
            _vcztools.compute_ac_an(G, num_alleles, self._ac(1, 0), self._an(1))
