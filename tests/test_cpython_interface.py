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
        for dtype in ["i1", "i2", "i4", "f4", "?", "S1"]:
            name = f"I{dtype}_{num_columns}"
            data = np.arange(num_variants * num_columns).astype(dtype)
            d[name] = data.reshape((num_variants, num_columns))
    return d


def example_format_data(num_variants, num_samples):
    d = {}
    for num_columns in range(1, 3):
        for dtype in ["i1", "i2", "i4", "f4", "?", "S1"]:
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
        ["add_info_field", "add_gt_field", "add_format_field", "print_state", "encode"],
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
