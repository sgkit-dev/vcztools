import pytest
import numpy as np

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


def example_encoder(num_variants=1, num_samples=0):
    encoder = _vcztools.VcfEncoder(
        num_variants, num_samples, **example_fixed_data(num_variants, num_samples)
    )

    return encoder


class TestPrintState:
    def test_nomimal_case(self, tmp_path):
        encoder = example_encoder()
        filename = tmp_path / "debug.txt"
        with open(filename, "w") as f:
            encoder.print_state(f)
        with open(filename, "r") as f:
            s = f.read()
        assert "CHROM" in s

    def test_read_only_file(self, tmp_path):
        encoder = example_encoder()
        filename = tmp_path / "debug.txt"
        with open(filename, "w") as f:
            f.write("x")
        with open(filename, "r") as f:
            with pytest.raises(OSError, match="22"):
                encoder.print_state(f)

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


class TestEncodeOverrun:
    def test_fixed_fields(self):
        encoder = example_encoder()
        s = encoder.encode(0, 1024)
        for length in range(len(s)):
            with pytest.raises(ValueError, match="-101"):
                encoder.encode(0, length)


class TestTypeChecking:
    @pytest.mark.parametrize("name", FIXED_FIELD_NAMES)
    def test_field_incorrect_length(self, name):
        num_variants = 5
        data = example_fixed_data(num_variants)
        data[name] = data[name][1:]
        with pytest.raises(ValueError, match=f"Array {name.upper()} must have "):
            _vcztools.VcfEncoder(num_variants, 0, **data)

    @pytest.mark.parametrize("name", FIXED_FIELD_NAMES)
    def test_field_incorrect_dtype(self, name):
        num_variants = 5
        data = example_fixed_data(num_variants)
        data[name] = np.zeros(data[name].shape, dtype=np.int64)
        with pytest.raises(ValueError, match=f"Wrong dtype for {name.upper()}"):
            _vcztools.VcfEncoder(num_variants, 0, **data)

    @pytest.mark.parametrize("name", FIXED_FIELD_NAMES)
    def test_field_incorrect_type(self, name):
        num_variants = 5
        data = example_fixed_data(num_variants)
        data[name] = "A Python string"
        with pytest.raises(TypeError, match=f"must be numpy.ndarray"):
            _vcztools.VcfEncoder(num_variants, 0, **data)


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

    def test_add_format_field_bad_array_num_samples(self):
        encoder = example_encoder(5, 2)
        with pytest.raises(ValueError, match="number of samples"):
            encoder.add_format_field("name", np.zeros((5, 1, 1), dtype=bool))

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
