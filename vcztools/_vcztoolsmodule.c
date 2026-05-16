
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <stddef.h> /* for offsetof() */
#include <stdbool.h>
#include <numpy/arrayobject.h>

#include "vcf_encoder.h"

typedef struct {
    PyObject_HEAD
    vcz_variant_encoder_t *vcf_encoder;
    PyObject *arrays;
} VcfEncoder;

static PyObject *VczBufferTooSmall;

static void
handle_library_error(int err)
{
    switch (err) {
        case VCZ_ERR_BUFFER_OVERFLOW:
            PyErr_Format(
                VczBufferTooSmall, "Error: %d; specified buffer size is too small", err);
            break;
        case VCZ_ERR_BGEN_INVALID_PLOIDY:
            PyErr_Format(PyExc_ValueError,
                "BGEN encoder: -2 in genotype slot 0 (zero-ploidy / unused sample) "
                "is not representable in BGEN");
            break;
        case VCZ_ERR_BGEN_INVALID_ALLELE:
            PyErr_Format(PyExc_ValueError,
                "BGEN encoder: genotype value out of range; biallelic input "
                "expects values in {-2, -1, 0, 1}");
            break;
        case VCZ_ERR_BGEN_MIXED_PLOIDY:
            PyErr_Format(PyExc_NotImplementedError,
                "BgenEncoder requires uniform ploidy across all samples and "
                "variants; this chunk contains mixed ploidy. Use write_bgen() "
                "instead.");
            break;
        // TODO handle the other error types.
        default:
            PyErr_Format(PyExc_ValueError, "Error occured: %d: ", err);
    }
}

/* The dup/fdopen/fclose pattern used by make_file does not work on Windows
 * because _dup shares the underlying OS handle rather than duplicating it.
 * When fclose closes the duped FILE*, it invalidates the original handle,
 * causing an access violation when Python later closes its file object.
 * Since print_state is only a debugging tool, we simply exclude it on Windows. */
#ifndef _WIN32

static FILE *
make_file(PyObject *fileobj, const char *mode)
{
    FILE *ret = NULL;
    FILE *file = NULL;
    int fileobj_fd, new_fd;

    fileobj_fd = PyObject_AsFileDescriptor(fileobj);
    if (fileobj_fd == -1) {
        goto out;
    }
    new_fd = dup(fileobj_fd);
    if (new_fd == -1) {
        PyErr_SetFromErrno(PyExc_OSError);
        goto out;
    }
    file = fdopen(new_fd, mode);
    if (file == NULL) {
        (void) close(new_fd);
        PyErr_SetFromErrno(PyExc_OSError);
        goto out;
    }
    ret = file;
out:
    return ret;
}

#endif /* !_WIN32 */

/*===================================================================
 * VcfEncoder
 *===================================================================
 */

static int
VcfEncoder_check_state(VcfEncoder *self)
{
    int ret = -1;
    if (self->vcf_encoder == NULL) {
        PyErr_SetString(PyExc_SystemError, "VcfEncoder not initialised");
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static void
VcfEncoder_dealloc(VcfEncoder *self)
{
    if (self->vcf_encoder != NULL) {
        vcz_variant_encoder_free(self->vcf_encoder);
        PyMem_Free(self->vcf_encoder);
        self->vcf_encoder = NULL;
    }
    Py_XDECREF(self->arrays);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static int
VcfEncoder_add_array(
    VcfEncoder *self, const char *prefix, const char *name, PyArrayObject *array)
{
    int ret = -1;
    PyObject *key = PyUnicode_FromFormat("%s%s", prefix, name);

    if (array == NULL || key == NULL) {
        goto out; // GCOVR_EXCL_LINE
    }
    ret = PyDict_SetItem(self->arrays, key, (PyObject *) array);
out:
    Py_XDECREF(key);
    return ret;
}

static int
check_array(const char *name, PyArrayObject *array, npy_intp dimension)
{
    int ret = -1;

    assert(PyArray_CheckExact(array));
    if (!PyArray_CHKFLAGS(array, NPY_ARRAY_IN_ARRAY)) {
        PyErr_Format(
            PyExc_ValueError, "Array %s must have NPY_ARRAY_IN_ARRAY flags.", name);
        goto out;
    }
    if (PyArray_NDIM(array) != dimension) {
        PyErr_Format(PyExc_ValueError, "Array %s has wrong dimension: %d != %d", name,
            (int) PyArray_NDIM(array), (int) dimension);
        goto out;
    }
    ret = 0;
out:
    return ret;
}

/* Output-buffer-only check. NPY_ARRAY_IN_ARRAY (used by check_array) is
 * C_CONTIGUOUS | ALIGNED — it does not include WRITEABLE. A read-only
 * out_buf would silently make the kernel write into read-only pages, so
 * wrappers that hand the kernel a caller-allocated output array must
 * call this on top of check_array. */
static int
check_array_writeable(const char *name, PyArrayObject *array)
{
    int ret = -1;
    if (!PyArray_CHKFLAGS(array, NPY_ARRAY_WRITEABLE)) {
        PyErr_Format(PyExc_ValueError, "Array %s must be writeable.", name);
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static int
VcfEncoder_add_field_array(VcfEncoder *self, const char *name, PyArrayObject *array,
    npy_intp dimension, const char *prefix, bool is_format_field)
{
    int ret = -1;
    npy_intp *shape;

    if (check_array(name, array, dimension) != 0) {
        goto out;
    }
    shape = PyArray_DIMS(array);
    if (shape[0] != (npy_intp) self->vcf_encoder->num_variants) {
        PyErr_Format(PyExc_ValueError,
            "Array %s must have first dimension equal to number of variants", name);
        goto out;
    }
    if (is_format_field) {
        if (shape[1] != (npy_intp) self->vcf_encoder->num_samples) {
            PyErr_Format(PyExc_ValueError,
                "Array %s must have second dimension equal to number of samples", name);
            goto out;
        }
    }
    ret = VcfEncoder_add_array(self, prefix, name, array);
out:
    return ret;
}

static int
np_type_to_vcz_type(const char *name, PyArrayObject *array)
{
    int ret = -1;

    switch (PyArray_DTYPE(array)->kind) {
        case 'i':
            ret = VCZ_TYPE_INT;
            break;
        case 'f':
            ret = VCZ_TYPE_FLOAT;
            break;
        case 'S':
            ret = VCZ_TYPE_STRING;
            break;
        case 'b':
            ret = VCZ_TYPE_BOOL;
            break;
        default:
            PyErr_Format(
                PyExc_ValueError, "Array '%s' has unsupported array dtype", name);
            goto out;
    }
out:
    return ret;
}

static int
check_dtype(const char *name, PyArrayObject *array, int type)
{
    if (PyArray_DTYPE(array)->type_num != type) {
        PyErr_Format(PyExc_ValueError, "Wrong dtype for %s", name);
        return -1;
    }
    return 0;
}

static int
VcfEncoder_store_fixed_array(
    VcfEncoder *self, PyArrayObject *array, const char *name, int type, int dimension)
{
    int ret = -1;

    if (VcfEncoder_add_field_array(self, name, array, dimension, "", false) != 0) {
        goto out;
    }
    if (check_dtype(name, array, type) != 0) {
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static int
VcfEncoder_init(VcfEncoder *self, PyObject *args, PyObject *kwds)
{
    int ret = -1;
    static char *kwlist[] = { "num_variants", "num_samples", "chrom", "pos", "id", "ref",
        "alt", "qual", "filter_ids", "filter", NULL };
    int num_variants, num_samples;
    PyArrayObject *chrom = NULL;
    PyArrayObject *pos = NULL;
    PyArrayObject *id = NULL;
    PyArrayObject *ref = NULL;
    PyArrayObject *alt = NULL;
    PyArrayObject *qual = NULL;
    PyArrayObject *filter_ids = NULL;
    PyArrayObject *filter = NULL;
    npy_intp num_filters;
    int err;

    self->vcf_encoder = NULL;
    self->arrays = NULL;
    self->vcf_encoder = PyMem_Calloc(1, sizeof(*self->vcf_encoder));
    self->arrays = PyDict_New();
    if (self->vcf_encoder == NULL || self->arrays == NULL) {
        PyErr_NoMemory(); // GCOVR_EXCL_LINE
        goto out;         // GCOVR_EXCL_LINE
    }

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiO!O!O!O!O!O!O!O!", kwlist,
            &num_variants, &num_samples, &PyArray_Type, &chrom, &PyArray_Type, &pos,
            &PyArray_Type, &id, &PyArray_Type, &ref, &PyArray_Type, &alt, &PyArray_Type,
            &qual, &PyArray_Type, &filter_ids, &PyArray_Type, &filter)) {
        goto out;
    }

    // This function currently cannot fail as there's no memory allocation, but
    // we keep the check in case this changes in later versions.
    err = vcz_variant_encoder_init(self->vcf_encoder, num_variants, num_samples);
    if (err < 0) {
        handle_library_error(err); // GCOVR_EXCL_LINE
        goto out;                  // GCOVR_EXCL_LINE
    }

    if (VcfEncoder_store_fixed_array(self, chrom, "chrom", NPY_STRING, 1) != 0) {
        goto out;
    }
    if (VcfEncoder_store_fixed_array(self, pos, "pos", NPY_INT32, 1) != 0) {
        goto out;
    }
    if (VcfEncoder_store_fixed_array(self, id, "id", NPY_STRING, 2) != 0) {
        goto out;
    }
    if (VcfEncoder_store_fixed_array(self, ref, "ref", NPY_STRING, 1) != 0) {
        goto out;
    }
    if (VcfEncoder_store_fixed_array(self, alt, "alt", NPY_STRING, 2) != 0) {
        goto out;
    }
    if (VcfEncoder_store_fixed_array(self, qual, "qual", NPY_FLOAT32, 1) != 0) {
        goto out;
    }

    /* These calls cannot fail, so we don't check the output. */
    vcz_variant_encoder_add_pos_field(self->vcf_encoder, PyArray_DATA(pos));
    vcz_variant_encoder_add_qual_field(self->vcf_encoder, PyArray_DATA(qual));

    /* In practise, these calls can't fail either because ITEMSIZE is always > 0.
     * We keep the checks in case this changes for later versions of numpy.
     */
    vcz_variant_encoder_add_chrom_field(
        self->vcf_encoder, PyArray_ITEMSIZE(chrom), PyArray_DATA(chrom));
    if (err < 0) {
        handle_library_error(err); // GCOVR_EXCL_LINE
        goto out;                  // GCOVR_EXCL_LINE
    }
    vcz_variant_encoder_add_ref_field(
        self->vcf_encoder, PyArray_ITEMSIZE(ref), PyArray_DATA(ref));
    if (err < 0) {
        handle_library_error(err); // GCOVR_EXCL_LINE
        goto out;                  // GCOVR_EXCL_LINE
    }
    /* Note: the only provokable error here is the zero-sized second dimension,
     * which we could catch easily above */
    err = vcz_variant_encoder_add_id_field(
        self->vcf_encoder, PyArray_ITEMSIZE(id), PyArray_DIMS(id)[1], PyArray_DATA(id));
    if (err < 0) {
        handle_library_error(err);
        goto out;
    }
    err = vcz_variant_encoder_add_alt_field(self->vcf_encoder, PyArray_ITEMSIZE(alt),
        PyArray_DIMS(alt)[1], PyArray_DATA(alt));
    if (err < 0) {
        handle_library_error(err);
        goto out;
    }

    /* NOTE: we could generalise this pattern for CHROM also to save a bit of time
     * in building numpy String arrays */
    if (check_array("filter_ids", filter_ids, 1) != 0) {
        goto out;
    }
    if (check_dtype("filter_ids", filter_ids, NPY_STRING) != 0) {
        goto out;
    }
    num_filters = PyArray_DIMS(filter_ids)[0];
    if (VcfEncoder_add_array(self, "", "filter_ids", filter_ids) != 0) {
        goto out; // GCOVR_EXCL_LINE
    }
    if (VcfEncoder_store_fixed_array(self, filter, "filter", NPY_BOOL, 2) != 0) {
        goto out;
    }
    if (PyArray_DIMS(filter)[1] != num_filters) {
        PyErr_Format(
            PyExc_ValueError, "filters dimension must be (num_variants, num_filters)");
        goto out;
    }
    err = vcz_variant_encoder_add_filter_field(self->vcf_encoder,
        PyArray_ITEMSIZE(filter_ids), num_filters, PyArray_DATA(filter_ids),
        PyArray_DATA(filter));
    if (err < 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    return ret;
}
static PyObject *
VcfEncoder_add_info_field(VcfEncoder *self, PyObject *args)
{
    PyObject *ret = NULL;
    PyArrayObject *array = NULL;
    const char *name;
    int err, type;

    if (VcfEncoder_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "sO!", &name, &PyArray_Type, &array)) {
        goto out;
    }
    if (VcfEncoder_add_field_array(self, name, array, 2, "INFO/", false) != 0) {
        goto out;
    }
    type = np_type_to_vcz_type(name, array);
    if (type < 0) {
        goto out;
    }
    err = vcz_variant_encoder_add_info_field(self->vcf_encoder, name, type,
        PyArray_ITEMSIZE(array), PyArray_DIMS(array)[1], PyArray_DATA(array));
    if (err < 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
VcfEncoder_add_format_field(VcfEncoder *self, PyObject *args)
{
    PyObject *ret = NULL;
    PyArrayObject *array = NULL;
    const char *name;
    int err, type;

    if (VcfEncoder_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "sO!", &name, &PyArray_Type, &array)) {
        goto out;
    }
    if (VcfEncoder_add_field_array(self, name, array, 3, "FORMAT/", true) != 0) {
        goto out;
    }
    type = np_type_to_vcz_type(name, array);
    if (type < 0) {
        goto out;
    }
    err = vcz_variant_encoder_add_format_field(self->vcf_encoder, name, type,
        PyArray_ITEMSIZE(array), PyArray_DIMS(array)[2], PyArray_DATA(array));
    if (err < 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
VcfEncoder_add_gt_field(VcfEncoder *self, PyObject *args)
{
    PyArrayObject *gt = NULL;
    PyArrayObject *gt_phased = NULL;
    PyObject *ret = NULL;
    int err;

    if (VcfEncoder_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &gt, &PyArray_Type, &gt_phased)) {
        goto out;
    }
    if (VcfEncoder_add_field_array(self, "gt", gt, 3, "FORMAT/", true) != 0) {
        goto out;
    }
    if (VcfEncoder_add_field_array(self, "gt_phased", gt_phased, 2, "FORMAT/", true)
        != 0) {
        goto out;
    }
    if (PyArray_DTYPE(gt)->kind != 'i') {
        PyErr_Format(PyExc_ValueError, "Array 'gt' has unsupported array dtype");
        goto out;
    }
    if (check_dtype("gt_phased", gt_phased, NPY_BOOL) != 0) {
        goto out;
    }
    err = vcz_variant_encoder_add_gt_field(self->vcf_encoder, PyArray_ITEMSIZE(gt),
        PyArray_DIMS(gt)[2], PyArray_DATA(gt), PyArray_DATA(gt_phased));
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }

    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
VcfEncoder_encode(VcfEncoder *self, PyObject *args)
{
    PyObject *ret = NULL;
    unsigned long long row;
    unsigned long long bufsize;
    char *buf = NULL;
    int64_t line_length;

    if (VcfEncoder_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "KK", &row, &bufsize)) {
        goto out;
    }
    /* Interpret bufsize as the length of the Python string, so add one to
     * allow for the NULL byte */
    bufsize++;
    buf = PyMem_Malloc(bufsize);
    if (buf == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    line_length = vcz_variant_encoder_encode(
        self->vcf_encoder, (size_t) row, buf, (size_t) bufsize);
    Py_END_ALLOW_THREADS
    if (line_length < 0) {
        handle_library_error((int) line_length);
        goto out;
    }
    ret = Py_BuildValue("s#", buf, (Py_ssize_t) line_length);
out:
    PyMem_Free(buf);
    return ret;
}

#ifndef _WIN32
static PyObject *
VcfEncoder_print_state(VcfEncoder *self, PyObject *args)
{
    PyObject *ret = NULL;
    PyObject *fileobj;
    FILE *file = NULL;

    if (VcfEncoder_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "O", &fileobj)) {
        goto out;
    }
    file = make_file(fileobj, "w");
    if (file == NULL) {
        goto out;
    }
    vcz_variant_encoder_print_state(self->vcf_encoder, file);
    ret = Py_BuildValue("");
out:
    if (file != NULL) {
        (void) fclose(file);
    }
    return ret;
}

#endif /* !_WIN32 */

/* Return a copy of the dictionary of arrays providing the memory backing.
 * Note that we return copy of the Dictionary here so that the arrays themselves
 * can't be removed from it.
 */
static PyObject *
VcfEncoder_getarrays(VcfEncoder *self, void *closure)
{
    PyObject *ret = NULL;

    if (VcfEncoder_check_state(self) != 0) {
        goto out;
    }
    ret = PyDict_Copy(self->arrays);
out:
    return ret;
}

static PyGetSetDef VcfEncoder_getsetters[] = {
    { "arrays", (getter) VcfEncoder_getarrays, NULL, "Arrays used for memory backing",
        NULL },
    { NULL } /* Sentinel */
};

static PyMethodDef VcfEncoder_methods[] = {
#ifndef _WIN32
    { .ml_name = "print_state",
        .ml_meth = (PyCFunction) VcfEncoder_print_state,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Debug method to print out the low-level state" },
#endif
    { .ml_name = "add_info_field",
        .ml_meth = (PyCFunction) VcfEncoder_add_info_field,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Add an INFO field to the encoder" },
    { .ml_name = "add_gt_field",
        .ml_meth = (PyCFunction) VcfEncoder_add_gt_field,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Add the GT field" },
    { .ml_name = "add_format_field",
        .ml_meth = (PyCFunction) VcfEncoder_add_format_field,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Add a format field to the encoder" },
    { .ml_name = "encode",
        .ml_meth = (PyCFunction) VcfEncoder_encode,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Return the specified row of VCF text" },
    { NULL } /* Sentinel */
};

// clang-format off
static PyTypeObject VcfEncoderType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_vcztools.VcfEncoder",
    .tp_basicsize = sizeof(VcfEncoder),
    .tp_dealloc = (destructor) VcfEncoder_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "VcfEncoder objects",
    .tp_methods = VcfEncoder_methods,
    .tp_init = (initproc) VcfEncoder_init,
    .tp_new = PyType_GenericNew,
    .tp_getset = VcfEncoder_getsetters,
};
// clang-format on

/*===================================================================
 * Module level code.
 *===================================================================
 */

static PyObject *
vcztools_encode_plink(PyObject *self, PyObject *args)
{
    PyObject *ret = NULL;
    PyArrayObject *genotypes = NULL;
    PyArrayObject *out_buf = NULL;
    npy_intp num_variants, num_samples, expected;

    if (!PyArg_ParseTuple(
            args, "O!O!", &PyArray_Type, &genotypes, &PyArray_Type, &out_buf)) {
        goto out;
    }
    if (check_array("genotypes", genotypes, 3) != 0) {
        goto out;
    }
    if (check_dtype("genotypes", genotypes, NPY_INT8) != 0) {
        goto out;
    }
    if (PyArray_DIMS(genotypes)[2] != 2) {
        PyErr_Format(PyExc_ValueError, "Only diploid genotypes supported");
        goto out;
    }
    if (check_array("out_buf", out_buf, 1) != 0) {
        goto out;
    }
    if (check_dtype("out_buf", out_buf, NPY_UINT8) != 0) {
        goto out;
    }
    if (check_array_writeable("out_buf", out_buf) != 0) {
        goto out;
    }
    num_variants = PyArray_DIMS(genotypes)[0];
    num_samples = PyArray_DIMS(genotypes)[1];
    expected = ((num_samples + 3) / 4) * num_variants;
    if (PyArray_DIMS(out_buf)[0] < expected) {
        PyErr_Format(VczBufferTooSmall, "out_buf is too small: got %zd bytes, need %zd",
            (Py_ssize_t) PyArray_DIMS(out_buf)[0], (Py_ssize_t) expected);
        goto out;
    }

    Py_BEGIN_ALLOW_THREADS
    vcz_encode_plink((size_t) num_variants, (size_t) num_samples,
        PyArray_DATA(genotypes), PyArray_DATA(out_buf));
    Py_END_ALLOW_THREADS

    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
vcztools_encode_bgen_geno_blocks(PyObject *self, PyObject *args)
{
    PyObject *ret = NULL;
    PyArrayObject *genotypes = NULL;
    PyArrayObject *phased = NULL;
    PyArrayObject *encoded = NULL;
    PyArrayObject *lens = NULL;
    npy_intp num_variants, num_samples;
    npy_intp out_dims[2];
    npy_intp lens_dims[1];
    size_t row_stride;
    int err;

    if (!PyArg_ParseTuple(
            args, "O!O!", &PyArray_Type, &genotypes, &PyArray_Type, &phased)) {
        goto out;
    }
    if (check_array("genotypes", genotypes, 3) != 0) {
        goto out;
    }
    if (check_dtype("genotypes", genotypes, NPY_INT8) != 0) {
        goto out;
    }
    if (PyArray_DIMS(genotypes)[2] != 2) {
        PyErr_Format(PyExc_ValueError,
            "BGEN encoder expects (V, S, 2) genotypes "
            "(haploid inputs are normalised to ploidy=2 with -2 padding)");
        goto out;
    }
    if (check_array("phased", phased, 1) != 0) {
        goto out;
    }
    if (check_dtype("phased", phased, NPY_BOOL) != 0) {
        goto out;
    }
    num_variants = PyArray_DIMS(genotypes)[0];
    num_samples = PyArray_DIMS(genotypes)[1];
    if (PyArray_DIMS(phased)[0] != num_variants) {
        PyErr_Format(PyExc_ValueError, "phased.shape[0] must equal genotypes.shape[0]");
        goto out;
    }
    row_stride = vcz_bgen_geno_block_row_max_size((size_t) num_samples);
    out_dims[0] = num_variants;
    out_dims[1] = (npy_intp) row_stride;
    encoded = (PyArrayObject *) PyArray_SimpleNew(2, out_dims, NPY_UINT8);
    if (encoded == NULL) {
        goto out; // GCOVR_EXCL_LINE
    }
    lens_dims[0] = num_variants;
    lens = (PyArrayObject *) PyArray_SimpleNew(1, lens_dims, NPY_UINT32);
    if (lens == NULL) {
        goto out; // GCOVR_EXCL_LINE
    }

    Py_BEGIN_ALLOW_THREADS
    err = vcz_encode_bgen_geno_blocks((size_t) num_variants, (size_t) num_samples,
        PyArray_DATA(genotypes), PyArray_DATA(phased), PyArray_DATA(encoded), row_stride,
        PyArray_DATA(lens));
    Py_END_ALLOW_THREADS

    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("(OO)", (PyObject *) encoded, (PyObject *) lens);
out:
    Py_XDECREF(encoded);
    Py_XDECREF(lens);
    return ret;
}

static PyObject *
vcztools_encode_bgen_chunk_slice_level0(PyObject *self, PyObject *args)
{
    PyObject *ret = NULL;
    PyArrayObject *varid = NULL;
    PyArrayObject *rsid = NULL;
    PyArrayObject *chrom = NULL;
    PyArrayObject *allele1 = NULL;
    PyArrayObject *allele2 = NULL;
    PyArrayObject *position = NULL;
    PyArrayObject *genotypes = NULL;
    PyArrayObject *phased = NULL;
    PyArrayObject *out_buf = NULL;
    Py_ssize_t uniform_ploidy;
    npy_intp num_variants;
    npy_intp num_samples;
    npy_intp varid_max, rsid_max, chrom_max, allele_max;
    npy_intp geno_size;
    npy_intp expected_bytes;
    size_t payload_size;
    int err;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!O!n", &PyArray_Type, &varid,
            &PyArray_Type, &rsid, &PyArray_Type, &chrom, &PyArray_Type, &allele1,
            &PyArray_Type, &allele2, &PyArray_Type, &position, &PyArray_Type, &genotypes,
            &PyArray_Type, &phased, &PyArray_Type, &out_buf, &uniform_ploidy)) {
        goto out;
    }

    if (check_array("varid", varid, 2) != 0) {
        goto out;
    }
    if (check_dtype("varid", varid, NPY_UINT8) != 0) {
        goto out;
    }
    if (check_array("rsid", rsid, 2) != 0) {
        goto out;
    }
    if (check_dtype("rsid", rsid, NPY_UINT8) != 0) {
        goto out;
    }
    if (check_array("chrom", chrom, 2) != 0) {
        goto out;
    }
    if (check_dtype("chrom", chrom, NPY_UINT8) != 0) {
        goto out;
    }
    if (check_array("allele1", allele1, 2) != 0) {
        goto out;
    }
    if (check_dtype("allele1", allele1, NPY_UINT8) != 0) {
        goto out;
    }
    if (check_array("allele2", allele2, 2) != 0) {
        goto out;
    }
    if (check_dtype("allele2", allele2, NPY_UINT8) != 0) {
        goto out;
    }
    if (check_array("position", position, 1) != 0) {
        goto out;
    }
    if (check_dtype("position", position, NPY_INT32) != 0) {
        goto out;
    }
    if (check_array("genotypes", genotypes, 3) != 0) {
        goto out;
    }
    if (check_dtype("genotypes", genotypes, NPY_INT8) != 0) {
        goto out;
    }
    if (PyArray_DIMS(genotypes)[2] != 2) {
        PyErr_Format(PyExc_ValueError,
            "BGEN encoder expects (V, S, 2) genotypes "
            "(haploid inputs are normalised to ploidy=2 with -2 padding)");
        goto out;
    }
    if (check_array("phased", phased, 1) != 0) {
        goto out;
    }
    if (check_dtype("phased", phased, NPY_BOOL) != 0) {
        goto out;
    }
    if (check_array("out_buf", out_buf, 1) != 0) {
        goto out;
    }
    if (check_dtype("out_buf", out_buf, NPY_UINT8) != 0) {
        goto out;
    }
    if (check_array_writeable("out_buf", out_buf) != 0) {
        goto out;
    }

    num_variants = PyArray_DIMS(varid)[0];
    num_samples = PyArray_DIMS(genotypes)[1];
    varid_max = PyArray_DIMS(varid)[1];
    rsid_max = PyArray_DIMS(rsid)[1];
    chrom_max = PyArray_DIMS(chrom)[1];
    allele_max = PyArray_DIMS(allele1)[1];

    if (PyArray_DIMS(rsid)[0] != num_variants || PyArray_DIMS(chrom)[0] != num_variants
        || PyArray_DIMS(allele1)[0] != num_variants
        || PyArray_DIMS(allele2)[0] != num_variants
        || PyArray_DIMS(position)[0] != num_variants
        || PyArray_DIMS(genotypes)[0] != num_variants
        || PyArray_DIMS(phased)[0] != num_variants) {
        PyErr_Format(PyExc_ValueError,
            "All per-variant inputs must share num_variants axis (got %zd)",
            (Py_ssize_t) num_variants);
        goto out;
    }
    if (PyArray_DIMS(allele2)[1] != allele_max) {
        PyErr_Format(
            PyExc_ValueError, "allele1 and allele2 must share the same max width");
        goto out;
    }
    if (uniform_ploidy != 1 && uniform_ploidy != 2) {
        PyErr_Format(PyExc_ValueError, "uniform_ploidy must be 1 or 2 (got %zd)",
            (Py_ssize_t) uniform_ploidy);
        goto out;
    }

    /* The kernel writes num_variants * bytes_per_variant bytes. */
    geno_size = 10 + ((npy_intp) uniform_ploidy + 1) * num_samples;
    payload_size = 2 + 4 + (size_t) geno_size;
    if (geno_size == 0) {
        payload_size += 5;
    } else {
        payload_size += 5 * (((size_t) geno_size + 65534) / 65535);
    }
    expected_bytes = num_variants
                     * (28 + varid_max + rsid_max + chrom_max + 2 * allele_max
                         + (npy_intp) payload_size);
    if (PyArray_DIMS(out_buf)[0] < expected_bytes) {
        PyErr_Format(VczBufferTooSmall, "out_buf is too small: got %zd bytes, need %zd",
            (Py_ssize_t) PyArray_DIMS(out_buf)[0], (Py_ssize_t) expected_bytes);
        goto out;
    }

    Py_BEGIN_ALLOW_THREADS
    err = vcz_encode_bgen_chunk_slice_level0((size_t) num_variants, (size_t) num_samples,
        (size_t) uniform_ploidy, PyArray_DATA(varid), (size_t) varid_max,
        PyArray_DATA(rsid), (size_t) rsid_max, PyArray_DATA(chrom), (size_t) chrom_max,
        PyArray_DATA(allele1), PyArray_DATA(allele2), (size_t) allele_max,
        PyArray_DATA(position), PyArray_DATA(genotypes), PyArray_DATA(phased),
        PyArray_DATA(out_buf));
    Py_END_ALLOW_THREADS

    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyMethodDef vcztools_methods[] = {
    { .ml_name = "encode_plink",
        .ml_meth = (PyCFunction) vcztools_encode_plink,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Encode genotypes in plink format" },
    { .ml_name = "encode_bgen_geno_blocks",
        .ml_meth = (PyCFunction) vcztools_encode_bgen_geno_blocks,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Encode BGEN Layout-2 genotype blocks (mixed-ploidy "
                  "supported). Returns (buffer, lengths)." },
    { .ml_name = "encode_bgen_chunk_slice_level0",
        .ml_meth = (PyCFunction) vcztools_encode_bgen_chunk_slice_level0,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Encode a full BGEN variant-block slice for the fixed-size "
                  "BgenEncoder path (zlib level 0, uniform ploidy). Writes "
                  "num_variants * bytes_per_variant bytes into out_buf." },
    { NULL } /* Sentinel */
};

static struct PyModuleDef vcztoolsmodule = { PyModuleDef_HEAD_INIT, "_vcztools",
    "C interface for vcztools.", -1, vcztools_methods, NULL, NULL, NULL, NULL };

PyObject *
PyInit__vcztools(void)
{
    PyObject *module = PyModule_Create(&vcztoolsmodule);

    if (module == NULL) {
        return NULL;
    }
    /* Initialise numpy */
    import_array();

    VczBufferTooSmall = PyErr_NewException("_vcztools.VczBufferTooSmall", NULL, NULL);
    Py_INCREF(VczBufferTooSmall);
    PyModule_AddObject(module, "VczBufferTooSmall", VczBufferTooSmall);

    /* VcfEncoder type */
    if (PyType_Ready(&VcfEncoderType) < 0) {
        return NULL;
    }
    Py_INCREF(&VcfEncoderType);
    PyModule_AddObject(module, "VcfEncoder", (PyObject *) &VcfEncoderType);

    return module;
}
