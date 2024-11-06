
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <stddef.h> /* for offsetof() */
#include <stdbool.h>
#include <numpy/arrayobject.h>

#include "vcf_encoder.h"

// clang-format off
typedef struct {
    PyObject_HEAD
    vcz_variant_encoder_t *vcf_encoder;
    PyObject *arrays;
} VcfEncoder;
// clang-format on

static PyObject *VczBufferTooSmall;

static void
handle_library_error(int err)
{
    switch (err) {
        case VCZ_ERR_BUFFER_OVERFLOW:
            PyErr_Format(
                VczBufferTooSmall, "Error: %d; specified buffer size is too small", err);
            break;
        // TODO handle the other error types.
        default:
            PyErr_Format(PyExc_ValueError, "Error occured: %d: ", err);
    }
}

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
    line_length = vcz_variant_encoder_encode(
        self->vcf_encoder, (size_t) row, buf, (size_t) bufsize);
    if (line_length < 0) {
        handle_library_error((int) line_length);
        goto out;
    }
    ret = Py_BuildValue("s#", buf, (Py_ssize_t) line_length);
out:
    PyMem_Free(buf);
    return ret;
}

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
    { .ml_name = "print_state",
        .ml_meth = (PyCFunction) VcfEncoder_print_state,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Debug method to print out the low-level state" },
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

static PyTypeObject VcfEncoderType = {
    // clang-format off
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
    // clang-format on
};

/*===================================================================
 * Module level code.
 *===================================================================
 */

static PyMethodDef vcztools_methods[] = {
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
