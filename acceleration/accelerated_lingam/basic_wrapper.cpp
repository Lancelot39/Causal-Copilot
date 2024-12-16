#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include "numpy/arrayobject.h"
#include <cstdio>
#include <string>
#include <vector>
#include "basic.cuh"

static PyObject *causal_order_wrapper(PyObject *self, PyObject *args) {
    PyObject *data_obj;
    // PyObject *mlist_obj;
    int m, n;

    // Parse a Python tuple and two ints from the args
    // if (!PyArg_ParseTuple(args, "O!iiO!", &PyArray_Type, &data_obj, &m, &n, &PyArray_Type, &mlist_obj)) {


    if (!PyArg_ParseTuple(args, "O!ii", &PyArray_Type, &data_obj, &m, &n)) {
        return nullptr;
    }

    // Ensure data_obj is a contiguous array of doubles
    PyArrayObject *data_array = (PyArrayObject*)PyArray_FROM_OTF(data_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (data_array == nullptr) {
        return nullptr;
    }

    // // Ensure mlist_obj is a contiguous array of doubles
    // PyArrayObject *mlist_array = (PyArrayObject*)PyArray_FROM_OTF(mlist_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    // if (mlist_array == nullptr) {
    //     Py_DECREF(data_array);
    //     return nullptr;
    // }

    // Get pointers to the data in the arrays
    double *data = (double*)PyArray_DATA(data_array);
    // double *mlist = (double*)PyArray_DATA(mlist_array);

    // Call the C++ function
    double *result = causal_order(data, m, n);

    // Handle the result
    // Assuming the result is an array of doubles of size n
    npy_intp dims[1] = {n};
    PyObject *ret = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, result);

    // Tell NumPy it owns the data, and it should free the memory when the array is deleted
    PyArray_ENABLEFLAGS((PyArrayObject*)ret, NPY_ARRAY_OWNDATA);

    // Decrease reference counts to the arrays
    Py_DECREF(data_array);
    // Py_DECREF(mlist_array);

    return ret;
}

static PyMethodDef CudaextMethods[] = {
    {"causal_order", causal_order_wrapper, METH_VARARGS, "Calculate causal order"},
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef lingam_cuda_definition = {
    PyModuleDef_HEAD_INIT,
    "lingam_cuda",
    "CULiNGAM accelerates LiNGAM analysis on GPUs.",
    -1,
    CudaextMethods
};

PyMODINIT_FUNC PyInit_lingam_cuda(void) {
    import_array();
    return PyModule_Create(&lingam_cuda_definition);
}
