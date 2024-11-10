/**
 * Python interface to the LIDc code.
 */

#ifndef PY_SSIZE_T_CLEAN
#	define PY_SSIZE_T_CLEAN
#endif
#define INIT_NUMPY_ARRAY_CPP

#include <Python.h>
#include "LIDException.hpp"
#include "dreamoutput.hpp"
#include "integrate.hpp"
#include "numpy.hpp"
#include "py.hpp"

/**
 * Initialize numpy (must (allegedly) be done in
 * every _file_ using the NumPy C API)
 */
int init_numpy() {
	import_array();
	return 0;
}
const static int numpy_initialized = init_numpy();

static PyMethodDef lidMethods[] = {
	{"integrate_dream", (PyCFunction)(void(*)(void))lid_integrate_dream, METH_VARARGS | METH_KEYWORDS, "Calculates the line-integrated density for the given DREAMOutput object."},
	{"integrate_dream_h5", (PyCFunction)(void(*)(void))lid_integrate_dream_h5, METH_VARARGS | METH_KEYWORDS, "Calculates the line-integrated density for the given DREAMOutput object."},
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef lidModule = {
	PyModuleDef_HEAD_INIT,
	"liblid",				// Name of module
	nullptr,				// Module documentation
	-1,						// Size of per-interpreter state of the module,
							// or -1 if the module keeps state in global variables
	lidMethods,
	nullptr,
	nullptr,
	nullptr,
	nullptr
};

/**
 * Module initialization.
 */
PyMODINIT_FUNC PyInit_liblid() {
	return PyModule_Create(&lidModule);
}


/**
 * Load a 1D list from the given Python dictionary.
 */
real_t *lid_load_list(
	PyObject *arr, len_t *n
) {
	if (PyArray_Check(arr)) {
		PyArrayObject *_t = reinterpret_cast<PyArrayObject*>(arr);
		npy_intp *_dims = PyArray_DIMS(_t);

		*n = _dims[0];
		real_t *x = new real_t[*n];
		real_t *d = reinterpret_cast<real_t*>(PyArray_DATA(_t));
		for (len_t i = 0; i < *n; i++)
			x[i] = d[i];
		
		return x;
	} else if (PyList_Check(arr)) {
		Py_ssize_t _n = PyList_Size(arr);
		*n = _n;

		real_t *x = new real_t[_n];
		for (Py_ssize_t i = 0; i < _n; i++) {
			PyObject *li = PyList_GetItem(arr, i);

			if (PyFloat_Check(li))
				x[i] = PyFloat_AsDouble(li);
			else if (PyLong_Check(li))
				x[i] = static_cast<real_t>(PyLong_AsLong(li));
			else
				return nullptr;
		}

		return x;
	} else
		return nullptr;
}
real_t *lid_load_list_from_dict(
	PyObject *dict, const char *name, len_t *n
) {
	PyObject *arr = PyDict_GetItemString(dict, name);
	real_t *v = lid_load_list(arr, n);
	if (v == nullptr)
		throw LID::LIDException(
			"DREAMOutput key '%s': Unrecognized data type of specified value. Expected numpy array.",
			name
		);
	return v;
}

/**
 * Load a 2D array from the given Python dictionary.
 */
real_t *lid_load_array(
	PyArrayObject *arr, len_t *n1, len_t *n2
) {
	npy_intp *_dims = PyArray_DIMS(arr);

	*n1 = _dims[0];
	*n2 = _dims[1];
	real_t *x = new real_t[(*n1)*(*n2)];
	real_t *d = reinterpret_cast<real_t*>(PyArray_DATA(arr));
	for (len_t i = 0; i < (*n1)*(*n2); i++)
		x[i] = d[i];
	
	return x;
}
real_t *lid_load_array_from_dict(
	PyObject *dict, const char *name, len_t *n1, len_t *n2
) {
	PyArrayObject *_t = reinterpret_cast<PyArrayObject*>(
		PyDict_GetItemString(dict, name)
	);
	real_t *v = lid_load_array(_t, n1, n2);
	if (v == nullptr)
		throw LID::LIDException(
			"DREAMOutput key '%s': Unrecognized data type of specified value. Expected numpy array.",
			name
		);
	return v;
}

/**
 * Internal integration routine.
 */
PyObject *lid_integrate_internal(
	struct LID::dream_data *dd, PyObject *x0_obj, PyObject *nhat_obj
) {
	//////////////////////////////
	/// Load detector settings
	//////////////////////////////
	len_t n3;
	real_t *x0 = lid_load_list(x0_obj, &n3);
	if (x0 == nullptr) {
		PyErr_SetString(
			PyExc_RuntimeError,
			"Invalid type for input argument 'x0'."
		);
		return NULL;
	}
	real_t *nhat = lid_load_list(nhat_obj, &n3);
	if (x0 == nullptr) {
		PyErr_SetString(
			PyExc_RuntimeError,
			"Invalid type for input argument 'nhat'."
		);
		return NULL;
	}

	struct LID::detector *det = new struct LID::detector(x0, nhat);

	delete [] nhat;
	delete [] x0;

	//////////////////////////////
	/// Integrate density
	//////////////////////////////
	real_t *n = LID::line_integrated_density(dd, det);

	npy_intp _dims = dd->nt;
	PyObject *arr = PyArray_SimpleNew(1, &_dims, NPY_DOUBLE);
	PyObject *tarr = PyArray_SimpleNew(1, &_dims, NPY_DOUBLE);
	real_t *p = reinterpret_cast<real_t*>(
		PyArray_DATA(reinterpret_cast<PyArrayObject*>(arr))
	);
	real_t *t = reinterpret_cast<real_t*>(
		PyArray_DATA(reinterpret_cast<PyArrayObject*>(tarr))
	);
	for (npy_intp i = 0; i < _dims; i++) {
		p[i] = n[i];
		t[i] = dd->t[i];
	}
	
	delete [] n;

	return PyTuple_Pack(2, tarr, arr);
}

extern "C" {

/**
 * Calculate the line-integrated density for the given DREAMOutput object.
 */
static PyObject *lid_integrate_dream(
	PyObject*, PyObject *args, PyObject *kwargs
) {
	static const char *kwlist[] = {"do", "x0", "nhat", NULL};
	PyObject *do_dict, *x0_obj, *nhat_obj;
	if (!PyArg_ParseTupleAndKeywords(
		args, kwargs, "OOO", const_cast<char**>(kwlist),
		&do_dict, &x0_obj, &nhat_obj)
	) {
		PyErr_SetString(
			PyExc_RuntimeError,
			"The arguments to 'integrate_dream()' must be Python dictionaries."
		);
		return NULL;
	}

	//////////////////////
	/// Load DREAMOutput
	//////////////////////
	struct LID::dream_data *dd = new struct LID::dream_data;
	PyObject *grid = PyDict_GetItemString(do_dict, "grid");

	// Time grid
	len_t nt;
	dd->t = lid_load_list_from_dict(grid, "t", &nt);
	dd->nt = nt;

	// Radial grid
	len_t nr, nr_f;
	dd->r   = lid_load_list_from_dict(grid, "r", &nr);
	dd->r_f = lid_load_list_from_dict(grid, "r_f", &nr_f);
	dd->dr  = lid_load_list_from_dict(grid, "dr", &nr);
	dd->nr  = nr;

	len_t _n;
	real_t *R0 = lid_load_list_from_dict(grid, "R0", &_n);
	dd->R0 = R0[0];
	delete [] R0;

	// Flux surfaces
	len_t ntheta;
	PyObject *eq = PyDict_GetItemString(grid, "eq");
	
	dd->ROverR0   = lid_load_array_from_dict(eq, "ROverR0", &ntheta, &nr);
	dd->ROverR0_f = lid_load_array_from_dict(eq, "ROverR0_f", &ntheta, &nr_f);
	dd->Z         = lid_load_array_from_dict(eq, "Z", &ntheta, &nr);
	dd->Z_f       = lid_load_array_from_dict(eq, "Z_f", &ntheta, &nr_f);

	lid_load_list_from_dict(eq, "theta", &ntheta);
	dd->ntheta = ntheta;

	// Electron density
	PyObject *eqsys = PyDict_GetItemString(do_dict, "eqsys");
	dd->ne = lid_load_array_from_dict(eqsys, "n_cold", &nt, &nr);

	Py_DECREF(eqsys);
	Py_DECREF(eq);
	Py_DECREF(grid);

	return lid_integrate_internal(dd, x0_obj, nhat_obj);
}


/**
 * Calculate the line-integrated density for a given DREAM output, taking data
 * from the specified HDF5 file.
 */
static PyObject *lid_integrate_dream_h5(
	PyObject*, PyObject *args, PyObject *kwargs
) {
	static const char *kwlist[] = {"filename", "x0", "nhat", NULL};
	PyObject *x0_obj, *nhat_obj;
	const char *filename;

	if (!PyArg_ParseTupleAndKeywords(
		args, kwargs, "sOO", const_cast<char**>(kwlist),
		&filename, &x0_obj, &nhat_obj)
	) {
		PyErr_SetString(
			PyExc_RuntimeError,
			"The arguments to 'integrate_dream()' must be Python dictionaries."
		);
		return NULL;
	}

	struct LID::dream_data *dd = LID::load_dream_output(filename);

	// Do integration
	return lid_integrate_internal(dd, x0_obj, nhat_obj);
}

}
