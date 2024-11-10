#ifndef _LID_NUMPY_HPP
#define _LID_NUMPY_HPP

#define PY_ARRAY_UNIQUE_SYMBOL LID_PyArray_API

// Macro for translation unit
#ifndef INIT_NUMPY_ARRAY_CPP
#	define NO_IMPORT_ARRAY
#endif

#ifndef NPY_NO_DEPRECATED_API
#	define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif

// Include numpy
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>

#endif/*_LID_NUMPY_HPP*/
