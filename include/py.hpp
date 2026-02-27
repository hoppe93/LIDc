#ifndef _LID_PY_HPP
#define _LID_PY_HPP

extern "C" {
static PyObject *lid_integrate_dream(PyObject*, PyObject*, PyObject*);
static PyObject *lid_integrate_dream_h5(PyObject*, PyObject*, PyObject*);
static PyObject *lid_greens_function(PyObject*, PyObject*, PyObject*);
}

#endif/*_LID_PY_HPP*/
