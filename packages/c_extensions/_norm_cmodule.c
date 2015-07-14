/* File: _norm_cmodule.c
 * This file is auto-generated with f2py (version:2).
 * f2py is a Fortran to Python Interface Generator (FPIG), Second Edition,
 * written by Pearu Peterson <pearu@cens.ioc.ee>.
 * See http://cens.ioc.ee/projects/f2py2e/
 * Generation date: Tue Aug  5 10:33:28 2014
 * $Revision:$
 * $Date:$
 * Do not edit this file directly unless you know what you are doing!!!
 */
#ifdef __cplusplus
extern "C" {
#endif

/*********************** See f2py2e/cfuncs.py: includes ***********************/
#include "Python.h"
#include <stdarg.h>
#include "fortranobject.h"
/*need_includes0*/

/**************** See f2py2e/rules.py: mod_rules['modulebody'] ****************/
static PyObject *_norm_c_error;
static PyObject *_norm_c_module;

/*********************** See f2py2e/cfuncs.py: typedefs ***********************/
/*need_typedefs*/

/****************** See f2py2e/cfuncs.py: typedefs_generated ******************/
/*need_typedefs_generated*/

/********************** See f2py2e/cfuncs.py: cppmacros **********************/
#define rank(var) var ## _Rank
#define shape(var,dim) var ## _Dims[dim]
#define old_rank(var) (((PyArrayObject *)(capi_ ## var ## _tmp))->nd)
#define old_shape(var,dim) (((PyArrayObject *)(capi_ ## var ## _tmp))->dimensions[dim])
#define fshape(var,dim) shape(var,rank(var)-dim-1)
#define len(var) shape(var,0)
#define flen(var) fshape(var,0)
#define old_size(var) PyArray_SIZE((PyArrayObject *)(capi_ ## var ## _tmp))
/* #define index(i) capi_i ## i */
#define slen(var) capi_ ## var ## _len
#define size(var, ...) f2py_size((PyArrayObject *)(capi_ ## var ## _tmp), ## __VA_ARGS__, -1)

#ifdef DEBUGCFUNCS
#define CFUNCSMESS(mess) fprintf(stderr,"debug-capi:"mess);
#define CFUNCSMESSPY(mess,obj) CFUNCSMESS(mess) \
  PyObject_Print((PyObject *)obj,stderr,Py_PRINT_RAW);\
  fprintf(stderr,"\n");
#else
#define CFUNCSMESS(mess)
#define CFUNCSMESSPY(mess,obj)
#endif

#ifndef max
#define max(a,b) ((a > b) ? (a) : (b))
#endif
#ifndef min
#define min(a,b) ((a < b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a,b) ((a > b) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a,b) ((a < b) ? (a) : (b))
#endif

#if defined(PREPEND_FORTRAN)
#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) _##F
#else
#define F_FUNC(f,F) _##f
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) _##F##_
#else
#define F_FUNC(f,F) _##f##_
#endif
#endif
#else
#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) F
#else
#define F_FUNC(f,F) f
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) F##_
#else
#define F_FUNC(f,F) f##_
#endif
#endif
#endif
#if defined(UNDERSCORE_G77)
#define F_FUNC_US(f,F) F_FUNC(f##_,F##_)
#else
#define F_FUNC_US(f,F) F_FUNC(f,F)
#endif


/************************ See f2py2e/cfuncs.py: cfuncs ************************/
static int f2py_size(PyArrayObject* var, ...)
{
  npy_int sz = 0;
  npy_int dim;
  npy_int rank;
  va_list argp;
  va_start(argp, var);
  dim = va_arg(argp, npy_int);
  if (dim==-1)
    {
      sz = PyArray_SIZE(var);
    }
  else
    {
      rank = PyArray_NDIM(var);
      if (dim>=1 && dim<=rank)
        sz = PyArray_DIM(var, dim-1);
      else
        fprintf(stderr, "f2py_size: 2nd argument value=%d fails to satisfy 1<=value<=%d. Result will be 0.\n", dim, rank);
    }
  va_end(argp);
  return sz;
}

static int int_from_pyobj(int* v,PyObject *obj,const char *errmess) {
  PyObject* tmp = NULL;
  if (PyInt_Check(obj)) {
    *v = (int)PyInt_AS_LONG(obj);
    return 1;
  }
  tmp = PyNumber_Int(obj);
  if (tmp) {
    *v = PyInt_AS_LONG(tmp);
    Py_DECREF(tmp);
    return 1;
  }
  if (PyComplex_Check(obj))
    tmp = PyObject_GetAttrString(obj,"real");
  else if (PyString_Check(obj) || PyUnicode_Check(obj))
    /*pass*/;
  else if (PySequence_Check(obj))
    tmp = PySequence_GetItem(obj,0);
  if (tmp) {
    PyErr_Clear();
    if (int_from_pyobj(v,tmp,errmess)) {Py_DECREF(tmp); return 1;}
    Py_DECREF(tmp);
  }
  {
    PyObject* err = PyErr_Occurred();
    if (err==NULL) err = _norm_c_error;
    PyErr_SetString(err,errmess);
  }
  return 0;
}


/********************* See f2py2e/cfuncs.py: userincludes *********************/
/*need_userincludes*/

/********************* See f2py2e/capi_rules.py: usercode *********************/


/* See f2py2e/rules.py */
extern void norm(int,double*,int*,double*);
/*eof externroutines*/

/******************** See f2py2e/capi_rules.py: usercode1 ********************/


/******************* See f2py2e/cb_rules.py: buildcallback *******************/
/*need_callbacks*/

/*********************** See f2py2e/rules.py: buildapi ***********************/

/************************************ norm ************************************/
static char doc_f2py_rout__norm_c_norm[] = "\
norm(m,sm,iptr,dat)\n\nWrapper for ``norm``.\
\n\nParameters\n----------\n"
"m : input int\n"
"sm : input rank-1 array('d') with bounds (m)\n"
"iptr : input rank-1 array('i') with bounds (m + 1)\n"
"dat : in/output rank-1 array('d') with bounds (:)";
/* extern void norm(int,double*,int*,double*); */
static PyObject *f2py_rout__norm_c_norm(const PyObject *capi_self,
                           PyObject *capi_args,
                           PyObject *capi_keywds,
                           void (*f2py_func)(int,double*,int*,double*)) {
  PyObject * volatile capi_buildvalue = NULL;
  volatile int f2py_success = 1;
/*decl*/

  int m = 0;
  PyObject *m_capi = Py_None;
  double *sm = NULL;
  npy_intp sm_Dims[1] = {-1};
  const int sm_Rank = 1;
  PyArrayObject *capi_sm_tmp = NULL;
  int capi_sm_intent = 0;
  PyObject *sm_capi = Py_None;
  int *iptr = NULL;
  npy_intp iptr_Dims[1] = {-1};
  const int iptr_Rank = 1;
  PyArrayObject *capi_iptr_tmp = NULL;
  int capi_iptr_intent = 0;
  PyObject *iptr_capi = Py_None;
  double *dat = NULL;
  npy_intp dat_Dims[1] = {-1};
  const int dat_Rank = 1;
  PyArrayObject *capi_dat_tmp = NULL;
  int capi_dat_intent = 0;
  PyObject *dat_capi = Py_None;
  static char *capi_kwlist[] = {"m","sm","iptr","dat",NULL};

/*routdebugenter*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_clock();
#endif
  if (!PyArg_ParseTupleAndKeywords(capi_args,capi_keywds,\
    "OOOO:_norm_c.norm",\
    capi_kwlist,&m_capi,&sm_capi,&iptr_capi,&dat_capi))
    return NULL;
/*frompyobj*/
  /* Processing variable dat */
  ;
  capi_dat_intent |= F2PY_INTENT_INOUT|F2PY_INTENT_C;
  capi_dat_tmp = array_from_pyobj(NPY_DOUBLE,dat_Dims,dat_Rank,capi_dat_intent,dat_capi);
  if (capi_dat_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(_norm_c_error,"failed in converting 4th argument `dat' of _norm_c.norm to C/Fortran array" );
  } else {
    dat = (double *)(capi_dat_tmp->data);

  /* Processing variable m */
    f2py_success = int_from_pyobj(&m,m_capi,"_norm_c.norm() 1st argument (m) can't be converted to int");
  if (f2py_success) {
  /* Processing variable iptr */
  iptr_Dims[0]=m + 1;
  capi_iptr_intent |= F2PY_INTENT_IN|F2PY_INTENT_C;
  capi_iptr_tmp = array_from_pyobj(NPY_INT,iptr_Dims,iptr_Rank,capi_iptr_intent,iptr_capi);
  if (capi_iptr_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(_norm_c_error,"failed in converting 3rd argument `iptr' of _norm_c.norm to C/Fortran array" );
  } else {
    iptr = (int *)(capi_iptr_tmp->data);

  /* Processing variable sm */
  sm_Dims[0]=m;
  capi_sm_intent |= F2PY_INTENT_IN|F2PY_INTENT_C;
  capi_sm_tmp = array_from_pyobj(NPY_DOUBLE,sm_Dims,sm_Rank,capi_sm_intent,sm_capi);
  if (capi_sm_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(_norm_c_error,"failed in converting 2nd argument `sm' of _norm_c.norm to C/Fortran array" );
  } else {
    sm = (double *)(capi_sm_tmp->data);

/*end of frompyobj*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_call_clock();
#endif
/*callfortranroutine*/
        (*f2py_func)(m,sm,iptr,dat);
if (PyErr_Occurred())
  f2py_success = 0;
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_call_clock();
#endif
/*end of callfortranroutine*/
    if (f2py_success) {
/*pyobjfrom*/
/*end of pyobjfrom*/
    CFUNCSMESS("Building return value.\n");
    capi_buildvalue = Py_BuildValue("");
/*closepyobjfrom*/
/*end of closepyobjfrom*/
    } /*if (f2py_success) after callfortranroutine*/
/*cleanupfrompyobj*/
  if((PyObject *)capi_sm_tmp!=sm_capi) {
    Py_XDECREF(capi_sm_tmp); }
  }  /*if (capi_sm_tmp == NULL) ... else of sm*/
  /* End of cleaning variable sm */
  if((PyObject *)capi_iptr_tmp!=iptr_capi) {
    Py_XDECREF(capi_iptr_tmp); }
  }  /*if (capi_iptr_tmp == NULL) ... else of iptr*/
  /* End of cleaning variable iptr */
  } /*if (f2py_success) of m*/
  /* End of cleaning variable m */
  if((PyObject *)capi_dat_tmp!=dat_capi) {
    Py_XDECREF(capi_dat_tmp); }
  }  /*if (capi_dat_tmp == NULL) ... else of dat*/
  /* End of cleaning variable dat */
/*end of cleanupfrompyobj*/
  if (capi_buildvalue == NULL) {
/*routdebugfailure*/
  } else {
/*routdebugleave*/
  }
  CFUNCSMESS("Freeing memory.\n");
/*freemem*/
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_clock();
#endif
  return capi_buildvalue;
}
/******************************** end of norm ********************************/
/*eof body*/

/******************* See f2py2e/f90mod_rules.py: buildhooks *******************/
/*need_f90modhooks*/

/************** See f2py2e/rules.py: module_rules['modulebody'] **************/

/******************* See f2py2e/common_rules.py: buildhooks *******************/

/*need_commonhooks*/

/**************************** See f2py2e/rules.py ****************************/

static FortranDataDef f2py_routine_defs[] = {
  {"norm",-1,{{-1}},0,(char *)norm,(f2py_init_func)f2py_rout__norm_c_norm,doc_f2py_rout__norm_c_norm},

/*eof routine_defs*/
  {NULL}
};

static PyMethodDef f2py_module_methods[] = {

  {NULL,NULL}
};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "_norm_c",
  NULL,
  -1,
  f2py_module_methods,
  NULL,
  NULL,
  NULL,
  NULL
};
#endif

#if PY_VERSION_HEX >= 0x03000000
#define RETVAL m
PyMODINIT_FUNC PyInit__norm_c(void) {
#else
#define RETVAL
PyMODINIT_FUNC init_norm_c(void) {
#endif
  int i;
  PyObject *m,*d, *s;
#if PY_VERSION_HEX >= 0x03000000
  m = _norm_c_module = PyModule_Create(&moduledef);
#else
  m = _norm_c_module = Py_InitModule("_norm_c", f2py_module_methods);
#endif
  Py_TYPE(&PyFortran_Type) = &PyType_Type;
  import_array();
  if (PyErr_Occurred())
    {PyErr_SetString(PyExc_ImportError, "can't initialize module _norm_c (failed to import numpy)"); return RETVAL;}
  d = PyModule_GetDict(m);
  s = PyString_FromString("$Revision: $");
  PyDict_SetItemString(d, "__version__", s);
#if PY_VERSION_HEX >= 0x03000000
  s = PyUnicode_FromString(
#else
  s = PyString_FromString(
#endif
    "This module '_norm_c' is auto-generated with f2py (version:2).\nFunctions:\n"
"  norm(m,sm,iptr,dat)\n"
".");
  PyDict_SetItemString(d, "__doc__", s);
  _norm_c_error = PyErr_NewException ("_norm_c.error", NULL, NULL);
  Py_DECREF(s);
  for(i=0;f2py_routine_defs[i].name!=NULL;i++)
    PyDict_SetItemString(d, f2py_routine_defs[i].name,PyFortranObject_NewAsAttr(&f2py_routine_defs[i]));

/*eof initf2pywraphooks*/
/*eof initf90modhooks*/

/*eof initcommonhooks*/


#ifdef F2PY_REPORT_ATEXIT
  if (! PyErr_Occurred())
    on_exit(f2py_report_on_exit,(void*)"_norm_c");
#endif

  return RETVAL;
}
#ifdef __cplusplus
}
#endif
