#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>


#include "Rinternals.h"
#include "survproto.h"
#include "survS.h"

static const R_CallMethodDef Callentries[] = {
  {"agmart3", (DL_FUNC) &agmart3, 8},
  {"agscore2", (DL_FUNC) &agscore2, 6},
  {"coxcount1", (DL_FUNC) &coxcount1, 2},
  {"coxcount2", (DL_FUNC) &coxcount2, 4},
  {"coxscore2", (DL_FUNC) &coxscore2, 6},
  {NULL, NULL, 0}
};

static const R_CMethodDef Centries[] = {
  {"agsurv4",    (DL_FUNC) &agsurv4,    6},
  {"agsurv5",    (DL_FUNC) &agsurv5,   10},
  {"coxmart",    (DL_FUNC) &coxmart,  8},
  {"coxscho",    (DL_FUNC) &coxscho, 8},
  {NULL, NULL, 0}
};

void R_init_survivalGPU(DllInfo *dll)
{
  R_registerRoutines(dll, Centries, Callentries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);

}
