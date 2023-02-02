#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>


#include "Rinternals.h"
#include "survproto.h"
#include "survS.h"

static const R_CallMethodDef Callentries[] = {
  {"agmart3", (DL_FUNC) &agmart3, 8},
  {"coxcount1", (DL_FUNC) &coxcount1, 2},
  {"coxcount2", (DL_FUNC) &coxcount2, 4},
  {NULL, NULL, 0}
};

void R_init_survivalGPU(DllInfo *dll)
{
  R_registerRoutines(dll, NULL, Callentries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);

}
