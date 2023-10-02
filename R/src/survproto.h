/*
 ** Prototypes for functions from `survival` R package
 **  Including this in each routine helps prevent mismatched argument errors
 */
SEXP agmart3(SEXP nused2,  SEXP surv2,  SEXP score2, SEXP weight2,
             SEXP strata2, SEXP sort12, SEXP sort22, SEXP method2);

void chinv2  (double **matrix, int n);

             int cholesky2(double **matrix, int n, double toler);

             void chsolve2(double **matrix, int n, double *y);

             SEXP coxcount1(SEXP y2, SEXP strat2) ;

             SEXP coxcount2(SEXP y2, SEXP isort1, SEXP isort2, SEXP strat2) ;

             void coxmart(int   *sn,     int   *method,    double *time,
                          int   *status, int   * strata,   double *score,
                          double *wt,    double *expect);

             double **dmatrix(double *array, int nrow, int ncol);

             int    **imatrix(int *array, int nrow, int ncol);

             SEXP multicheck(SEXP time12,  SEXP time22, SEXP status2, SEXP id2,
                             SEXP istate2, SEXP sort2);

             SEXP tmerge (SEXP id2,  SEXP time1x, SEXP newx2,
                          SEXP nid2, SEXP ntime2, SEXP x2,  SEXP indx2);

             SEXP tmerge2(SEXP id2,  SEXP time1x, SEXP nid2, SEXP ntime2);

             SEXP tmerge3(SEXP id2, SEXP miss2);
