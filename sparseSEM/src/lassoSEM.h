#ifndef LASSOSEM_H
#define LASSOSEM_H



void printMat(double *a, int M, int N);

void centerYX(double *Y,double *X, double *meanY, double *meanX,int M, int N);

double constrained_ridge_cff(double *Ycopy, double *Xcopy, double rho_factor, int M, int N,
                             double *B, double *f, double *mue, int verbose);

double lambdaMax(double *Y,double *X,double * W,int M, int N);

void QlambdaStart(double *Y,double *X, double *Q, double sigma2,int M, int N);

void QlambdaMiddle(double *Y,double *X, double *Q,double *B,double *f, double *mue, double sigma2,int M, int N);


void QlambdaMiddleCenter(double *Y,double *X, double *Q,double *B,double *f, double sigma2,int M, int N,
                         double *IBinv);

void UpdateIBinvPermute(double *QIBinv, double *B, int M);


void UpdateIBinv(double *QIBinv, double *B, int M);

double Weighted_LassoSf(double * W, double *B, double *f, double *Ycopy,double *Xcopy,
                        double *Q, double lambda_factor, double lambda_factor_prev, double sigma2, int max_iter,
                        int M, int N, int verbose,double *QIBinv,double lambda_max);			//double * mue,


double Weighted_LassoSf_MLf(double * W, double *BL, double *fL, double *Ycopy,double *Xcopy,
                            double *Q, double lambda_factor, double lambda_factor_prev, double sigma2, int max_iter,
                            int M, int N, int verbose,
                            double *BC, double *fC, double *mue,double *QIBinv,double *IBinvZero,double lambda_max);



#endif
