
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
#include <time.h>
#include <string.h>

#include "lassoSEM.h" 



void printMat(double *a, int M, int N) //MxN
{
	int i,j;
	printf("Printing the matrix\n\n");
	for(i=0;i<M;i++)
	{
		for(j=0;j<N;j++)
		{
			printf("%f\t", a[j*M +i]); //a[i,j]
		}
		printf("\n");
	}
}

void centerYX(double *Y,double *X, double *meanY, double *meanX,int M, int N) //M genes; N samples
{
	//matrix is vectorized by column: 1-M is the first column of Y
	//missing values are from X; set corresponding Y to zero.  in main_SMLX.m

	int i,index;
	double *Xptr;
	double *Yptr;

	int inci = 1;
	int incj = 1;
	int inc0 = 0;
	int lda  = M; //leading dimension
	double *eye;
	eye = (double* ) calloc(N, sizeof(double));
	double alpha = 1;
	double beta = 0;
	 dcopy(&N,&alpha,&inc0,eye,&inci);
	char transa = 'N';

	 dgemv(&transa, &M, &N,&alpha, X, &lda, eye, &inci, &beta,meanX, &incj);
	 dgemv(&transa, &M, &N,&alpha, Y, &lda, eye, &inci, &beta,meanY, &incj);
	double scale;
	scale = 1.0/N;
	 dscal(&M,&scale,meanY,&inci);
	 dscal(&M,&scale,meanX,&inci);
	// OUTPUT Y X, set missing values to zero
	scale = -1;
	for(i=0;i<N;i++)
	{
		index = i*M;
		Xptr = &X[index];
		Yptr = &Y[index];
		 daxpy(&M,&scale,meanY,&inci,Yptr,&incj);
		 daxpy(&M,&scale,meanX,&inci,Xptr,&incj);
	}
	free(eye);
}


//ridge regression;
double constrained_ridge_cff(double *Ycopy, double *Xcopy, double rho_factor, int M, int N,
		double *B, double *f, double *mue, int verbose)
{

	int i,j,k,lda,ldb,ldc,ldk;
	// center Y, X
	double *meanY, *meanX;
	meanY = (double* ) calloc(M, sizeof(double));
	meanX = (double* ) calloc(M, sizeof(double));

	//copy Y, X;
	double *Y, *X;
	int MN = M*N;
	Y = (double* ) calloc(MN, sizeof(double));
	X = (double* ) calloc(MN, sizeof(double));

	//F77_NAME(dcopy)(const int *n, const double *dx, const int *incx,
		//double *dy, const int *incy);
	int inci = 1;
	int incj = 1;
	 dcopy(&MN,Ycopy,&inci,Y,&incj);
	 dcopy(&MN,Xcopy,&inci,X,&incj);

	centerYX(Y,X,meanY, meanX,M, N);

//	double NPresent = 0;
//	for(i=0;i<M;i++)
//	{
//		for(j=0;j<N;j++)
//		{
//			if(Y[j*M + i]!=0) NPresent = NPresent + 1; //Y[i,j]
//		}
//	}
	if(verbose>7) printf("\t\t\t\t\t\t\t\tEnter Function: Ridge Regression. Shrinkage ratio rho is: %f.\n\n",rho_factor);

	int Mi = M -1;
	//for usage in loop
	double *YiPi; //Yi'*Pi
	YiPi =(double* ) calloc(Mi*N, sizeof(double));
	double xixi,xixiInv; //xi'*xi;
	int jj,index; //jj = 1:(M-1) index of YiPi
	double normYiPi,rho;
	double *bi,*YiPi2Norm; 	//YiPi2Norm: first term of biInv;

	double *Hi,*Yi,*xi,*yi,*xii;//xii for Hi calculation Hi= xi*xi'
	Hi = (double* ) calloc(N*N, sizeof(double));
	Yi =(double* ) calloc(Mi*N, sizeof(double));
	xi = (double* ) calloc(N, sizeof(double));
	xii = (double* ) calloc(N, sizeof(double));
	yi = (double* ) calloc(N, sizeof(double));
	double alpha, beta;
	char transa = 'N';
	char transb = 'N';

	//
	int MiMi = Mi*Mi;
	int NN = N*N;
	YiPi2Norm 	= (double* ) calloc(MiMi, sizeof(double));
	bi 			= (double* ) calloc(Mi, sizeof(double));
	//YiPiyi 		= (double* ) calloc(Mi, double);

	//bi,fi
	double *xiYi; //xi*Yi
	xiYi = (double* ) calloc(Mi, sizeof(double));
	double xiYibi, xiyi;
	//main loop:
	alpha = 1;
	beta = 0;

//largest Eigenvalue
	double *biInv;
	biInv 		= (double* ) calloc(MiMi, sizeof(double)); //copy of YiPi2Norm
	//dsyevd
	char jobz = 'N'; // yes for eigenvectors
	char uplo = 'U'; //both ok
	double *w, *work;
	w = (double *) calloc(Mi,sizeof(double));
	int lwork = 5*Mi + 10;
	work  = (double *) calloc(lwork,sizeof(double));
	int liwork = 10;
	int *iwork;
	iwork = (int *) calloc(liwork,sizeof(int));
	int info = 0;
	//dsyevd function

	//linear system
	int *ipiv;
	//ipiv = (int *) R_alloc(N,sizeof(int));
	ipiv = (int *) calloc(Mi,sizeof(int));
	double *readPtr,*readPtr2;
	//loop starts here
	for(i=0;i<M;i++)
	{
		//xi = X[i,:]
		readPtr = &X[i];
		 dcopy(&N,readPtr,&M,xi,&inci);
		 dcopy(&N,xi,&inci,xii,&incj);
		readPtr = &Y[i];
		 dcopy(&N,readPtr,&M,yi,&inci);

		//xixi =  dnrm2)(&N,xi,&inci);
		//xixi = pow(xixi,2);
		xixi =  ddot(&N, xi, &inci,xi, &incj);
		xixiInv = -1/xixi;
		//xi'*xi

		//YiPi
		//Hi          = xi*xi'/(xi'*xi);
        //Pi          = eye(N)-Hi;

		//MatrixMult(xi,xi, Hi,alpha, beta, N, k, N);
		transb = 'N';
		lda = N;
		ldb = N;
		ldc = N;
		 dgemm(&transa, &transb,&N, &ldb, &inci,&alpha, xi,&lda, xii, &incj, &beta,Hi, &ldc);


		//F77_NAME(dscal)(const int *n, const double *alpha, double *dx, const int *incx);
		//k= N*N;
		 dscal(&NN,&xixiInv,Hi,&inci); // Hi = -xi*xi'/(xi'*xi);
		for(j=0;j<N;j++)
		{	index = j*N + j;
			Hi[index] = Hi[index] + 1;
		}//Pi


		//Yi
		readPtr2 = &Yi[0];
		jj = 0;
		for(j=0;j<M;j++)
		{	if(j!=i)
			{
				//copy one j row
				readPtr = &Y[j];
				 dcopy(&N,readPtr,&M,readPtr2,&Mi);
				jj = jj + 1;
				readPtr2 = &Yi[jj];
			}
		}//jj 1:(M-1), YiPi[jj,:]
		//YiPi=Yi*Pi

		//transb = 'N';
		lda = Mi;
		ldb = N;
		ldc = Mi;
		ldk = N; //b copy
		 dgemm(&transa, &transb,&Mi, &N, &ldk,&alpha, Yi, &lda, Hi, &ldb, &beta, YiPi, &ldc);
		// dgemm)(&transa, &transb,&Mi, &N, &N,&alpha, Yi, &Mi, Hi, &N, &beta, YiPi, &Mi);


		//YiPi*Yi' --> MixMi
		transb = 'T';
		ldk = Mi;
		lda = Mi;
		ldb = Mi;
		ldc = Mi;
		 dgemm(&transa, &transb,&Mi, &ldk, &N,&alpha, YiPi, &lda, Yi, &ldb, &beta, YiPi2Norm, &ldc);
		// dgemm)(&transa, &transb,&Mi, &Mi, &N,&alpha, YiPi, &Mi, Yi, &Mi, &beta, YiPi2Norm, &Mi); //M-> N -> K

		//transa = 'N';
		transb = 'N';
		//j = Mi*Mi;
		 dcopy(&MiMi,YiPi2Norm,&inci,biInv,&incj);

		// dgeev)(&transa, &transb,&Mi, biInv, &Mi, wr, wi, vl, &ldvl,vr, &ldvr, work, &lwork, &info);
		lda = Mi;
		 dsyevd(&jobz, &uplo,&Mi, biInv, &lda, w, work, &lwork, iwork, &liwork,&info);
		normYiPi = w[Mi -1];

		rho = rho_factor*normYiPi; // 2Norm = sqrt(lambda_Max)

		if(verbose>8) printf("\t\t\t\t\t\t\t\t\t Gene number: %d,\t shrinkage rho: %f\n",i,rho);

		//
		for(j=0;j<Mi;j++)
		{
			index = j*Mi + j;
			YiPi2Norm[index] = YiPi2Norm[index] + rho;
		}
		//biInv;

		lda = Mi;
		 dgemv(&transa, &Mi, &N,&alpha, YiPi, &lda, yi, &inci, &beta,bi, &incj);

		lda = Mi;
		ldb = Mi;
		 dgesv(&Mi, &inci, YiPi2Norm, &lda, ipiv, bi, &ldb, &info);

		//------------------------------------------------Ridge coefficient beta obtained for row i
		// f(i)        = (xi'*yi-xi'*Yi*bi)/(xi'*xi);
		//xiYi (M-1) x1
		lda = Mi;

		 dgemv(&transa, &Mi, &N,&alpha, Yi, &lda, xi, &inci, &beta,xiYi, &incj);

		//xiyi = xi*yi 	= X[i,j]*Y[i,j]
		//dot product
		xiyi =  ddot(&N, xi, &inci,yi, &incj);

		//xiYibi = xiYi*bi
		xiYibi =  ddot(&Mi, xiYi, &inci,bi, &incj);

		f[i] = (xiyi-xiYibi)/xixi;

		//update B
		jj = 0;
		for(j = 0;j<M;j++)
		{
			if(j!=i)
			{
				//B[i,j] = bi[jj];
				B[j*M+i] = bi[jj];
				jj = jj +1;
			}
		}
	}//i = 1:M


	//I -B
	double *ImB;
	k = M*M;
	ImB = (double* ) calloc(k, sizeof(double));
	 dcopy(&k,B,&inci,ImB,&incj);
	//F77_NAME(dcopy)(const int *n, const double *dx, const int *incx,
	//	double *dy, const int *incy);
	xixiInv = -1;
	 dscal(&k,&xixiInv,ImB,&inci);
	for(i=0;i<M;i++)
	{
		index = i*M + i;
		ImB[index] = 1 + ImB[index];
	}




	//noise, sigma2learnt,mue;
	double * NOISE; 	//MxN
	NOISE =(double* ) calloc(MN, sizeof(double));
	transb = 'N';
	ldk = M;
	lda = M;
	ldb = M;
	ldc = M;
	 dgemm(&transa, &transb,&M, &N, &ldk,&alpha, ImB, &lda, Y, &ldb, &beta, NOISE, &ldc);//(I-B)*Y - fX
	for(i=0;i<M;i++)
	{
		// row i of X
		readPtr2 = &X[i];
		// dcopy)(&N,readPtr2,&M,xi,&inci);
		//row i of noise
		readPtr = &NOISE[i];
		alpha = -f[i];
		// daxpy)(&N, &alpha,xi, &inci,readPtr, &M);
		 daxpy(&N, &alpha,readPtr2, &ldk,readPtr, &M);
	}//row i = 1:M


	double noiseNorm, sigma2learnt;

	noiseNorm =  ddot(&MN, NOISE, &inci,NOISE, &incj);
	sigma2learnt = noiseNorm/(MN -1);

	//mue[i] = -f[i]*meanX[i];
	for(i=0;i<M;i++)
	{
		mue[i] = -f[i]*meanX[i];
	}
	beta = 1;
	ldk = M;
	lda = M;
	alpha = 1;
	 dgemv(&transa, &M, &ldk,&alpha, ImB, &lda, meanY, &inci, &beta,mue, &incj);


	if(verbose>7) printf("\t\t\t\t\t\t\t\tExit function: Ridge Regression. sigma^2 is: %f.\n\n",sigma2learnt);

	free(meanY);
	free(meanX);
	free(Y);
	free(X);
	free(YiPi);
	//free(biInv);
	free(YiPi2Norm);
	free(bi);
	//free(YiPiyi);
	free(xiYi);
	free(NOISE);
	//
	free(Hi);
	free(Yi);
	free(xi);
	free(yi);
	//
	free(ImB);

	//
	free(biInv);

	free(w);
	free(iwork);
	free(work);

	free(ipiv);
	return sigma2learnt;

}


//by Weighted_LassoSf xi

double lambdaMax(double *Y,double *X,double * W,int M, int N)
{
	// Oct 08, 2012: assume one eQTL for each gene; This fucntion needs significant revision if this assumption doesnot hold
	double *dxx, *rxy, *DxxRxy,*readPtr1,*readPtr2;
	double lambda_max = 0;
	dxx				= (double* ) calloc(M, sizeof(double));
	rxy				= (double* ) calloc(M, sizeof(double));
	DxxRxy			= (double* ) calloc(M, sizeof(double));
	int i,k,index,lda;
	int inci = 1;
	int incj = 1;
	lda = M;

	for(i=0;i<M;i++)
	{
		readPtr1 	= &X[i]; //ith row
		readPtr2 	= &Y[i];

		dxx[i] =  ddot(&N,readPtr1,&lda,readPtr1,&M);
		//res = ddot(n, x, incx, y, incy)
		rxy[i] 		=  ddot(&N,readPtr1,&lda,readPtr2,&M);
		DxxRxy[i] 	= rxy[i]/dxx[i];
	}

	//cache X[k,:]*DxxRxy[k]
	double * XDxxRxy;
	int MN = M*N;
	XDxxRxy = (double* ) calloc(MN, sizeof(double));

	 dcopy(&MN,X,&inci,XDxxRxy,&incj);
	double alpha;
	for(i=0;i<M;i++)
	{
		alpha  = -DxxRxy[i];
		readPtr1 = &XDxxRxy[i]; //ith row
		 dscal(&N,&alpha, readPtr1,&M);//	(n, a, x, incx)
	}

	// Y- XDxxRxy  			daxpy(n, a, x, incx, y, incy) y= ax + y
	alpha  = 1.0;
	// XDxxRxy <- alpha*Y + XDxxRxy
	 daxpy(&MN,&alpha,Y,&inci,XDxxRxy,&inci);
	double *YYXDR; //= Y*XDxxRxy'
	int MM = M*M;
	YYXDR = (double* ) calloc(MM, sizeof(double));

	double beta;
	char transa = 'N';
	char transb = 'T';
	alpha = -1;
	beta = 0;
	 dgemm(&transa, &transb,&M, &M, &N,&alpha, Y,&M, XDxxRxy, &M, &beta,YYXDR, &M); //M xK, K xN  --> MxN, N xM --> M <-M, N<-M, k<-N

	for(i=0;i<M;i++)
	{
		for(k=0;k<M;k++)
		{
			index  = k*M + i;
			if(i==k)
			{
				YYXDR[index] = 0;
			}else
			{
				YYXDR[index] = YYXDR[index]/W[index];
			}
		}
	}
	//BLAS_extern int    /* IDAMAX - return the index of the element with max abs value */
	index =  idamax(&MM,YYXDR,&inci);
	lambda_max = fabs(YYXDR[index-1]);

	free(dxx);
	free(rxy);
	free(DxxRxy);
	//free(XX);
	free(XDxxRxy);
	free(YYXDR);

	return lambda_max;
}

//Q[i,k] =	N*sigma2*IM - (Y*(Y'-X'*DxxRxy)))
void QlambdaStart(double *Y,double *X, double *Q, double sigma2,int M, int N)
{
	// Oct 08, 2012: assume one eQTL for each gene; This fucntion needs significant revision if this assumption doesnot hold
	double *dxx, *rxy, *DxxRxy,*readPtr1,*readPtr2;

	dxx				= (double* ) calloc(M, sizeof(double));
	rxy				= (double* ) calloc(M, sizeof(double));
	DxxRxy			= (double* ) calloc(M, sizeof(double));
	int i,index,ldk,lda,ldb,ldc;
	int inci = 1;
	int incj = 1;
	//double norm;
	lda = M;
	for(i=0;i<M;i++)
	{
		readPtr1 	= &X[i]; //ith row
		readPtr2 	= &Y[i];

		//norm  		=  dnrm2)(&N,readPtr1,&M);
		//dxx[i] 		= pow(norm,2);
		dxx[i] =  ddot(&N,readPtr1,&lda,readPtr1,&M);
		//res = ddot(n, x, incx, y, incy)
		rxy[i] 		=  ddot(&N,readPtr1,&lda,readPtr2,&M);
		DxxRxy[i] 	= rxy[i]/dxx[i];
	}
	//abs(N*sigma2*IM - (Y*(Y'-X'*DxxRxy)))./W ; W[i,i] = inf.
	double Nsigma2  = N*sigma2; 			// int * double --> double

	//cache X[k,:]*DxxRxy[k]
	double * XDxxRxy;
	int MN = M*N;
	XDxxRxy = (double* ) calloc(MN, sizeof(double));
	 dcopy(&MN,X,&inci,XDxxRxy,&incj);
	double alpha;
	for(i=0;i<M;i++)
	{
		alpha  = -DxxRxy[i];
		readPtr1 = &XDxxRxy[i]; //ith row
		 dscal(&N,&alpha, readPtr1,&M);//	(n, a, x, incx)
	}

	alpha  = 1.0;
	// XDxxRxy <- alpha*Y + XDxxRxy
	 daxpy(&MN,&alpha,Y,&inci,XDxxRxy,&incj);
	//double *YYXDR; //= Y*XDxxRxy' 		--> Q

	double beta;
	char transa = 'N';
	char transb = 'T';
	alpha = -1;
	beta = 0;
	// dgemm)(&transa, &transb,&M, &M, &N,&alpha, Y,&M, XDxxRxy, &M, &beta,Q, &M); //M xK, K xN  --> MxN, N xM --> M <-M, N<-M, k<-N

	ldb = M;
	ldc = M;
	ldk = M;
	 dgemm(&transa, &transb,&M, &lda, &N,&alpha, XDxxRxy,&ldb, Y, &ldc, &beta,Q, &ldk); //M xK, K xN  --> MxN, N xM --> M <-M, N<-M, k<-N

	for(i=0;i<M;i++)
	{
		index = i*M + i;
		Q[index]= Q[index] + Nsigma2;
	}

	free(dxx);
	free(rxy);
	free(DxxRxy);
	//free(XX);
	free(XDxxRxy);


}


void QlambdaMiddle(double *Y,double *X, double *Q,double *B,double *f, double *mue, double sigma2,int M, int N)
{
	// Oct 08, 2012: assume one eQTL for each gene; This fucntion needs significant revision if this assumption doesnot hold
	//I - B; copy of IB for inverse
	double *IB, *IBinv,*IBcopy;
	int MM = M*M;
	int MN = M*N;
	IB = (double* ) calloc(MM, sizeof(double));
	IBinv = (double* ) calloc(MM, sizeof(double));
	IBcopy = (double* ) calloc(MM, sizeof(double));
	int inci = 1;
	int incj = 1;
	 dcopy(&MM,B,&inci,IB,&incj);
	int i,index;
	double alpha;
	double beta = 0;
	alpha = -1;
	 dscal(&MM,&alpha,IB,&inci);
	alpha = 0;
	int inc0 = 0;
	// dscal)(&MM,&alpha,IBinv,&inci);//initialized
	 dcopy(&MM,&alpha,&inc0,IBinv,&inci);

	for(i=0;i<M;i++)
	{
		index = i*M + i;
		IB[index] = 1 + IB[index];
		IBinv[index] = 1;
	}
	 dcopy(&MM,IB,&inci,IBcopy,&incj);

	int info = 0;
	int *ipiv;
	ipiv = (int *) calloc(M,sizeof(int));
	int lda = M;
	int ldb = M;
	int ldc = M;
	int ldk = M;
	 dgesv(&M, &ldk, IBcopy, &lda, ipiv, IBinv, &ldb, &info);



	//abs(N*sigma2*inv(I-B) - NOISE*Y'.
	double Nsigma2  = N*sigma2; 			// int * double --> double
	double *Noise;
	Noise = (double* ) calloc(MN, sizeof(double));
	//(I-B)*Y-bsxfun(@times,f,X);
	char transa = 'N';
	char transb = 'N';
	alpha = 1;
	 dgemm(&transa, &transb,&M, &N, &ldk,&alpha, IB, &lda, Y, &ldb, &beta, Noise, &ldc);
	double *readPtr1, *readPtr2;
	for(i=0;i<M;i++)
	{
		readPtr1 = &X[i];
		readPtr2 = &Noise[i];
		alpha = -f[i]; // y= alpha x + y
		 daxpy(&N, &alpha,readPtr1, &lda,readPtr2, &M);
	}//row i = 1:M

	alpha = -1;
	for(i=0;i<N;i++)
	{
		readPtr1 = &Noise[i*M];
		 daxpy(&M, &alpha,mue, &inci,readPtr1, &incj);
	}

	//alpha = -1;
	transb = 'T';
	 dgemm(&transa, &transb,&M, &ldk, &N,&alpha, Noise, &lda, Y, &ldb, &beta, Q, &ldc);
	//eiB = ei-BiT			//daxpy(n, a, x, incx, y, incy) 		y := a*x + y
	alpha = Nsigma2;
	 daxpy(&MM, &alpha,IBinv, &inci,Q, &incj);

	free(IB);
	free(IBinv);
	free(IBcopy);
	free(Noise);
	free(ipiv);

}


void QlambdaMiddleCenter(double *Y,double *X, double *Q,double *B,double *f, double sigma2,int M, int N,
					double *IBinv)
{
	// Oct 08, 2012: assume one eQTL for each gene; This fucntion needs significant revision if this assumption doesnot hold
	//I - B; copy of IB for inverse
	double *IB; 	//, *IBinv,*IBcopy
	int MM = M*M;
	int MN = M*N;
	IB = (double* ) calloc(MM, sizeof(double));
	//IBinv = (double* ) calloc(MM, double);
	//IBcopy = (double* ) calloc(MM, double);
	int inci = 1;
	int incj = 1;
	//int inc0 = 0;
	 dcopy(&MM,B,&inci,IB,&incj);
	int i,index;
	double alpha;
	double beta = 0;
	alpha = -1;
	 dscal(&MM,&alpha,IB,&inci);
	//alpha = 0;
	// dscal)(&MM,&alpha,IBinv,&inci);//initialized
	// dcopy)(&MM,&alpha,&inc0,IBinv,&inci);
	for(i=0;i<M;i++)
	{
		index = i*M + i;
		IB[index] = 1 + IB[index];
		//IBinv[index] = 1;
	}
	// dcopy)(&MM,IB,&inci,IBcopy,&incj);


	int lda = M;
	int ldb = M;
	int ldc = M;
	int ldk = M;
	// dgesv)(&M, &ldk, IBcopy, &lda, ipiv, IBinv, &ldb, &info);

	double Nsigma2  = N*sigma2; 			// int * double --> double
	double *Noise;
	Noise = (double* ) calloc(MN, sizeof(double));
	char transa = 'N';
	char transb = 'N';
	alpha = 1;
	 dgemm(&transa, &transb,&M, &N, &ldk,&alpha, IB, &lda, Y, &ldb, &beta, Noise, &ldc);
	double *readPtr1, *readPtr2;
	for(i=0;i<M;i++)
	{
		readPtr1 = &X[i];
		readPtr2 = &Noise[i];
		alpha = -f[i]; // y= alpha x + y
		 daxpy(&N, &alpha,readPtr1, &lda,readPtr2, &M);
	}//row i = 1:M

	alpha = -1;
	transb = 'T';
	 dgemm(&transa, &transb,&M, &ldk, &N,&alpha, Noise, &lda, Y, &ldb, &beta, Q, &ldc);
	//eiB = ei-BiT			//daxpy(n, a, x, incx, y, incy) 		y := a*x + y
	alpha = Nsigma2;
	 daxpy(&MM, &alpha,IBinv, &inci,Q, &incj);

	free(IB);
	//free(IBinv);
	//free(IBcopy);
	free(Noise);
	//free(ipiv);

}


void UpdateIBinvPermute(double *QIBinv, double *B, int M)
{
	//I - B; copy of IB for inverse
	double *IB,*IBinv;	//, *IBinv,*IBcopy;
	int MM = M*M;
	int lda = M;
	int ldb = M;
	int ldk = M;
	IB = (double* ) calloc(MM, sizeof(double));
	IBinv = (double* ) calloc(MM, sizeof(double));
	int inci = 1;
	int incj = 1;
	int inc0 = 0;
	 dcopy(&MM,B,&inci,IB,&incj);
	int i,index;
	double alpha;
	//double beta = 0;
	alpha = -1;
	 dscal(&MM,&alpha,IB,&inci);
	alpha = 0;
	// dscal)(&MM,&alpha,IBinv,&inci);//initialized
	 dcopy(&MM,&alpha,&inc0,IBinv,&inci);
	for(i=0;i<M;i++)
	{
		index = i*M + i;
		IB[index] = 1 + IB[index];
		IBinv[index] = 1;
	}

	int info = 0;
	int *ipiv;
	ipiv = (int *) calloc(M,sizeof(int));
	 dgesv(&M, &ldk, IB, &lda, ipiv, IBinv, &ldb, &info);
	double *ptr1,*ptr2;


	for(i=0;i<M;i++)
	{
		index = ipiv[i] -1;
		ptr1 = &QIBinv[index*M];
		ptr2 = &IBinv[i*M];
		 dcopy(&M,ptr2,&inci,ptr1,&incj);

	}

	free(IB);
	free(ipiv);
	free(IBinv);
}


void UpdateIBinv(double *QIBinv, double *B, int M)
{
	//I - B; copy of IB for inverse
	double *IB;	//, *IBinv,*IBcopy;
	int MM = M*M;
	int lda = M;
	int ldb = M;
	int ldk = M;
	IB = (double* ) calloc(MM, sizeof(double));

	int inci = 1;
	int incj = 1;
	int inc0 = 0;
	 dcopy(&MM,B,&inci,IB,&incj);
	int i,index;
	double alpha;
	//double beta = 0;
	alpha = -1;
	 dscal(&MM,&alpha,IB,&inci);
	alpha = 0;
	// dscal)(&MM,&alpha,IBinv,&inci);//initialized
	 dcopy(&MM,&alpha,&inc0,QIBinv,&inci);
	for(i=0;i<M;i++)
	{
		index = i*M + i;
		IB[index] = 1 + IB[index];
		QIBinv[index] = 1;
	}

	int info = 0;
	int *ipiv;
	ipiv = (int *) calloc(M,sizeof(int));
	 dgesv(&M, &ldk, IB, &lda, ipiv, QIBinv, &ldb, &info);

	free(IB);
	free(ipiv);
}

//no Missing
					
double Weighted_LassoSf(double * W, double *B, double *f, double *Ycopy,double *Xcopy,
		double *Q, double lambda_factor, double lambda_factor_prev, double sigma2, int max_iter,
		int M, int N, int verbose,double *QIBinv,double lambda_max)			//double * mue,
{
	int i,j,index,ldM;
	//lda = M;
	//ldb = M;ldb,
	ldM = M;//fixed
	// return lambda;
	double *meanY, *meanX;
	meanY = (double* ) calloc(M, sizeof(double));
	meanX = (double* ) calloc(M, sizeof(double));

	//copy Y, X;
	double *Y, *X;
	int MN = M*N;
	int MM = M*M;
	Y = (double* ) calloc(MN, sizeof(double));
	X = (double* ) calloc(MN, sizeof(double));

	//F77_NAME(dcopy)(const int *n, const double *dx, const int *incx,
		//double *dy, const int *incy);
	int inci,incj, inc0;
	inci	= 1;
	incj 	= 1;
	inc0 	= 0;
	 dcopy(&MN,Ycopy,&inci,Y,&incj);
	 dcopy(&MN,Xcopy,&inci,X,&incj);
	centerYX(Y,X, meanY, meanX,M, N);

	//return value
	//double sigma2 			= SIGMA2[0];
	double lambda;//lambda_max,
	//lambdaMax
	if(verbose>4) printf("\t\t\t\tEnter Function: weighted_LassoSf. The maximum lambda is: %f\n\n",lambda_max);
	lambda 					= lambda_factor*lambda_max;

	//none zeros
	double alpha,beta;
	beta = 0;
	double deltaLambda;
	double *s, *S,*Wcopy;
	S = (double* ) calloc(MM, sizeof(double));
	s = (double* ) calloc(M, sizeof(double));
	Wcopy = (double* ) calloc(MM, sizeof(double));
	 dcopy(&MM,W,&inci,Wcopy,&incj);

	deltaLambda 			= (2*lambda_factor - lambda_factor_prev)*lambda_max;
	 dscal(&MM,&deltaLambda,Wcopy,&inci); //wcopy = deltaLambda*W

	//ei = 0
	double *ei,toyZero;
	toyZero= 0;
	ei = (double* ) calloc(M, sizeof(double));
	// dscal)(&M,&toyZero,ei,&inci);
	 dcopy(&M,&toyZero,&inc0,ei,&inci);
/*	double *eye;
	eye = (double* ) calloc(M, double);
	alpha = 1;
	 dcopy)(&M,&alpha,&inc0,eye,&inci);
*/
	double *readPtr,*readPtr2;
	for(i=0;i<M;i++)
	{
		for(j=0;j<M;j++)
		{
			//W[i,j]
			index = j*M  +i;
			if(fabs(Q[index])>= Wcopy[index] && i!= j)
			{
				S[index] 	= 1;
			}else
			{
				S[index] 	= 0;
				B[index] 	= 0;
			}
		}
		readPtr = &S[i]; //S[i,];
		s[i] =  dasum(&M, readPtr, &ldM);
	}
	char transa = 'N';
/*	ldk = M;
	//lda = M;
	 dgemv)(&transa, &M, &ldk,&alpha, S, &ldM, eye, &inci, &beta,s, &incj);
*/
	//f0, F1
	double *f0,*F1;
	// int qdif = M*M;
	f0 	= (double* ) calloc(M, sizeof(double));
	F1 	= (double* ) calloc(MM, sizeof(double));

	double *y_j;
	//xi 	= (double* ) calloc(N, double);
	y_j 	= (double* ) calloc(N, sizeof(double));
	double *F1ptr;


	double XYi, XXi;
	for(i=0;i<M;i++)
	{
		readPtr = &X[i];
		// dcopy)(&N,readPtr,&M,xi,&inci);
		readPtr2 = &Y[i];
		// dcopy)(&N,readPtr2,&M,y_j,&inci);

		//dot product
		//XYi =  ddot)(&N, xi, &inci,y_j, &incj);
		XYi =  ddot(&N, readPtr, &M,readPtr2, &M);

		XXi =  ddot(&N, readPtr, &M,readPtr, &M);
		f0[i] 	= XYi/XXi;
		F1ptr	= &F1[M*i];//start from ith column
		//Y*X(i,:)' 		y := alpha*A*x + beta*y,
		alpha = 1/XXi;
		 dgemv(&transa, &M, &N,&alpha, Y, &ldM, readPtr, &M, &beta,F1ptr, &incj);
	}

	// entering loop
	double *IBinv,*zi,*a_iT;		// y_j: one row of Y: Nx1
	IBinv 	= (double* ) calloc(MM, sizeof(double));
	//zi 		= (double* ) calloc(M, double);
	a_iT 	= (double* ) calloc(N, sizeof(double));



	//loop starts here
	int iter = 0;
	double js_i, m_ij,B_old, lambdaW,beta_ij,r_ij, Bij;
	//dynamic variable keep intermidiate values
	double *eiB;
	eiB = (double* ) calloc(M, sizeof(double));
	double *BiT;
	BiT = (double* ) calloc(M, sizeof(double));
	//quadratic function
	double d_ij, theta_ijp,k_ijp,q_ijp,Bijpp, Bijpm; //case (14)
	double q_ijm, theta_ijm, Bijmm, Bijmp,Lss,candsBij,LssCands;

	//converge of gene i
	double dB,ziDb,BF1;

	//converge of while
	double delta_BF,FnormOld, FnormChange;
	double *BfOld,*BfNew,*BfChange;
	index = M*(M  +1);
	BfOld = (double* ) calloc(index, sizeof(double));
	BfNew = (double* ) calloc(index, sizeof(double));
	BfChange = (double* ) calloc(index, sizeof(double));


	while(iter < max_iter)
	{
		iter = iter + 1;
		//converge Bfold = [B f];
		 dcopy(&MM,B,&inci,BfOld,&incj);
		//last column
		F1ptr = &BfOld[MM];
		 dcopy(&M,f,&inci,F1ptr,&incj);
		//
		for(i=0;i<M;i++)
		{
			if(s[i] >0)
			{ 	//
				if(verbose>6) printf("\t\t\t\t\t updating gene %d \n",i);
				//
				ei[i] = 1;
				//zi   IBinv = I -B
				zi = &QIBinv[i*M];

				for(j=0;j<M;j++)
				{
					js_i = S[j*M + i]; 		//ith row
					if(js_i >0)
					{

						m_ij 	= zi[j];
						B_old 	= B[j*M + i]; //B[i,j]
						if(j!=i)
						{

							//y_j; jth row Nx1
							readPtr = &Y[j];
							 dcopy(&N,readPtr,&M,y_j,&inci);
							//Y[j,:]

							lambdaW 	= lambda*W[j*M + i]; 	//W[i,j];
							//BiT = -B[i:]
							readPtr = &B[i];

							 dcopy(&M,readPtr,&ldM,BiT,&inci);
							alpha = -1;
							 dscal(&M,&alpha,BiT,&inci);
							BiT[j] = 0;
							//eiB
							 dcopy(&M,ei,&inci,eiB,&incj);
							//eiB 		//daxpy(n, a, x, incx, y, incy) 		y := a*x + y
							alpha = 1;
							 daxpy(&M, &alpha,BiT, &inci,eiB, &incj);
							//a_iT
							readPtr = &X[i];
							 dcopy(&N,readPtr,&M,a_iT,&inci);

							alpha = -f[i];
							 dscal(&N,&alpha,a_iT,&inci);

							transa='T'; //y := alpha*A**T*x + beta*y, 		 dgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
							beta = 1;
							alpha = 1;
							 dgemv(&transa, &M, &N,&alpha, Y, &ldM, eiB, &inci, &beta,a_iT, &incj);

							//r_ij
							r_ij =  ddot(&N, y_j, &inci,y_j, &incj);

							//beta_ij
							beta_ij =  ddot(&N, y_j, &inci,a_iT, &incj);

							if (fabs(m_ij)<1e-10) //go to the linear equation
							{
								//
								if(verbose>7) printf("\t\t\t\t\t\t\t gene %d \t interact with gene %d.\tLinear equation\n",i,j);
								//
								Bij = (beta_ij-lambdaW)/r_ij;

								if(Bij>0)
								{
									B[j*M+i] = Bij;//B(i,j)      = Bij;
								}else
								{
									Bij         = (beta_ij+lambdaW)/r_ij;
									if(Bij<0)
									{
										B[j*M+i] = Bij;//B(i,j)      = Bij;
									}else
									{
										B[j*M+i] = 0;
									}
								}//B_ij>0
							}else //m_ij ~=0 go to the quadratic equation
							{
								//
								if(verbose>7) printf("\t\t\t\t\t\t\t gene %d \t interact with gene %d.\tQuadratic equation\n",i,j);
								//
								//assume Bij >0
								d_ij = 1/m_ij + B[j*M+i];
								theta_ijp = r_ij*d_ij + beta_ij - lambdaW;
								k_ijp = d_ij*(beta_ij - lambdaW) - N*sigma2;

								q_ijp = theta_ijp*theta_ijp - 4*r_ij * k_ijp;
								Bijpp = (1/(2*r_ij))*(theta_ijp + sqrt(q_ijp));
								Bijpm = (1/(2*r_ij))*(theta_ijp - sqrt(q_ijp));

								//assume Bij<0
								q_ijm = q_ijp + 4*lambdaW *(beta_ij - r_ij *d_ij);
								theta_ijm = theta_ijp + 2*lambdaW;
								Bijmm = (1/(2*r_ij))*(theta_ijm - sqrt(q_ijm));
								Bijmp = (1/(2*r_ij))*(theta_ijm + sqrt(q_ijm));
								candsBij = 0;

								Lss = sigma2*N*log(fabs(d_ij)+1e-16);

								// a_iT daxpy(n, a, x, incx, y, incy) 	y := a*x + y
								if (Bijpp>0)
								{

									LssCands = sigma2*N*log(fabs(d_ij - Bijpp)+1e-16) - r_ij*pow(Bijpp,2)/2 + beta_ij*Bijpp -lambdaW*fabs(Bijpp);

									if(LssCands>Lss)
									{
										candsBij = Bijpp;
										Lss 	= LssCands;
									}
								}
								if (Bijpm>0)
								{

									LssCands = sigma2*N*log(fabs(d_ij - Bijpm)+1e-16) - r_ij*pow(Bijpm,2)/2 + beta_ij*Bijpm -lambdaW*fabs(Bijpm);
									if(LssCands>Lss)
									{
										candsBij = Bijpm;
										Lss 	= LssCands;
									}
								}
								//
								if (Bijmm<0)
								{

									LssCands = sigma2*N*log(fabs(d_ij - Bijmm)+1e-16) - r_ij*pow(Bijmm,2)/2 + beta_ij*Bijmm -lambdaW*fabs(Bijmm);
									if(LssCands>Lss)
									{
										candsBij = Bijmm;
										Lss 	= LssCands;
									}
								}
								if (Bijmp<0)
								{
									LssCands = sigma2*N*log(fabs(d_ij - Bijmp)+1e-16) - r_ij*pow(Bijmp,2)/2 + beta_ij*Bijmp -lambdaW*fabs(Bijmp);
									if(LssCands>Lss)
									{
										candsBij = Bijmp;
										Lss 	= LssCands;
									}
								}
								B[j*M+i] = candsBij;
							}//m_ij
						}//if(j!= i)
						dB = B_old - B[j*M +i];
						//update c_ij
						ziDb = 1/(1 + dB*m_ij);
						 dscal(&M,&ziDb,zi,&inci);

					}//js_i >0
				}//j = 1:M
				//f
				//BF1
				readPtr = &B[i];
				 dcopy(&M,readPtr,&ldM,BiT,&inci);

				F1ptr = &F1[M*i];
				BF1 =  ddot(&M, BiT, &inci,F1ptr, &incj);

				f[i] = f0[i] - BF1;
				ei[i] = 0; // re-set ei for next i
			}else//s[i]  no un-zero weight in this gene
			{
				readPtr = &B[i];
				// dscal)(&M,&toyZero,readPtr,&ldM);
				 dcopy(&M,&toyZero,&inc0,readPtr,&ldM);
				f[i] = f0[i];
			} // s[i]
		}//i= 1:M

		 dcopy(&MM,B,&inci,BfNew,&incj);
		F1ptr = &BfNew[MM];
		 dcopy(&M,f,&inci,F1ptr,&incj);
		//convergence
		index = (M+1)*M;			//daxpy(n, a, x, incx, y, incy) 	y := a*x + y
		alpha = -1;
		 dcopy(&index,BfOld,&inci,BfChange,&incj);
		 daxpy(&index, &alpha,BfNew, &inci,BfChange, &incj);

		FnormOld =  dnrm2(&index,BfOld,&inci);
		FnormChange =  dnrm2(&index,BfChange,&inci);
		//
		delta_BF = FnormChange/(FnormOld + 1e-10);

		UpdateIBinv(QIBinv, B,M);

		if(verbose>5) printf("\t\t\t\t\t\tdelta_BF: %f\n",delta_BF);
		if(delta_BF<1e-3)		//break out
		{
			break;
		}



	}//while


	if(verbose>4) printf("\t\t\t\tCurrent lambda: %f;\t number of iteration is: %d.\tExiting Weighted_LassoSf\n\n",lambda, iter);

	free(meanY);
	free(meanX);
	free(Y);
	free(X);

	free(S);
	free(s);
	free(f0);
	free(F1);
	free(Wcopy);

	//free(xi);
	free(y_j);

	free(ei);
	free(IBinv);
	//free(zi);
	free(a_iT);

	free(eiB);
	free(BiT);
	free(BfOld);
	free(BfNew);
	free(BfChange);

	//free(ipiv);

	//free(aiTQuad);
	//free(eye);
	return lambda;
	//sigma2 remains the same
}//weighted_LassoSf


double Weighted_LassoSf_MLf(double * W, double *BL, double *fL, double *Ycopy,double *Xcopy,
		double *Q, double lambda_factor, double lambda_factor_prev, double sigma2, int max_iter,
		int M, int N, int verbose,
		double *BC, double *fC, double *mue,double *QIBinv,double *IBinvZero,double lambda_max)
{

	double *B, *f;
	B = &BL[0];
	f = &fL[0];

	int i,j,index,ldk,ldM;
	//lda = M;
	//ldb = M;ldb,
	ldM = M;//fixed
	// return lambda;
	double *meanY, *meanX;
	meanY = (double* ) calloc(M, sizeof(double));
	meanX = (double* ) calloc(M, sizeof(double));

	//copy Y, X;
	double *Y, *X;
	int MN = M*N;
	int MM = M*M;
	Y = (double* ) calloc(MN, sizeof(double));
	X = (double* ) calloc(MN, sizeof(double));

	//F77_NAME(dcopy)(const int *n, const double *dx, const int *incx,
		//double *dy, const int *incy);
	int inci,incj,inc0;
	inci	= 1;
	incj 	= 1;
	inc0 	= 0;
	 dcopy(&MN,Ycopy,&inci,Y,&incj);
	 dcopy(&MN,Xcopy,&inci,X,&incj);
	centerYX(Y,X, meanY, meanX,M, N);

	//return value
	//double sigma2 			= SIGMA2[0];
	double lambda;//lambda_max,
	//lambdaMax
	if(verbose>4) printf("\t\t\t\tEnter Function: weighted_LassoSf. The maximum lambda is: %f\n\n",lambda_max);
	lambda 					= lambda_factor*lambda_max;

	//none zeros
	double alpha,beta;
	beta = 0;
	double deltaLambda;
	double *s, *S,*Wcopy;
	S = (double* ) calloc(MM, sizeof(double));
	s = (double* ) calloc(M, sizeof(double));
	Wcopy = (double* ) calloc(MM, sizeof(double));
	 dcopy(&MM,W,&inci,Wcopy,&incj);

	deltaLambda 			= (2*lambda_factor - lambda_factor_prev)*lambda_max;
	 dscal(&MM,&deltaLambda,Wcopy,&inci); //wcopy = deltaLambda*W

	//ei = 0
	double *ei,toyZero;
	toyZero= 0;
	ei = (double* ) calloc(M, sizeof(double));
	// dscal)(&M,&toyZero,ei,&inci);
	 dcopy(&M,&toyZero,&inc0,ei,&inci);

	double *readPtr,*readPtr2;
	for(i=0;i<M;i++)
	{
		for(j=0;j<M;j++)
		{
			//W[i,j]
			index = j*M  +i;
			if(fabs(Q[index])>= Wcopy[index] && i!= j)
			{
				S[index] 	= 1;
			}else
			{
				S[index] 	= 0;
				B[index] 	= 0;
			}
		}
		readPtr = &S[i]; //S[i,];
		s[i] =  dasum(&M, readPtr, &ldM);
	}
	char transa = 'N';


	//f0, F1
	double *f0,*F1;
	//int qdif = M*M;
	f0 	= (double* ) calloc(M, sizeof(double));
	F1 	= (double* ) calloc(MM, sizeof(double));

	//double *xi, *y_j;
	double *y_j;
	//xi 	= (double* ) calloc(N, double);
	y_j 	= (double* ) calloc(N, sizeof(double));
	double *F1ptr;


	double XYi, XXi;
	for(i=0;i<M;i++)
	{
		readPtr = &X[i];
		// dcopy)(&N,readPtr,&M,xi,&inci);
		readPtr2 = &Y[i];
		// dcopy)(&N,readPtr,&M,y_j,&inci);

		//dot product
		//XYi =  ddot)(&N, xi, &inci,y_j, &incj);
		XYi =  ddot(&N, readPtr, &M,readPtr2, &M);

		XXi =  ddot(&N, readPtr, &M,readPtr, &M);
		f0[i] 	= XYi/XXi;
		F1ptr	= &F1[M*i];//start from ith column
		//Y*X(i,:)' 		y := alpha*A*x + beta*y, alpha*Y *xi + beta*F1
		alpha = 1/XXi;
		 dgemv(&transa, &M, &N,&alpha, Y, &ldM, readPtr, &M, &beta,F1ptr, &incj);
	}

	// entering loop
	double *IBinv,*zi,*a_iT;// y_j: one row of Y: Nx1
	IBinv 	= (double* ) calloc(MM, sizeof(double));
	//zi 		= (double* ) calloc(M, double);
	a_iT 	= (double* ) calloc(N, sizeof(double));



	//loop starts here
	int iter = 0;
	double js_i, m_ij,B_old, lambdaW,beta_ij,r_ij, Bij;
	//dynamic variable keep intermidiate values
	double *eiB;
	eiB = (double* ) calloc(M, sizeof(double));
	double *BiT;
	BiT = (double* ) calloc(M, sizeof(double));
	//quadratic function
	double d_ij, theta_ijp,k_ijp,q_ijp,Bijpp, Bijpm; //case (14)
	double q_ijm, theta_ijm, Bijmm, Bijmp,Lss,candsBij,LssCands;

	//converge of gene i
	double dB,ziDb,BF1;

	//converge of while
	double delta_BF,FnormOld, FnormChange;
	double *BfOld,*BfNew,*BfChange;
	index = M*(M  +1);
	BfOld = (double* ) calloc(index, sizeof(double));
	BfNew = (double* ) calloc(index, sizeof(double));
	BfChange = (double* ) calloc(index, sizeof(double));

	while(iter < max_iter)
	{
		iter = iter + 1;
		//converge Bfold = [B f];
		 dcopy(&MM,B,&inci,BfOld,&incj);
		//last column
		F1ptr = &BfOld[MM];
		 dcopy(&M,f,&inci,F1ptr,&incj);

		for(i=0;i<M;i++)
		{
			if(s[i] >0)
			{ 	//
				if(verbose>6) printf("\t\t\t\t\t updating gene %d \n",i);
				//
				ei[i] = 1;

				zi = &QIBinv[i*M];

				for(j=0;j<M;j++)
				{
					js_i = S[j*M + i]; 		//ith row
					if(js_i >0)
					{

						m_ij 	= zi[j];
						B_old 	= B[j*M + i]; //B[i,j]
						if(j!=i)
						{

							//y_j; jth row Nx1
							readPtr = &Y[j];
							 dcopy(&N,readPtr,&M,y_j,&inci);
							//Y[j,:]

							lambdaW 	= lambda*W[j*M + i]; 	//W[i,j];
							//BiT = -B[i:]
							readPtr = &B[i];

							 dcopy(&M,readPtr,&ldM,BiT,&inci);
							alpha = -1;
							 dscal(&M,&alpha,BiT,&inci);
							BiT[j] = 0;
							//eiB
							 dcopy(&M,ei,&inci,eiB,&incj);

							alpha = 1;
							 daxpy(&M, &alpha,BiT, &inci,eiB, &incj);
							//a_iT
							readPtr = &X[i];
							 dcopy(&N,readPtr,&M,a_iT,&inci);

							//a_iT = -f[i]*xi 		dscal(n, a, x, incx) 		x = a*x
							alpha = -f[i];
							 dscal(&N,&alpha,a_iT,&inci);

							transa='T'; //y := alpha*A**T*x + beta*y, 		 dgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
							beta = 1;
							alpha = 1;
							 dgemv(&transa, &M, &N,&alpha, Y, &ldM, eiB, &inci, &beta,a_iT, &incj);

							//r_ij
							r_ij =  ddot(&N, y_j, &inci,y_j, &incj);

							//beta_ij
							beta_ij =  ddot(&N, y_j, &inci,a_iT, &incj);

							if (fabs(m_ij)<1e-10) //go to the linear equation
							{
								//
								if(verbose>7) printf("\t\t\t\t\t\t\t gene %d \t interact with gene %d.\tLinear equation\n",i,j);
								//
								Bij = (beta_ij-lambdaW)/r_ij;
								//printf("\t\t gene (%d,\t%d): linear Bij %f\n",i,j,Bij);

								if(Bij>0)
								{
									B[j*M+i] = Bij;//B(i,j)      = Bij;
								}else
								{
									Bij         = (beta_ij+lambdaW)/r_ij;
									if(Bij<0)
									{
										B[j*M+i] = Bij;//B(i,j)      = Bij;
									}else
									{
										B[j*M+i] = 0;
									}
								}//B_ij>0
							}else //m_ij ~=0 go to the quadratic equation
							{
								//
								if(verbose>7) printf("\t\t\t\t\t\t\t gene %d \t interact with gene %d.\tQuadratic equation\n",i,j);
								//
								//assume Bij >0
								d_ij = 1/m_ij + B[j*M+i];
								theta_ijp = r_ij*d_ij + beta_ij - lambdaW;
								k_ijp = d_ij*(beta_ij - lambdaW) - N*sigma2;

								q_ijp = theta_ijp*theta_ijp - 4*r_ij * k_ijp;
								Bijpp = (1/(2*r_ij))*(theta_ijp + sqrt(q_ijp));
								Bijpm = (1/(2*r_ij))*(theta_ijp - sqrt(q_ijp));

								//assume Bij<0
								q_ijm = q_ijp + 4*lambdaW *(beta_ij - r_ij *d_ij);
								theta_ijm = theta_ijp + 2*lambdaW;
								Bijmm = (1/(2*r_ij))*(theta_ijm - sqrt(q_ijm));
								Bijmp = (1/(2*r_ij))*(theta_ijm + sqrt(q_ijm));
								candsBij = 0;


								Lss = sigma2*N*log(fabs(d_ij)+1e-16);

								// a_iT 		daxpy(n, a, x, incx, y, incy) 	y := a*x + y
								if (Bijpp>0)
								{
									LssCands = sigma2*N*log(fabs(d_ij - Bijpp)+1e-16) - r_ij*pow(Bijpp,2)/2 + beta_ij*Bijpp -lambdaW*fabs(Bijpp);

									if(LssCands>Lss)
									{
										candsBij = Bijpp;
										Lss 	= LssCands;
									}
								}
								if (Bijpm>0)
								{
									LssCands = sigma2*N*log(fabs(d_ij - Bijpm)+1e-16) - r_ij*pow(Bijpm,2)/2 + beta_ij*Bijpm -lambdaW*fabs(Bijpm);
									if(LssCands>Lss)
									{
										candsBij = Bijpm;
										Lss 	= LssCands;
									}
								}
								//
								if (Bijmm<0)
								{
									LssCands = sigma2*N*log(fabs(d_ij - Bijmm)+1e-16) - r_ij*pow(Bijmm,2)/2 + beta_ij*Bijmm -lambdaW*fabs(Bijmm);
									if(LssCands>Lss)
									{
										candsBij = Bijmm;
										Lss 	= LssCands;
									}
								}
								if (Bijmp<0)
								{
									LssCands = sigma2*N*log(fabs(d_ij - Bijmp)+1e-16) - r_ij*pow(Bijmp,2)/2 + beta_ij*Bijmp -lambdaW*fabs(Bijmp);
									if(LssCands>Lss)
									{
										candsBij = Bijmp;
										Lss 	= LssCands;
									}
								}
								B[j*M+i] = candsBij;
							}//m_ij
						}//if(j!= i)
						dB = B_old - B[j*M +i];
						//update c_ij
						ziDb = 1/(1 + dB*m_ij);
						 dscal(&M,&ziDb,zi,&inci);

					}//js_i >0
				}//j = 1:M
				//f
				//BF1 = B(i,:)*F1(:,i)
				readPtr = &B[i];
				 dcopy(&M,readPtr,&ldM,BiT,&inci);

				F1ptr = &F1[M*i];
				BF1 =  ddot(&M, BiT, &inci,F1ptr, &incj);

				f[i] = f0[i] - BF1;
				ei[i] = 0;
			}else//s[i]  no un-zero weight in this gene
			{
				readPtr = &B[i];
				// dscal)(&M,&toyZero,readPtr,&ldM);
				 dcopy(&M,&toyZero,&inc0,readPtr,&ldM);
				f[i] = f0[i];
			} // s[i]
		}//i= 1:M

		 dcopy(&MM,B,&inci,BfNew,&incj);
		F1ptr = &BfNew[MM];
		 dcopy(&M,f,&inci,F1ptr,&incj);
		//convergence
		index = (M+1)*M;			//daxpy(n, a, x, incx, y, incy) 	y := a*x + y
		alpha = -1;
		 dcopy(&index,BfOld,&inci,BfChange,&incj);
		 daxpy(&index, &alpha,BfNew, &inci,BfChange, &incj);

		FnormOld =  dnrm2(&index,BfOld,&inci);
		FnormChange =  dnrm2(&index,BfChange,&inci);
		//
		delta_BF = FnormChange/(FnormOld + 1e-10);

		UpdateIBinv(QIBinv, B,M);

		if(verbose>5) printf("\t\t\t\t\t\tdelta_BF: %f\n",delta_BF);
		if(delta_BF<1e-3)		//break out
		{
			break;
		}



	}//while

	if(verbose>3) printf("\t\t\t\tCurrent lambda: %f;\t number of iteration is: %d.\tExiting Weighted_LassoSf\n\n",lambda, iter);

	//----------------------------------------------------------------------------------END OF LASSO
	//double *B, double *f;
	 dcopy(&MM,BL,&inci,BC,&incj);
	 dcopy(&M,fL,&inci,fC,&incj);
	B = &BC[0];
	f = &fC[0];

	if(verbose>4) printf("Enter Function: constrained-MLf. Shrinkage lambda is: 0. \n");
	// SET SL
	for(i=0;i<MM;i++)
	{
		if(BL[i]==0)
		{
			S[i] = 0;
		}else
		{
			S[i] = 1;
		}
	}
	//none zeros
	//double *s;
	//s = (double* ) calloc(M, double);
	for(i=0;i<M;i++)
	{
		readPtr = &S[i]; //S[i,];
		s[i] =  dasum(&M, readPtr, &ldM);
	}

	beta = 1;
	//f0, F1
	//ei = 0
	// entering loop
	//loop starts here
	iter = 0;

	//quadratic function
	double theta_ij,k_ij,q_ij,Bijp, Bijm; //case (14)

	//converge of gene i

	//converge of while
	max_iter = max_iter/5;
	while(iter < max_iter)
	{
		iter = iter + 1;
		//converge Bfold = [B f];
		 dcopy(&MM,B,&inci,BfOld,&incj);
		//last column
		F1ptr = &BfOld[MM];
		 dcopy(&M,f,&inci,F1ptr,&incj);
		//
		// inner loop
		for(i=0;i<M;i++)
		{
			if(s[i] >0)
			{
				//
				if(verbose>6) printf("\t\t updating gene %d \n",i);
				//
			ei[i] = 1;

				zi = &IBinvZero[i*M];
				//for j = js_i
				for(j=0;j<M;j++)
				{
					js_i = S[j*M + i]; 		//ith row
					if(js_i >0)
					{
						//if(verbose>4) printf("\t\t\t gene %d \t interact with gene %d\n",i,j);

						m_ij 	= zi[j];
						B_old 	= B[j*M + i]; //B[i,j]

						//y_j
						readPtr = &Y[j];
						 dcopy(&N,readPtr,&M,y_j,&inci);
						//BiT = -B[i:]
						readPtr = &B[i];
						 dcopy(&M,readPtr,&ldM,BiT,&inci);
						alpha = -1;
						 dscal(&M,&alpha,BiT,&inci);
						BiT[j] = 0;
						 dcopy(&M,ei,&inci,eiB,&incj);
						//a_iT
						//eiB		//daxpy(n, a, x, incx, y, incy) 		y := a*x + y
						alpha = 1;
						 daxpy(&M, &alpha,BiT, &inci,eiB, &incj);
						readPtr = &X[i];
						 dcopy(&N,readPtr,&M,a_iT,&inci);

						alpha = -f[i];
						 dscal(&N,&alpha,a_iT,&inci);

						transa='T'; //y := alpha*A**T*x + beta*y, 		 dgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
						//beta = 1;
						alpha = 1;
						 dgemv(&transa, &M, &N,&alpha, Y, &ldM, eiB, &inci, &beta,a_iT, &incj);

						//r_ij
						r_ij =  ddot(&N, y_j, &inci,y_j, &inci);

						beta_ij =  ddot(&N, y_j, &inci,a_iT, &incj);

						if (fabs(m_ij)<1e-10) //go to the linear equation
						{
							//
							if(verbose>7) printf("\t\t\t gene %d \t interact with gene %d.\tLinear equation\n",i,j);
							//

							B[j*M+i] = beta_ij/r_ij;

						}else //m_ij ~=0 go to the quadratic equation
						{
							//
							if(verbose>7) printf("\t\t\t gene %d \t interact with gene %d.\tQuadratic equation\n",i,j);
							//

							//assume Bij >0
							d_ij = 1/m_ij + B[j*M+i];
							theta_ij = r_ij*d_ij + beta_ij;
							k_ij = d_ij*beta_ij - N*sigma2;

							q_ij = theta_ij*theta_ij - 4*r_ij* k_ij;
							Bijp = (1/(2*r_ij))*(theta_ij + sqrt(q_ij));
							Bijm = (1/(2*r_ij))*(theta_ij - sqrt(q_ij));

							candsBij = 0;

							Lss = sigma2*N*log(fabs(d_ij)+1e-16);
							//Bijp

							LssCands = sigma2*N*log(fabs(d_ij - Bijp)+1e-16) - r_ij*pow(Bijp,2)/2 + beta_ij*Bijp;
							if(LssCands>Lss)
							{
								candsBij = Bijp;
								Lss 	= LssCands;
							}
							//Bijm>

							LssCands = sigma2*N*log(fabs(d_ij - Bijm)+1e-16) - r_ij*pow(Bijm,2)/2 + beta_ij*Bijm;
							if(LssCands>Lss)
							{
								candsBij = Bijm;
								Lss 	= LssCands;
							}

							B[j*M+i] = candsBij;
						}//m_ij

						dB = B_old - B[j*M +i];
						//update c_ij
						ziDb = 1/(1 + dB*m_ij);
						 dscal(&M,&ziDb,zi,&inci);

					}//js_i >0
				}//j = 1:M

				//f
				//BF1
				readPtr = &B[i];
				 dcopy(&M,readPtr,&ldM,BiT,&inci);
				F1ptr = &F1[M*i];
				BF1 =  ddot(&M, BiT, &inci,F1ptr, &incj);

				f[i] = f0[i] - BF1;
				ei[i] = 0; // re-set ei for next i

			}//[si]
		}//i= 1:M

		//convergence
		 dcopy(&MM,B,&inci,BfNew,&incj);
		F1ptr = &BfNew[MM];
		 dcopy(&M,f,&inci,F1ptr,&incj);
		index = (M+1)*M;			//daxpy(n, a, x, incx, y, incy) 	y := a*x + y
		alpha = -1;
		 dcopy(&index,BfOld,&inci,BfChange,&incj);
		 daxpy(&index, &alpha,BfNew, &inci,BfChange, &incj);

		FnormOld =  dnrm2(&index,BfOld,&inci);
		FnormChange =  dnrm2(&index,BfChange,&inci);

		delta_BF = FnormChange/(FnormOld + 1e-10);
		if(verbose>5) printf("\t\tdelta_BF: %f\n",delta_BF);

		UpdateIBinv(IBinvZero, B,M);

		if(delta_BF<1e-2)		//break out
		{
			break;
		}

	}//while


	if(verbose>3) printf("\t number of iteration is: %d.\nExiting constrained_MLf\n",iter);

	//IBinv = I -B
	 dcopy(&MM,B,&inci,IBinv,&incj);
	alpha = -1;
	 dscal(&MM,&alpha,IBinv,&inci); // dscal(n, a, x, incx) x = a*x
	//diagonal + 1
	for(j=0;j<M;j++)
	{
		index = j*M + j;
		IBinv[index] = 1 + IBinv[index];
		mue[j] = -f[j]*meanX[j];
	}
	//mue[i] = -f[i]*meanX[i];
	//	//dgemv mue = Ax + beta*mue
	transa = 'N';
	alpha = 1;
	//beta = 1;
	ldk = M;
	 dgemv(&transa, &M, &ldk,&alpha, IBinv, &ldM, meanY, &inci, &beta,mue, &incj);


	//----------------------------------------------------------------------------------END OF LASSO_ZERO

	free(meanY);
	free(meanX);
	free(Y);
	free(X);

	free(S);
	free(s);
	free(f0);
	free(F1);
	free(Wcopy);

	//free(xi);
	free(y_j);

	free(ei);
	free(IBinv);
	//free(zzi);
	free(a_iT);

	free(eiB);
	free(BiT);
	free(BfOld);
	free(BfNew);
	free(BfChange);

	//free(ipiv);

	//free(aiTQuad);
	//free(eye);
	return lambda;
	//sigma2 remains the same
}//weighted_LassoSf








