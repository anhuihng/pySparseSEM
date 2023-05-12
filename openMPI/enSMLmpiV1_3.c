//version info: 
//				version 13 based on ver12: add parallel for function: 	Weighted_LassoSf_MLf
//				version 13beta: add parallel for function: 				 Weighted_lassoSf
//				version 14: based on version 13beta: for ridge regression: calculate lambda-->normYiPi once for all.
//				version 14beta: SVD for YiPi, i = 1, 2, ..., M; for all Kcv folds
//								solve SVD-ridge for every rho. 
//				version 14gamma: lambda in 1ste
// 				version 15: (tentative) add output; print to log 
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
#include <time.h>
#include <mpi.h>
#include <string.h>

#define BLOCK_LOW(id,p,n) ((id)*(n)/(p))  		//The Sieve of Eratoshenes
#define BLOCK_HIGH(id,p,n) (BLOCK_LOW((id)+1,p,n)-1)
#define BLOCK_SIZE(id,p,n) (BLOCK_HIGH(id,p,n) - BLOCK_LOW(id,p,n) +1)
#define BLOCK_OWNER(index,p,n) (((p)*((index)+1)-1)/(n))
#define MIN(a,b) ((a)<(b)?(a):(b))

//version Note: block coordinate ascent: after each block updated: update IBinv before next iteration
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


//version 12 added function transpose
void transposeB(double *B, int M, int N) //MxN input
{
	int MN = M*N;
	double *tB,*readPtr1,*readPtr2;
	tB 	= (double* ) calloc(MN, sizeof(double)); 
	
	int i,inci,incj;
	inci = 1;
	incj = N;
	for(i=0;i<N;i++)
	{
		readPtr1 = &tB[i];
		readPtr2 = &B[i*M];
		dcopy(&M,readPtr2,&inci,readPtr1,&incj);
	}
	
	dcopy(&MN,tB,&inci,B,&inci);
	free(tB);
}
//
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

//--------------------- LINEAR SYSTEM SOLVER END ---------

//VERSION 14beta

//ridge regression; return sigma2learnt
void constrained_ridge_cffSVD(double *Ycopy, double *Xcopy, double *rho_factors, int M, int N,
		int verbose,int rank, int size, 
		double *Ytest, double *Xtest, double *Errs, int cv,int Nrho,int Ntest,double *mue)
{
	
	int i,j,k,lda,ldb,ldc,ldk;
	double rho_factor;
	// center Y, X
	double *meanY, *meanX;
if(rank==0)
{	
	meanY = (double* ) calloc(M, sizeof(double));
	meanX = (double* ) calloc(M, sizeof(double));
}	
	//copy Y, X; 
	double *Y, *X;
	int MN = M*N;
	int MM = M*M;
	Y = (double* ) calloc(MN, sizeof(double));
	X = (double* ) calloc(MN, sizeof(double));
	
	//F77_NAME(dcopy)(const int *n, const double *dx, const int *incx,
		//double *dy, const int *incy);
	int inci = 1;
	int incj = 1;

if(rank==0)
{	
	 dcopy(&MN,Ycopy,&inci,Y,&incj);
	 dcopy(&MN,Xcopy,&inci,X,&incj);
	
	centerYX(Y,X,meanY, meanX,M, N);
}
	//---------------------------MPI::
	//MPI::Bcast
	MPI_Bcast (Y, MN, MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast (X, MN, MPI_DOUBLE,0,MPI_COMM_WORLD);

//version 14 beta: 
	double *Brho,*fRho,*B, *f; 		//B, f are ptrs to Brho, fRho
	int rhoMM 	= Nrho*MM;
	int rhoM 	= Nrho*M;
if(rank==0)
{
	Brho = (double* ) calloc(rhoMM, sizeof(double));
	fRho = (double* ) calloc(rhoM, sizeof(double));
}	

	// MPI message vector for for loop
	double *message,*messageRho; //message is ptr to messageRho

	//MPI block decomposition
	int low,high,count;
	low = BLOCK_LOW(rank,size,M);
	high = BLOCK_HIGH(rank,size,M);
	count = BLOCK_SIZE(rank,size,M); 	//number of rows to compute
	count = count * M; 					// number of elements to communicate

	//MPI loop
	int *receive_counts;
    int *receive_displacements;
	receive_counts = (int *) calloc(size,sizeof(int));
	receive_displacements = (int *) calloc(size,sizeof(int));
	//number of elements for each process	
	MPI_Gather(&count, 1, MPI_INT, receive_counts,1, MPI_INT, 0, MPI_COMM_WORLD);
	receive_displacements[0] = 0;
	for(i=1;i<size;i++)
	{	
		receive_displacements[i] = receive_counts[i-1] + receive_displacements[i-1];
	}
	
	messageRho = (double* ) calloc(count*Nrho, sizeof(double));
	
	//--------------------------mpi::	
	
if(rank==0)
{
	if(verbose>6) printf("\t\t\t\t\t\tEnter Function: Ridge Regression-SVD. cv: %d; Nrho: %d\n",cv, Nrho);
}	
	int Mi = M -1;
	//for usage in loop
	double *YiPi; //Yi'*Pi
	YiPi =(double* ) calloc(Mi*N, sizeof(double));
	double xixi,xixiInv; //xi'*xi;
	int jj,index; //jj = 1:(M-1) index of YiPi
	//double normYiPi,rho;
	double rho;
	double *bi; 	//YiPi2Norm: first term of biInv;,*YiPi2Norm
	
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
	//YiPi2Norm 	= (double* ) calloc(MiMi, sizeof(double));
	bi 			= (double* ) calloc(Mi, sizeof(double));
	//YiPiyi 		= (double* ) calloc(Mi, double);
	
	//bi,fi
	double *xiYi; //xi*Yi
	xiYi = (double* ) calloc(Mi, sizeof(double));
	double xiYibi, xiyi;
	//main loop:
//printf("check point 1: before loop\n");
	alpha = 1;
	beta = 0;
		
	//dgesdd
	char jobz 	= 'S'; // only lda=min(M,N)-->k calculated in SVD
	double *S,*U,*VT,*Ucopy;
	int ldSVD 	= MIN(Mi,N);//-->k
	int SVDmat 	= ldSVD*Mi;
	S 			= (double *) calloc(ldSVD,sizeof(double));
	U 			= (double *) calloc(SVDmat,sizeof(double)); //Mi x k
	Ucopy 		= (double *) calloc(SVDmat,sizeof(double)); 
	VT 			= (double *) calloc(ldSVD*N,sizeof(double)); //k xN 
	double *work;
	int lwork 	= 10*ldSVD + 5*ldSVD*ldSVD + ldSVD;
	work  		= (double *) calloc(lwork,sizeof(double));	
	int liwork 	= 8*ldSVD;
	int *iwork;
	iwork 		= (int *) calloc(liwork,sizeof(int));
	int info 	= 0;
	//dgesdd
	double *scale;
	scale 		= (double *) calloc(ldSVD,sizeof(double));
	
	int iRho;
	
	double *readPtr,*readPtr2;
	//loop starts here	
	
	//for(i=0;i<M;i++)
	int iter;
	iter = 0;

	for(i=low;i<=high;i++)
	{
		//xi = X[i,:]
		readPtr = &X[i];
		 dcopy(&N,readPtr,&M,xi,&inci);
		 dcopy(&N,xi,&inci,xii,&incj);
		readPtr = &Y[i];
		 dcopy(&N,readPtr,&M,yi,&inci);

		//xixi =  dnrm2)(&N,xi,&inci);
		xixi =  ddot(&N, xi, &inci,xi, &incj);
		xixiInv = -1/xixi;
//printMat(xi,1,N);		
		//xi'*xi
		
		//YiPi
		//Hi          = xi*xi'/(xi'*xi);
        //Pi          = eye(N)-Hi;
		transb = 'N';
		lda = N;
		ldb = N;
		ldc = N;
		 dgemm(&transa, &transb,&N, &ldb, &inci,&alpha, xi,&lda, xii, &incj, &beta,Hi, &ldc);
		
		
		//F77_NAME(dscal)(const int *n, const double *alpha, double *dx, const int *incx);
		 dscal(&NN,&xixiInv,Hi,&inci); // Hi = -xi*xi'/(xi'*xi);
		for(j=0;j<N;j++) 
		{	
			index = j*N + j;
			Hi[index] = Hi[index] + 1;
		}//Pi
//printMat(Hi,N,N);	
		
		//Yi
		readPtr2 = &Yi[0];
		jj = 0;
		for(j=0;j<M;j++)
		{	
			if(j!=i)
			{
				//copy one j row
				readPtr = &Y[j];
				 dcopy(&N,readPtr,&M,readPtr2,&Mi);
				jj = jj + 1;
				readPtr2 = &Yi[jj];
			}
		}//jj 1:(M-1), YiPi[jj,:]
		//YiPi=Yi*Pi
//printMat(Yi,Mi,N);		
		transb = 'N';
		lda = Mi;
		ldb = N;
		ldc = Mi;
		ldk = N; //b copy
		 dgemm(&transa, &transb,&Mi, &N, &ldk,&alpha, Yi, &lda, Hi, &ldb, &beta, YiPi, &ldc);
		// dgemm)(&transa, &transb,&Mi, &N, &N,&alpha, Yi, &Mi, Hi, &N, &beta, YiPi, &Mi);
		

		//printf("check point 3: YiPi\n");
		//svd of YiPi:
		ldb = Mi;
		dgesdd(&jobz,&Mi,&N,YiPi,&lda,S,U,&ldb,VT,&ldSVD,work,&lwork,iwork,&info);
		// void dgesdd( char* jobz, MKL_INT* m, MKL_INT* n, double* a, MKL_INT* lda, double* s, double* u, MKL_INT* ldu, double* vt, MKL_INT* ldvt, double* work, MKL_INT* lwork, MKL_INT* iwork, MKL_INT* info );
		//dgesdd output: option S:
		//			U: M x lsSVD
		//			S: singular values 1-ldSVD; descend order
		//			VT: ldSVD x N 			---> return Vtranspose
		// 			YiPi: destroyee
		

		// f(i)        = (xi'*yi-xi'*Yi*bi)/(xi'*xi);
		//xiYi (M-1) x1
		lda = Mi;		
		 dgemv(&transa, &Mi, &N,&alpha, Yi, &lda, xi, &inci, &beta,xiYi, &incj);
		
		//xiyi = xi*yi 	= X[i,j]*Y[i,j]
		//dot product 
		xiyi =  ddot(&N, xi, &inci,yi, &incj);	
			
			
		for(iRho=0;iRho<Nrho;iRho++)
		{
			dcopy(&SVDmat,U,&inci,Ucopy,&incj);  //dcopy(n, x, incx, y, incy) ---> y = x

			message = &messageRho[iRho*count]; 
			rho_factor = rho_factors[iRho];
			rho = rho_factor*S[0]*S[0];
			for(j=0;j<ldSVD;j++)
			{
				scale[j] = S[j]/(S[j]*S[j] + rho);
				readPtr = &Ucopy[j*Mi];
				alpha = scale[j];
				dscal(&Mi,&alpha,readPtr,&inci); // Hi = -xi*xi'/(xi'*xi);
			}
			//U*diag(S)
			//YiPi <-- U*sigma*VT 		Mi x N			

			transb = 'N';// Mi x k  x k x N
			lda = Mi;
			ldb = ldSVD;
			ldc = Mi;
			alpha = 1;
			dgemm(&transa, &transb,&Mi, &N, &ldSVD,&alpha, Ucopy, &lda, VT, &ldb, &beta, YiPi, &ldc);
			
			lda = Mi;
			 dgemv(&transa, &Mi, &N,&alpha, YiPi, &lda, yi, &inci, &beta,bi, &incj);
			 
			 //obtained bi

if(rank==0)
{			
			if(verbose>7) printf("\t\t\t\t\t\t\t\t cv: %d \t Gene number: %d,\t rho: %d/%d(Nrho); shrinkage rho: %f\n",cv,i,iRho,Nrho,rho);
}
			
			//printf("check point 6: bi updated\n");
			//--------------------------------------Ridge coefficient beta obtained for row i
			
			//xiYibi = xiYi*bi
			xiYibi =  ddot(&Mi, xiYi, &inci,bi, &incj);
			//		f[i] = (xiyi-xiYibi)/xixi;
			
			//MPI::MESSAGE
			//message[iter*M + i] = (xiyi-xiYibi)/xixi;
			message[iter*M+ i] = (xiyi-xiYibi)/xixi;
			//printf("check point 7: fi calculated\n");
			//update B
			jj = 0;
			for(j = 0;j<M;j++)
			{
				if(j!=i)
				{
					//B[i,j] = bi[jj];
					message[iter*M + j] = bi[jj];		//message[j,iter]; message is Mx count; //column wise
					jj = jj +1;
				}
			}
		}//iRho = 1: Nrho						
		iter = iter  +1;	
	}//i = 1:M	
	
	//gather
	for(i= 0;i<Nrho;i++)
	{
if(rank==0)
{	
		B = &Brho[i*MM];
}
		message = &messageRho[i*count];	
		//MPI::Gather	
		MPI_Gatherv(message, count, MPI_DOUBLE, B,receive_counts,receive_displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);		
	}

	double noiseNorm;
	double *ImB;
	double * NOISE; 	//MxN
	int MNtest = M*Ntest;
	//Error calculation
if(rank==0)
{		
	k = M*M;
	ImB = (double* ) calloc(MM, sizeof(double));
	NOISE =(double* ) calloc(MNtest, sizeof(double));
	
//printf("MNtest is : %d\n",MNtest);	
	for(iRho= 0;iRho<Nrho;iRho++)
	{
		B = &Brho[iRho*MM];
		f = &fRho[iRho*M];
		transposeB(B,M,M);

		//MPI::RANK0 f, B
		for(i=0;i<M;i++)
		{
			f[i] = B[i*M + i];
			B[i*M + i] = 0;
		}

		//I -B
		 dcopy(&MM,B,&inci,ImB,&incj);
		//F77_NAME(dcopy)(const int *n, const double *dx, const int *incx,
		//	double *dy, const int *incy);
		xixiInv = -1;
		 dscal(&k,&xixiInv,ImB,&inci);
		for(i=0;i<M;i++) 
		{
			index = i*M + i;
			ImB[index] = 1 + ImB[index];
		}
		
		//NOISE = (I-B)*Ytest
		transb = 'N';
		ldk = M;
		lda = M;
		ldb = M;
		ldc = M;
		alpha = 1;
		beta  = 0;
		 dgemm(&transa, &transb,&M, &Ntest, &ldk,&alpha, ImB, &lda, Ytest, &ldb, &beta, NOISE, &ldc);//(I-B)*Y - fX
		
		//NOISE = (I-B)*Ytest - fXtest
		for(i=0;i<M;i++)
		{
			// row i of X
			readPtr2 = &Xtest[i];
			// dcopy)(&N,readPtr2,&M,xi,&inci);
			//row i of noise
			readPtr = &NOISE[i];
			alpha = -f[i];
			// daxpy)(&N, &alpha,xi, &inci,readPtr, &M);
			 daxpy(&Ntest, &alpha,readPtr2, &ldk,readPtr, &M);
			//NOISE[i,1:N]	
		}//row i = 1:M		

		//mue          = (IM-B)*meanY-bsxfun(@times,f,meanX);
		//dgemv mue = Ax + beta*mue	
	
		//mue[i] = -f[i]*meanX[i];
		for(i=0;i<M;i++)
		{
			mue[i] = -f[i]*meanX[i];
		}
		
		//mue = (I-B)*Ymean - fXmean
		beta = 1;
		ldk = M;
		lda = M;
		alpha = 1;
		 dgemv(&transa, &M, &ldk,&alpha, ImB, &lda, meanY, &inci, &beta,mue, &incj);
	
		//if(verbose>7) printf("\t\t\t\t\t\t\t\tExit function: Ridge Regression. sigma^2 is: %f.\n\n",sigma2learnt);
		
		//error:  
		// y = ax + y   daxpy)(&MNtest, &alpha,Xptr, &inci,XsubPtr, &incj);
		alpha = -1;
		for(i=0;i<Ntest;i++)
		{
			readPtr = &NOISE[i*M];
			 daxpy(&M, &alpha,mue, &inci,readPtr, &incj);
		}
	
		//testNorm = FrobeniusNorm(testNoise, M, N);
		noiseNorm =  ddot(&MNtest, NOISE, &inci,NOISE, &incj);
		Errs[cv*Nrho + iRho] = noiseNorm;	//Errs[iRho, cv]
	} //iRho = 1: Nrho	
	
	if(verbose>6) printf("\t\t\t\t\t\tExit function: Ridge Regression-SVD. CV: %d; Error: %f.\n\n",cv, noiseNorm);

	free(NOISE);
	free(ImB);
	free(meanY);
	free(meanX);
	free(Brho);
	free(fRho);
}//rank 0

	free(Y);
	free(X);
	free(YiPi);
	//free(biInv);
	free(bi);	
	//free(YiPiyi);
	free(xiYi);

	//
	free(Hi);
	free(Yi);
	free(xi);
	free(xii);
	free(yi);
	//

	//
	free(S);
	free(U);
	free(Ucopy);
	free(VT);
	free(work);
	free(iwork);	
	free(scale);
	
	free(messageRho);
	free(receive_counts);
	free(receive_displacements);
	//MPI_Barrier (MPI_COMM_WORLD);
	// sigma2learnt;

}


//ridge regression; return sigma2learnt
double constrained_ridge_cff(double *Ycopy, double *Xcopy, double rho_factor, int M, int N,
		double *B, double *f, double *mue, int verbose,int rank, int size)
{
	
	int i,j,k,lda,ldb,ldc,ldk;
	// center Y, X
	double *meanY, *meanX;
if(rank==0)
{	
	meanY = (double* ) calloc(M, sizeof(double));
	meanX = (double* ) calloc(M, sizeof(double));
}	
	//copy Y, X; 
	double *Y, *X;
	int MN = M*N;
	Y = (double* ) calloc(MN, sizeof(double));
	X = (double* ) calloc(MN, sizeof(double));
	
	//F77_NAME(dcopy)(const int *n, const double *dx, const int *incx,
		//double *dy, const int *incy);
	int inci = 1;
	int incj = 1;

	
	
if(rank==0)
{	
	 dcopy(&MN,Ycopy,&inci,Y,&incj);
	 dcopy(&MN,Xcopy,&inci,X,&incj);
	
	centerYX(Y,X,meanY, meanX,M, N);

	//printf("MPI:: broadcast LOCAL Y,X with Number of elements: %d.\n\n",MN);
		
}
	//---------------------------MPI::
	//MPI::Bcast
	MPI_Bcast (Y, MN, MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast (X, MN, MPI_DOUBLE,0,MPI_COMM_WORLD);
		
	// MPI message vector for for loop
	double *message;

	//MPI block decomposition
	int low,high,count;
	low = BLOCK_LOW(rank,size,M);
	high = BLOCK_HIGH(rank,size,M);
	count = BLOCK_SIZE(rank,size,M); 	//number of rows to compute
	count = count * M; 					// number of elements to communicate

	//MPI loop
	int *receive_counts;
    int *receive_displacements;
	receive_counts = (int *) calloc(size,sizeof(int));
	receive_displacements = (int *) calloc(size,sizeof(int));
	//number of elements for each process	
	MPI_Gather(&count, 1, MPI_INT, receive_counts,1, MPI_INT, 0, MPI_COMM_WORLD);
	receive_displacements[0] = 0;
	for(i=1;i<size;i++)
	{	
		receive_displacements[i] = receive_counts[i-1] + receive_displacements[i-1];
	}
	
	message = (double* ) calloc(count, sizeof(double));
	
	//--------------------------mpi::	
	
	
	
//	double NPresent = 0;
//	for(i=0;i<M;i++)
//	{
//		for(j=0;j<N;j++)
//		{
//			if(Y[j*M + i]!=0) NPresent = NPresent + 1; //Y[i,j]
//		}
//	}	
if(rank==0)
{
	if(verbose>7) printf("\t\t\t\t\t\t\t\tEnter Function: Ridge Regression. Shrinkage ratio rho is: %f.\n\n",rho_factor);
}	
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
//printf("check point 1: before loop\n");
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
	
	
	//for(i=0;i<M;i++)
	int iter;
	iter = 0;

	for(i=low;i<=high;i++)
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
//printMat(xi,1,N);		
		//xi'*xi
		
		//YiPi
		//Hi          = xi*xi'/(xi'*xi);
        //Pi          = eye(N)-Hi;
//printf("check point 2: xixi: %f.\n",xixi);

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
//printMat(Hi,N,N);	
		
	
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
//printMat(Yi,Mi,N);		
		//transb = 'N';
		lda = Mi;
		ldb = N;
		ldc = Mi;
		ldk = N; //b copy
		 dgemm(&transa, &transb,&Mi, &N, &ldk,&alpha, Yi, &lda, Hi, &ldb, &beta, YiPi, &ldc);
		// dgemm)(&transa, &transb,&Mi, &N, &N,&alpha, Yi, &Mi, Hi, &N, &beta, YiPi, &Mi);
		
		//printf("check point 3: YiPi\n");
		
		//YiPi*Yi' --> MixMi
		transb = 'T';
		ldk = Mi;
		lda = Mi;
		ldb = Mi;
		ldc = Mi;
		 dgemm(&transa, &transb,&Mi, &ldk, &N,&alpha, YiPi, &lda, Yi, &ldb, &beta, YiPi2Norm, &ldc);
		// dgemm)(&transa, &transb,&Mi, &Mi, &N,&alpha, YiPi, &Mi, Yi, &Mi, &beta, YiPi2Norm, &Mi); //M-> N -> K
		//2-norm YiPi this is the largest eigenvalue of (Yi'*Pi)*YiPi
		//Matrix 2-norm;
		//Repeat compution; use biInv;
		//normYiPi = Mat2NormSq(YiPi,Mi,N); //MixN
		//YiPi2Norm
		
		//printf("check point 4: YiPi2Norm\n");
		
//normYiPi = largestEigVal(YiPi2Norm, Mi,verbose);  //YiPi2Norm make a copy inside largestEigVal, YiPi2Norm will be intermediate value of biInv;
		//transa = 'N';
		transb = 'N';
		//j = Mi*Mi;
		 dcopy(&MiMi,YiPi2Norm,&inci,biInv,&incj);
		
		// dgeev)(&transa, &transb,&Mi, biInv, &Mi, wr, wi, vl, &ldvl,vr, &ldvr, work, &lwork, &info);
		lda = Mi;
		 dsyevd(&jobz, &uplo,&Mi, biInv, &lda, w, work, &lwork, iwork, &liwork,&info);
		normYiPi = w[Mi -1];
//printMat(w,1,Mi);		
//printf("Eigenvalue: %f.\n",normYiPi);		
		rho = rho_factor*normYiPi; // 2Norm = sqrt(lambda_Max)
if(rank==0)
{		
		if(verbose>8) printf("\t\t\t\t\t\t\t\t\t Gene number: %d,\t shrinkage rho: %f\n",i,rho);
}
		//biInv = (YiPi*Yi+rho*I) ; (M-1) x (M-1)

		//
		for(j=0;j<Mi;j++) 
		{
			index = j*Mi + j;
			YiPi2Norm[index] = YiPi2Norm[index] + rho;
		}
		//biInv;

		//Inverse
		//MatrixInverse(biInv,Mi);
		//printf("check point 5: biInv inversed\n");
		//YiPiyi = Yi'*Pi*yi  = YiPi *yi;
		//F77_NAME(dgemv)(const char *trans, const int *m, const int *n,
		//const double *alpha, const double *a, const int *lda,
		//const double *x, const int *incx, const double *beta,
		//double *y, const int *incy);
		lda = Mi;
		 dgemv(&transa, &Mi, &N,&alpha, YiPi, &lda, yi, &inci, &beta,bi, &incj);
		

//linearSystem(YiPi2Norm,Mi,bi);//A NxN matrix
		lda = Mi;
		ldb = Mi;
		 dgesv(&Mi, &inci, YiPi2Norm, &lda, ipiv, bi, &ldb, &info);
		//printf("check point 6: bi updated\n");
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

//		f[i] = (xiyi-xiYibi)/xixi;
		//MPI::MESSAGE
		message[iter*M + i] = (xiyi-xiYibi)/xixi;
		
		//printf("check point 7: fi calculated\n");
		//update B
		jj = 0;
		for(j = 0;j<M;j++)
		{
			if(j!=i)
			{
				//B[i,j] = bi[jj];
				message[iter*M + j] = bi[jj];		//message[iter,j]; message is Mx count; 
				jj = jj +1;
			}
		}

		iter = iter  +1;
	
	}//i = 1:M
	//MPI::Gather	
	MPI_Gatherv(message, count, MPI_DOUBLE, B,receive_counts,receive_displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);		
			
	
if(rank==0)
{
	//transpose B
	transposeB(B,M,M);
	//printMat(B,M,M);
}	
	
	
	
	
	double noiseNorm, sigma2learnt;
	double *ImB;
	double * NOISE; 	//MxN
if(rank==0)
{
	//MPI::RANK0 f, B
	for(i=0;i<M;i++)
	{
		f[i] = B[i*M + i];
		B[i*M + i] = 0;
	}



	//I -B

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
		//NOISE[i,1:N]	
	}//row i = 1:M
	
	
	//NoiseMatF77(NOISE,ImB,Y, f, X, M, N);
		
	
	//noiseNorm = FrobeniusNorm(NOISE, M, N); //NOISE  = sparse(speye(M)-B)*Y-bsxfun(@times,f,X);
	//noiseNorm =  dnrm2)(&MN,NOISE,&inci);
	//sigma2learnt = noiseNorm*noiseNorm/(MN -1); //sigma2learnt    = sum(sum(NOISE.^2))/(sum(NPresent)-1);
	noiseNorm =  ddot(&MN, NOISE, &inci,NOISE, &incj);
	sigma2learnt = noiseNorm/(MN -1);
	
	//mue          = (IM-B)*meanY-bsxfun(@times,f,meanX);
	//dgemv mue = Ax + beta*mue
	

	
	
	
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
}	

	free(Y);
	free(X);
	free(YiPi);
	//free(biInv);
	free(YiPi2Norm);
	free(bi);	
	//free(YiPiyi);
	free(xiYi);

	//
	free(Hi);
	free(Yi);
	free(xi);
	free(yi);
	//
if(rank==0)
{	
	free(NOISE);
	free(ImB);
	free(meanY);
	free(meanX);
}
	//
	free(biInv);

	free(w);
	free(iwork);
	free(work);
	
	free(ipiv);
	
	free(message);
	free(receive_counts);
	free(receive_displacements);
	MPI_Barrier (MPI_COMM_WORLD);
	return sigma2learnt;

}




//by Weighted_LassoSf xi
//lambda_max          = max(max(abs(N*sigma2*IM - (Y*(Y'-X'*DxxRxy)))./W  ));
double lambdaMax_adaEN(double *Y,double *X,double * Wori,int M, int N,double alpha_factor)
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
	int MN = M*N;
	int MM = M*M;
	//------adaEN Apr. 2013	
	double *W;
	W 				= (double* ) calloc(MM, sizeof(double));
	dcopy(&MM,Wori,&inci,W,&incj);
	dscal(&MM,&alpha_factor,W,&inci); 
	//------adaEN Apr. 2013	
	//for(i=0;i<M;i++)
	//{
	//	dxx[i] 		= 0;
	//	rxy[i] 		= 0;
	//	for(j=0;j<N;j++)
	//	{
	//		index = j*M+i;
	//		dxx[i] 	= dxx[i] + pow(X[index],2);	// sum of each row X[i,j]
	//		rxy[i] 	= rxy[i] + X[index]*Y[index];			
	//	}
	//	DxxRxy[i] 	= rxy[i]/dxx[i];		
	//}
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
	//printMat(DxxRxy,M,1);

	//cache X[k,:]*DxxRxy[k]
	double * XDxxRxy;

	XDxxRxy = (double* ) calloc(MN, sizeof(double));
	//for(i=0;i<M;i++)
	//{
	//	for(j=0;j<N;j++)
	//	{
			//X[i,j] * DxxRxy[i];
	//		index = j*M + i;
	//		XDxxRxy[index] = -X[index]*DxxRxy[i];
	//	}
	//}
	 dcopy(&MN,X,&inci,XDxxRxy,&incj);
	double alpha;	
	for(i=0;i<M;i++)
	{
		alpha  = -DxxRxy[i];
		readPtr1 = &XDxxRxy[i]; //ith row
		 dscal(&N,&alpha, readPtr1,&M);//	(n, a, x, incx)
	}
	
	
	//printMat(XDxxRxy,M,1);
	// Y- XDxxRxy  			daxpy(n, a, x, incx, y, incy) y= ax + y
	alpha  = 1.0;
	// XDxxRxy <- alpha*Y + XDxxRxy
	 daxpy(&MN,&alpha,Y,&inci,XDxxRxy,&inci);
	//printMat(XDxxRxy,M,1);
	double *YYXDR; //= Y*XDxxRxy'

	YYXDR = (double* ) calloc(MM, sizeof(double));
	
//C := alpha*op(A)*op(B) + beta*C,
	double beta;
	char transa = 'N';
	char transb = 'T';
	alpha = -1;
	beta = 0;
	 dgemm(&transa, &transb,&M, &M, &N,&alpha, Y,&M, XDxxRxy, &M, &beta,YYXDR, &M); //M xK, K xN  --> MxN, N xM --> M <-M, N<-M, k<-N
	//diagonal -->0; other element /wij
	
	//printMat(YYXDR,M,3);
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
	//printMat(YYXDR,M,3);
	//BLAS_extern int    /* IDAMAX - return the index of the element with max abs value */
	//F77_NAME(idamax)(const int *n, const double *dx, const int *incx);
	index =  idamax(&MM,YYXDR,&inci);
	//printf("index: %d\n",index);
	lambda_max = fabs(YYXDR[index-1]);

	free(dxx);
	free(rxy);
	free(DxxRxy);
	//free(XX);
	free(XDxxRxy);
	free(YYXDR);
	//------adaEN Apr. 2013	
	free(W);
	//------adaEN Apr. 2013	
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
	
	// Y- XDxxRxy  			daxpy(n, a, x, incx, y, incy) y= ax + y
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
	//transpose 	(Y-X*DxxRxy)*Y'

	ldb = M;
	ldc = M;
	ldk = M;
	 dgemm(&transa, &transb,&M, &lda, &N,&alpha, XDxxRxy,&ldb, Y, &ldc, &beta,Q, &ldk); //M xK, K xN  --> MxN, N xM --> M <-M, N<-M, k<-N	
	
	//diagonal -->0; other element /wij
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

// 8888888888888888888888888888888888888888888888888888888888888888888888
//Q = N*sigma2*inv(I-B)-(Y-B*Y-fL*X)-mueL*ones(1,N))*Y';
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

	
	//MatrixInverse(IBinv,M);
	//By linear solver: not inverse (IB*x = IM) result stored in IM; 
	//multiLinearSystem(IB, M,IBinv,M);
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

	//NoiseMat(Noise,B,Y, f, X, M, N);
	//Noise - Mue
	//Errs(irho,cv)   = norm((1-Missing_test).*(A*Ytest-bsxfun(@times,fR,Xtest)-mueR*ones(1,Ntest)),'fro')^2;
	//Noise - mue: Mx N
	alpha = -1;
	for(i=0;i<N;i++)
	{
		readPtr1 = &Noise[i*M];
		 daxpy(&M, &alpha,mue, &inci,readPtr1, &incj);
	}	
	//Nsigma2*IBinv -  Noise *Y'
	//-Noise*Y' -->Q  	C := alpha*op(A)*op(B) + beta*C,
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

	
	//MatrixInverse(IBinv,M);
	//By linear solver: not inverse (IB*x = IM) result stored in IM; 
	//multiLinearSystem(IB, M,IBinv,M);
	//int info = 0;
	//int *ipiv;
	//ipiv = (int *) calloc(M,int);
	int lda = M;
	int ldb = M;
	int ldc = M;
	int ldk = M;
	// dgesv)(&M, &ldk, IBcopy, &lda, ipiv, IBinv, &ldb, &info);

	
	
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

	//NoiseMat(Noise,B,Y, f, X, M, N);
	//Noise - Mue
	//Errs(irho,cv)   = norm((1-Missing_test).*(A*Ytest-bsxfun(@times,fR,Xtest)-mueR*ones(1,Ntest)),'fro')^2;
	//Noise - mue: Mx N

	//Nsigma2*IBinv -  Noise *Y'
	//-Noise*Y' -->Q  	C := alpha*op(A)*op(B) + beta*C,
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


// 8888888888888888888888888888888888888888888888888888888888888888888888
//BLOCK COORDINATE ASCENSION: QIBinv = inv(I-B): by multi linear system
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
	
	//MatrixInverse(IBinv,M);
	//By linear solver: not inverse (IB*x = IM) result stored in IM; 
	//multiLinearSystem(IB, M,IBinv,M);
	int info = 0;
	int *ipiv;
	ipiv = (int *) calloc(M,sizeof(int));
	 dgesv(&M, &ldk, IB, &lda, ipiv, IBinv, &ldb, &info);
	double *ptr1,*ptr2;
	//for(i=0;i<M;i++) printf("IPIV: \n: %d \t",ipiv[i]);
	//printf("\n");
	
	
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


// 8888888888888888888888888888888888888888888888888888888888888888888888
//BLOCK COORDINATE ASCENSION: QIBinv = inv(I-B): by multi linear system
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
	
	//MatrixInverse(IBinv,M);
	//By linear solver: not inverse (IB*x = IM) result stored in IM; 
	//multiLinearSystem(IB, M,IBinv,M);
	int info = 0;
	int *ipiv;
	ipiv = (int *) calloc(M,sizeof(int));
	 dgesv(&M, &ldk, IB, &lda, ipiv, QIBinv, &ldb, &info);

	free(IB);
	free(ipiv);
}

//no Missing
//--------------------------------------------------------------------------------  WEIGHTED_LASSOSF
double Weighted_LassoSf_adaEN(double * Wori, double *B, double *f, double *Ycopy,double *Xcopy,
		double *Q, double lambda_factor, double lambda_factor_prev, double sigma2, int max_iter,
		int M, int N, int verbose,double *QIBinv,double lambda_max, int rank,int size,			//double * mue,
		double alpha_factor)
{
	int i,j,index,ldM;
	//lda = M;
	//ldb = M;ldb,
	ldM = M;//fixed
	int MN = M*N;
	int MM = M*M;
	//---------------------------MPI::
	//MPI_Bcast (W, MM, MPI_DOUBLE,0,MPI_COMM_WORLD);		//allocated from mainSML
	MPI_Bcast (QIBinv, MM, MPI_DOUBLE,0,MPI_COMM_WORLD);	//ptr to a large chunck at cv_support
//	MPI_Bcast (BL, MM, MPI_DOUBLE,0,MPI_COMM_WORLD);	
	MPI_Bcast (f, M, MPI_DOUBLE,0,MPI_COMM_WORLD);	
//--------------------------mpi::
	

	// return lambda;
	double *meanY, *meanX;
if(rank==0)
{	
	meanY = (double* ) calloc(M, sizeof(double));
	meanX = (double* ) calloc(M, sizeof(double));
}
	
	//copy Y, X; 
	double *Y, *X;

	Y = (double* ) calloc(MN, sizeof(double));
	X = (double* ) calloc(MN, sizeof(double));
	
	//F77_NAME(dcopy)(const int *n, const double *dx, const int *incx,
		//double *dy, const int *incy);
	int inci,incj, inc0;
	inci	= 1;
	incj 	= 1;
	inc0 	= 0;
if(rank==0)
{	
	 dcopy(&MN,Ycopy,&inci,Y,&incj);
	 dcopy(&MN,Xcopy,&inci,X,&incj);
	centerYX(Y,X, meanY, meanX,M, N);
}
	
	//MPI::Bcast
	MPI_Bcast (Y, MN, MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast (X, MN, MPI_DOUBLE,0,MPI_COMM_WORLD);
	
	//return value
	//double sigma2 			= SIGMA2[0];
	double lambda;//lambda_max,
	//lambdaMax
	//lambda_max 				= lambdaMax(Y,X,W,M, N);
if(rank==0)
{	
	if(verbose>4) printf("\t\t\t\tEnter Function: weighted_LassoSf. The maximum lambda is: %f\n\n",lambda_max);
}
	lambda 					= lambda_factor*lambda_max;
	
	//none zeros
	double alpha,beta;
	beta = 0;
	double deltaLambda;
//------adaEN Apr. 2013		
	double *s, *S,*Wcopy, *W;
	S = (double* ) calloc(MM, sizeof(double));
	s = (double* ) calloc(M, sizeof(double));
	W 						= (double* ) calloc(MM, sizeof(double));
	dcopy(&MM,Wori,&inci,W,&incj);
	dscal(&MM,&alpha_factor,W,&inci); 
//------adaEN Apr. 2013		
if(rank==0)
{	
	Wcopy = (double* ) calloc(MM, sizeof(double));
	 dcopy(&MM,W,&inci,Wcopy,&incj);

	deltaLambda 			= (2*lambda_factor - lambda_factor_prev)*lambda_max;	
	 dscal(&MM,&deltaLambda,Wcopy,&inci); //wcopy = deltaLambda*W
}	
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
if(rank==0)
{	
	for(i=0;i<M;i++)
	{
		for(j=0;j<M;j++)
		{
			//W[i,j]
			index = j*M  +i;
// version 1_3: Qij			
			//if(fabs(Q[index])>= Wcopy[index] && i!= j)
			//if(fabs(Q[index] -(1-alpha_factor)*lambda*B[index])>= Wcopy[index] && i!= j)	
			if(fabs(Q[index] -2*(1-alpha_factor)*lambda*B[index]*Wori[index]*Wori[index])>= Wcopy[index] && i!= j)	
// version 1_3: Qij	
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
}
	//MPI::Bcast	
	MPI_Bcast (B, MM, MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast (S, MM, MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast (s, M, MPI_DOUBLE,0,MPI_COMM_WORLD);

	
	char transa = 'N'; 
/*	ldk = M;
	//lda = M;
	 dgemv)(&transa, &M, &ldk,&alpha, S, &ldM, eye, &inci, &beta,s, &incj);
*/	
//printMat(W,M,M);
	//f0, F1
	double *f0,*F1;
	//int qdif = M*M;
	f0 	= (double* ) calloc(M, sizeof(double));
	F1 	= (double* ) calloc(MM, sizeof(double));
	
	double *y_j;
	//xi 	= (double* ) calloc(N, double);
	y_j 	= (double* ) calloc(N, sizeof(double));
	double *F1ptr;


	double XYi, XXi;
if(rank==0)
{	
	for(i=0;i<M;i++)
	{
		readPtr = &X[i];
		// dcopy)(&N,readPtr,&M,xi,&inci);
		readPtr2 = &Y[i];
		// dcopy)(&N,readPtr2,&M,y_j,&inci);

		//dot product
		//XYi =  ddot)(&N, xi, &inci,y_j, &incj);
		XYi =  ddot(&N, readPtr, &M,readPtr2, &M);
		//XXi =  ddot)(&N, xi, &inci,xi, &incj);
		//norm2 =  dnrm2)(&N,xi,&inci);
		//XXi 	= pow(norm2,2);
		XXi =  ddot(&N, readPtr, &M,readPtr, &M);
		f0[i] 	= XYi/XXi;
		F1ptr	= &F1[M*i];//start from ith column
		//Y*X(i,:)' 		y := alpha*A*x + beta*y,
		alpha = 1/XXi;
		 dgemv(&transa, &M, &N,&alpha, Y, &ldM, readPtr, &M, &beta,F1ptr, &incj);
	}
}
	//MPI::Bcast	
	MPI_Bcast (F1, MM, MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast (f0, M, MPI_DOUBLE,0,MPI_COMM_WORLD);

	
	
	//printMat(f0,M,1);

	// entering loop		
	double *IBinv,*zi,*a_iT;		// y_j: one row of Y: Nx1
if(rank==0)
{	
	IBinv 	= (double* ) calloc(MM, sizeof(double));
}
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
if(rank==0)
{	
	index = M*(M  +1);
	BfOld = (double* ) calloc(index, sizeof(double));
	BfNew = (double* ) calloc(index, sizeof(double));
	BfChange = (double* ) calloc(index, sizeof(double));
}	
	//	//linear system
	//int *ipiv;
	//ipiv = (int *) R_alloc(N,sizeof(int));
	//ipiv = (int *) calloc(M,int);
	//int nrhs = 1;
	//int info = 0;
//for(i=0;i<M;i++) ipiv[i] = i+1;
	//
	//double *aiTQuad; 
	//aiTQuad 	= (double* ) calloc(N, double);

	//alpha =   dnrm2)(&index,BfOld,&inci);	
//printf("BfOldnorm:%f\n",alpha);	
	//int anyUpdateRow = 0;
//MPI:: ------------------------------------------ MESSAGE, BLOCKS
	int low,high,count,MPIiter;
	low = BLOCK_LOW(rank,size,M);
	high = BLOCK_HIGH(rank,size,M);
	count = BLOCK_SIZE(rank,size,M); 	//number of rows to compute
	count = count * M; 					// number of elements to communicate

//MPI loop
	int *receive_counts;
    int *receive_displacements;
	receive_counts = (int *) calloc(size,sizeof(int));
	receive_displacements = (int *) calloc(size,sizeof(int));
	//number of elements for each process	
	MPI_Gather(&count, 1, MPI_INT, receive_counts,1, MPI_INT, 0, MPI_COMM_WORLD);
	receive_displacements[0] = 0;
	for(i=1;i<size;i++)
	{	
		receive_displacements[i] = receive_counts[i-1] + receive_displacements[i-1];
	}
	
	// MPI message vector for for loop
	double *message;
	message = (double* ) calloc(count, sizeof(double));

	//--------------------------mpi::
	
	while(iter < max_iter)
	{
		iter = iter + 1;
if(rank==0)
{		
		//converge Bfold = [B f];
		 dcopy(&MM,B,&inci,BfOld,&incj);
		//last column
		F1ptr = &BfOld[MM];
		 dcopy(&M,f,&inci,F1ptr,&incj);
}		 
	//alpha =   dnrm2)(&index,BfOld,&inci);	
//printf("BfOldnorm:%f\n",alpha);
		//printMat(BfOld,M,M+1);
		//
		//for(i=0;i<M;i++)
		MPIiter = 0;
		for(i=low;i<=high;i++)
		{
			if(s[i] >0)
			{ 	//
if(rank==0)
{			
				if(verbose>6) printf("\t\t\t\t\t updating gene %d \n",i);
				//
}				
				ei[i] = 1;
				//zi   IBinv = I -B
				// dcopy)(&MM,B,&inci,IBinv,&incj);
				//alpha = -1; 
				// dscal)(&MM,&alpha,IBinv,&inci); // dscal(n, a, x, incx) x = a*x
				//diagonal + 1
				//for(j=0;j<M;j++) 
				//{
				//	index = j*M + j;
				//	IBinv[index] = 1 + IBinv[index];
				//}

				//call matrix inverse
				//MatrixInverse(IBinv,M);
				//zi is ith column of IBinv
				//for(j =0;j<M;j++) zi[j] 	= IBinv[i*M+j];
				
				// by linear solver: zi = (I - B)\ei;
				//copy ei to zi;
				zi = &QIBinv[i*M];
				// dcopy)(&M,ei,&inci,zi,&incj);
				//
				//if (i==0) printMat(zi,M,1);
				//
				//call linearSystem(double *a, int N,double *B) a is NxN
//linearSystem(IBinv, M,zi);  //zi is updated.

				// dgesv)(&M, &nrhs, IBinv, &ldM, ipiv, zi, &ldb, &info);
				//anyUpdateRow = 1;
//j reserved				//for j = js_i
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
							//eiB = ei-BiT			//daxpy(n, a, x, incx, y, incy) 		y := a*x + y
							alpha = 1;
							 daxpy(&M, &alpha,BiT, &inci,eiB, &incj);
							//a_iT      = (ei'-BiT)*Y-f(i)*X(i,:);
							readPtr = &X[i];
							 dcopy(&N,readPtr,&M,a_iT,&inci);	

							//a_iT = -f[i]*xi 		dscal(n, a, x, incx) 		x = a*x
							alpha = -f[i];
							 dscal(&N,&alpha,a_iT,&inci);							

							transa='T'; //y := alpha*A**T*x + beta*y, 		 dgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
							beta = 1;
							alpha = 1;
							 dgemv(&transa, &M, &N,&alpha, Y, &ldM, eiB, &inci, &beta,a_iT, &incj);
							
							//r_ij                = y_j'*y_j;	
r_ij =  ddot(&N, y_j, &inci,y_j, &incj);
//------adaEN Apr. 2013	
r_ij = r_ij + (1 -alpha_factor)*lambda*Wori[j*M + i]*2*Wori[j*M + i];
//------adaEN Apr. 2013		
							//norm2 =  dnrm2)(&N,y_j,&inci);
							//r_ij 	= pow(norm2,2);
							
							//beta_ij             = a_iT*y_j;
							beta_ij =  ddot(&N, y_j, &inci,a_iT, &incj);
							
							if (fabs(m_ij)<1e-10) //go to the linear equation 
							{
if(rank==0)
{							
								//
								if(verbose>7) printf("\t\t\t\t\t\t\t gene %d \t interact with gene %d.\tLinear equation\n",i,j);
								//
}								
								Bij = (beta_ij-lambdaW)/r_ij;
								//printf("\t\t gene (%d,\t%d): linear Bij %f\n",i,j,Bij);
				//
				//if (i==0) 
				//{
				//	printf("\t\t\t beta_ij: %f;\t r_ij:%f; lambdaW: %f\n ",beta_ij,r_ij,lambdaW);
				//	printMat(a_iT,N,1);
					//printMat(y_j,N,1);
				//	printMat(eiB,M,1);
				//}
				//
								if(Bij>0) 
								{
									B[j*M+i] = Bij;//B(i,j)      = Bij;
									//message[i,j]
									message[MPIiter*M + j] = Bij;
								}else
								{
									Bij         = (beta_ij+lambdaW)/r_ij;
									if(Bij<0)
									{
										B[j*M+i] = Bij;//B(i,j)      = Bij;
										message[MPIiter*M + j] = Bij;
									}else
									{
										B[j*M+i] = 0;
										message[MPIiter*M + j] = 0;
									}
								}//B_ij>0 
							}else //m_ij ~=0 go to the quadratic equation
							{
if(rank==0)
{							
								//
								if(verbose>7) printf("\t\t\t\t\t\t\t gene %d \t interact with gene %d.\tQuadratic equation\n",i,j);
								//
}								
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
								//Lss = quadraticLss(sigma2, N, d_ij, candsBij, r_ij, lambdaW,beta_ij);								
								//Lss = quadraticLssJuan(sigma2,N, d_ij, candsBij, a_iT,y_j,lambdaW);
//beta =  ddot)(&N, a_iT, &inci,a_iT, &incj);
								//norm2 =  dnrm2)(&N,a_iT,&inci);
								//beta 	= pow(norm2,2);
							
								//Lss = sigma2*N*log(fabs(d_ij - candsBij)+1e-16) - beta/2 -lambdaW*fabs(candsBij);
								//Lss = sigma2*N*log(fabs(d_ij)+1e-16) - beta/2;
								Lss = sigma2*N*log(fabs(d_ij)+1e-16);
								
								// a_iT = a_iT - candsBij*y_j 		daxpy(n, a, x, incx, y, incy) 	y := a*x + y							
								if (Bijpp>0)
								{
									//LssCands = quadraticLss(sigma2, N, d_ij, Bijpp, r_ij, lambdaW,beta_ij);
									//LssCands = quadraticLssJuan(sigma2,N, d_ij, Bijpp, a_iT,y_j,lambdaW);	
									//aiTQuad dcopy(n, x, incx, y, incy)  y= x
									// dcopy)(&N,a_iT,&inci,aiTQuad,&incj);
									//aiTQuad = aiTQuad - candsBij*y_j  daxpy(n, a, x, incx, y, incy) 	y := a*x + y
									//alpha = -Bijpp;
									// daxpy)(&N, &alpha,y_j, &inci,aiTQuad, &incj);
//beta =  ddot)(&N, aiTQuad, &inci,aiTQuad, &incj);
									//norm2 =  dnrm2)(&N,aiTQuad,&inci);
									//beta 	= pow(norm2,2);
									//LssCands = sigma2*N*log(fabs(d_ij - Bijpp)+1e-16) - beta/2 -lambdaW*fabs(Bijpp);
									LssCands = sigma2*N*log(fabs(d_ij - Bijpp)+1e-16) - r_ij*pow(Bijpp,2)/2 + beta_ij*Bijpp -lambdaW*fabs(Bijpp); 
									
									if(LssCands>Lss) 
									{
										candsBij = Bijpp;
										Lss 	= LssCands;
									}	
								}
								if (Bijpm>0)
								{
									//LssCands = quadraticLss(sigma2, N, d_ij, Bijpm, r_ij, lambdaW,beta_ij);
									//LssCands = quadraticLssJuan(sigma2,N, d_ij, Bijpm, a_iT,y_j,lambdaW);								
									// dcopy)(&N,a_iT,&inci,aiTQuad,&incj);
									//aiTQuad = aiTQuad - candsBij*y_j  daxpy(n, a, x, incx, y, incy) 	y := a*x + y
									//alpha = -Bijpm;
									// daxpy)(&N, &alpha,y_j, &inci,aiTQuad, &incj);
//beta =  ddot)(&N, aiTQuad, &inci,aiTQuad, &incj);
									//norm2 =  dnrm2)(&N,aiTQuad,&inci);
									//beta 	= pow(norm2,2);
									//LssCands = sigma2*N*log(fabs(d_ij - Bijpm)+1e-16) - beta/2 -lambdaW*fabs(Bijpm);
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
									//LssCands = quadraticLss(sigma2, N, d_ij, Bijmm, r_ij, lambdaW,beta_ij);
									//LssCands = quadraticLssJuan(sigma2,N, d_ij, Bijmm, a_iT,y_j,lambdaW);								
									// dcopy)(&N,a_iT,&inci,aiTQuad,&incj);
									//aiTQuad = aiTQuad - candsBij*y_j  daxpy(n, a, x, incx, y, incy) 	y := a*x + y
									//alpha = -Bijmm;
									// daxpy)(&N, &alpha,y_j, &inci,aiTQuad, &incj);
//beta =  ddot)(&N, aiTQuad, &inci,aiTQuad, &incj);
									//norm2 =  dnrm2)(&N,aiTQuad,&inci);
									//beta 	= pow(norm2,2);
									//LssCands = sigma2*N*log(fabs(d_ij - Bijmm)+1e-16) - beta/2 -lambdaW*fabs(Bijmm);
									LssCands = sigma2*N*log(fabs(d_ij - Bijmm)+1e-16) - r_ij*pow(Bijmm,2)/2 + beta_ij*Bijmm -lambdaW*fabs(Bijmm);  
									if(LssCands>Lss) 
									{
										candsBij = Bijmm;
										Lss 	= LssCands;
									}	
								}
								if (Bijmp<0)
								{
									//LssCands = quadraticLss(sigma2, N, d_ij, Bijmp, r_ij, lambdaW,beta_ij);
									//LssCands = quadraticLssJuan(sigma2,N, d_ij, Bijmp, a_iT,y_j,lambdaW);									
									// dcopy)(&N,a_iT,&inci,aiTQuad,&incj);
									//aiTQuad = aiTQuad - candsBij*y_j  daxpy(n, a, x, incx, y, incy) 	y := a*x + y
									//alpha = -Bijmp;
									// daxpy)(&N, &alpha,y_j, &inci,aiTQuad, &incj);
//beta =  ddot)(&N, aiTQuad, &inci,aiTQuad, &incj);
									//norm2 =  dnrm2)(&N,aiTQuad,&inci);
									//beta 	= pow(norm2,2);
									//LssCands = sigma2*N*log(fabs(d_ij - Bijmp)+1e-16) - beta/2 -lambdaW*fabs(Bijmp);			
									LssCands = sigma2*N*log(fabs(d_ij - Bijmp)+1e-16) - r_ij*pow(Bijmp,2)/2 + beta_ij*Bijmp -lambdaW*fabs(Bijmp); 
									if(LssCands>Lss) 
									{
										candsBij = Bijmp;
										Lss 	= LssCands;
									}	
								}
								B[j*M+i] = candsBij;
								message[MPIiter*M + j] = candsBij;
							}//m_ij
						}//if(j!= i)
						dB = B_old - B[j*M +i];
						//update c_ij
						ziDb = 1/(1 + dB*m_ij);
						 dscal(&M,&ziDb,zi,&inci);
						
						//update QIBinv ith column updated; jth row: 
						//readPtr = &QIBinv[j];
						// dscal)(&M,&ziDb,readPtr,&M);
						//QIBinv[j*M + i] = QIBinv[j*M + i]*ziDb;
						// adjugate of QIBinv = ziDb*(QIBinvadj - (-1)^kl) + (-1)^kl
						//for(k=0;k<M;k++) // kth row
						//{
						//	for(l=0;l<M;l++)//lth column
						//	{
						//		if(k!=j && l!=i) 
						//		{
						//			alpha = pow(-1.0,k+l);
						//			QIBinv[l*M+k] = ziDb*(QIBinv[l*M+k] -alpha) + alpha;
						//		}	
						//	}
						//}
						
					}//js_i >0
				}//j = 1:M	
				//f
				//BF1 = B(i,:)*F1(:,i)
				readPtr = &B[i];
				 dcopy(&M,readPtr,&ldM,BiT,&inci);

				F1ptr = &F1[M*i];
				BF1 =  ddot(&M, BiT, &inci,F1ptr, &incj);

				f[i] = f0[i] - BF1;
				
				//message
				message[MPIiter*M + i] = f[i];
				
				ei[i] = 0; // re-set ei for next i
			}else//s[i]  no un-zero weight in this gene
			{
				readPtr = &B[i];
				// dscal)(&M,&toyZero,readPtr,&ldM);
				 dcopy(&M,&toyZero,&inc0,readPtr,&ldM);
				f[i] = f0[i];
				
				//message : if B is "Bcast" each while loop, the update is not needed
				readPtr= &message[MPIiter*M];
				dcopy(&M,&toyZero,&inc0,readPtr,&inci);
				message[MPIiter*M + i] = f0[i];
				
			} // s[i]
			MPIiter = MPIiter  +1;
		}//i= 1:M
		MPI_Gatherv(message, count, MPI_DOUBLE, B,receive_counts,receive_displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);		
if(rank==0)
{
	//transpose B
	transposeB(B,M,M);
	//printMat(B,M,M);
	for(i=0;i<M;i++)
	{
		f[i] = B[i*M + i];
		B[i*M + i] = 0;
	}
}
		MPI_Bcast (B, MM, MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Bcast (f, M, MPI_DOUBLE,0,MPI_COMM_WORLD);

		//END: MPI parallel computation
if(rank==0)
{		
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
		
//printf("FnormOld: %f; \t FnormChange: %f\n",FnormOld,FnormChange);
		//MASTER IS HERE: Update before break: IBinv will be used in Qij
		// BLOCK COORDINATE ASCEND: Update IBinv
		UpdateIBinv(QIBinv, B,M);	
		
		if(verbose>5) printf("\t\t\t\t\t\tdelta_BF: %f\n",delta_BF);
}
		//MPI 
		MPI_Bcast (&delta_BF, 1, MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Bcast (QIBinv, MM, MPI_DOUBLE,0,MPI_COMM_WORLD);

		
		if(delta_BF<1e-3)		//break out
		{
			break;
		}
		

		
	}//while

	//QIBinv: compute not updated ones
	//F77_NAME(dgetrs)(const char* trans, const int* n, const int* nrhs,
	//	 const double* a, const int* lda, const int* ipiv,
	//	 double* b, const int* ldb, int* info);
	//if(anyUpdateRow>0) //QIBinv --> = Cij/det(I-B) != eye(M); ipiv not zeros
	//{
	//	transa = 'N';
	//	for(i=0;i<M;i++)
	//	{
	//		if(s[i] ==0)
	//		{ 	
	//			info = 0;
	//			ei[i] = 1;
	//			zi = &QIBinv[i*M];
	//			 dcopy)(&M,ei,&inci,zi,&incj);
				// dgetrs
	//			 dgetrs)(&transa, &M, &nrhs,IBinv,&M,ipiv,zi,&ldb,&info); //column i updated
	//			ei[i] = 0;
	//		}
	//	}
	//}else
	//{
		//initialize IBinv
	//	 dcopy)(&MM,&beta,&inc0,QIBinv,&incj);
	//	for(i=0;i<M;i++) QIBinv[i*M+i] =1;
	//}
	
	
if(rank==0)
{	
	if(verbose>4) printf("\t\t\t\tCurrent lambda: %f;\t number of iteration is: %d.\tExiting Weighted_LassoSf\n\n",lambda, iter);
	
	//mue          = (IM-B)*meanY-bsxfun(@times,f,meanX);

	//IBinv = I -B
	// dcopy)(&MM,B,&inci,IBinv,&incj);
	//alpha = -1; 
	// dscal)(&MM,&alpha,IBinv,&inci); // dscal(n, a, x, incx) x = a*x
	//diagonal + 1
	//for(j=0;j<M;j++) 
	//{
	//	index = j*M + j;
	//	IBinv[index] = 1 + IBinv[index];
	//	mue[j] = -f[j]*meanX[j];
	//}
	//mue[i] = -f[i]*meanX[i];
	
	//	//dgemv mue = Ax + beta*mue
	//transa = 'N';
	//alpha = 1;
	//beta = 1;
	//ldk = M;
	// dgemv)(&transa, &M, &ldk,&alpha, IBinv, &ldM, meanY, &inci, &beta,mue, &incj);

	//mue                     	= (IM-B)*meanY-bsxfun(@times,f,meanX); diag(f)*meanX

	free(meanY);
	free(meanX);
	free(Wcopy);
	free(IBinv);
	free(BfOld);
	free(BfNew);
	free(BfChange);	
}	
	free(Y);
	free(X);
	
	free(S);
	free(s);
	free(f0);
	free(F1);

	
	//free(xi);
	free(y_j);
	
	free(ei);

	//free(zi);
	free(a_iT);
	
	free(eiB);
	free(BiT);

	
	//free(ipiv);
	
	//free(aiTQuad);
	//free(eye);
	free(receive_counts);
	free(receive_displacements);
	free(message);
	//------adaEN Apr. 2013	
	free(W);
	//------adaEN Apr. 2013		
	
	return lambda;
	//sigma2 remains the same
}//weighted_LassoSf


//version 14 normYiPi
double normYiPiZero(double *Ycopy, double *Xcopy,int M, int N)
{
	int i,j,k,lda,ldb,ldc,ldk;
	// center Y, X
	double *meanY, *meanX;
	meanY = (double* ) calloc(M, sizeof(double));
	meanX = (double* ) calloc(M, sizeof(double));
	double *Y, *X;
	int MN = M*N;
	int MM = M*M;
	int NN = N*N;	
	Y = (double* ) calloc(MN, sizeof(double));
	X = (double* ) calloc(MN, sizeof(double));
	int inci = 1;
	int incj = 1;
	 dcopy(&MN,Ycopy,&inci,Y,&incj);
	 dcopy(&MN,Xcopy,&inci,X,&incj);
	// int Mi = M; MM
	centerYX(Y,X,meanY, meanX,M, N);
	
	double *YiPi; //Yi'*Pi
	YiPi =(double* ) calloc(M*N, sizeof(double));
	double xixi,xixiInv; //xi'*xi;
	int jj,index; //jj = 1:(M-1) index of YiPi
	double normYiPi;
	double *YiPi2Norm; 	//YiPi2Norm: first term of biInv;
	
	double *Hi,*Yi,*xi,*yi,*xii;//xii for Hi calculation Hi= xi*xi'
	Hi = (double* ) calloc(N*N, sizeof(double));
	Yi =(double* ) calloc(M*N, sizeof(double));
	xi = (double* ) calloc(N, sizeof(double));
	xii = (double* ) calloc(N, sizeof(double));
	yi = (double* ) calloc(N, sizeof(double));
	double alpha, beta;
	char transa = 'N';
	char transb = 'N';
	
	//
	YiPi2Norm 	= (double* ) calloc(MM, sizeof(double));
	alpha = 1;
	beta = 0;
		
//largest Eigenvalue
	//dsyevd
	char jobz = 'N'; // yes for eigenvectors
	char uplo = 'U'; //both ok
	double *w, *work;
	w = (double *) calloc(M,sizeof(double));
	int lwork = 5*M + 10;
	work  = (double *) calloc(lwork,sizeof(double));	
	int liwork = 10;
	int *iwork;
	iwork = (int *) calloc(liwork,sizeof(int));
	int info = 0;
	//dsyevd function 


	double *readPtr,*readPtr2;
	//loop starts here
	i=0;
	
		readPtr = &X[i];
		 dcopy(&N,readPtr,&M,xi,&inci);
		 dcopy(&N,xi,&inci,xii,&incj);
		readPtr = &Y[i];
		 dcopy(&N,readPtr,&M,yi,&inci);

		//xixi =  dnrm2)(&N,xi,&inci);
		//xixi = pow(xixi,2);
		xixi =  ddot(&N, xi, &inci,xi, &incj);
		xixiInv = -1/xixi;
//printMat(xi,1,N);		
		//xi'*xi
		
		//YiPi
		//Hi          = xi*xi'/(xi'*xi);
        //Pi          = eye(N)-Hi;
//printf("check point 2: xixi: %f.\n",xixi);

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
//printMat(Hi,N,N);	
		
	
		//Yi
		readPtr2 = &Yi[0];
		jj = 0;
		for(j=0;j<M;j++)
		{	if(j!=i)
			{
				//copy one j row
				readPtr = &Y[j];
				 dcopy(&N,readPtr,&M,readPtr2,&M);
				jj = jj + 1;
				readPtr2 = &Yi[jj];
			}
		}//jj 1:(M-1), YiPi[jj,:]
		//YiPi=Yi*Pi
//printMat(Yi,Mi,N);		
		//transb = 'N';
		lda = M;
		ldb = N;
		ldc = M;
		ldk = N; //b copy
		 dgemm(&transa, &transb,&M, &N, &ldk,&alpha, Yi, &lda, Hi, &ldb, &beta, YiPi, &ldc);
		// dgemm)(&transa, &transb,&Mi, &N, &N,&alpha, Yi, &Mi, Hi, &N, &beta, YiPi, &Mi);
		
		//printf("check point 3: YiPi\n");
		
		//YiPi*Yi' --> MixMi
		transb = 'T';
		ldk = M;
		lda = M;
		ldb = M;
		ldc = M;
		 dgemm(&transa, &transb,&M, &ldk, &N,&alpha, YiPi, &lda, Yi, &ldb, &beta, YiPi2Norm, &ldc);
		// dgemm)(&transa, &transb,&Mi, &Mi, &N,&alpha, YiPi, &Mi, Yi, &Mi, &beta, YiPi2Norm, &Mi); //M-> N -> K
		//2-norm YiPi this is the largest eigenvalue of (Yi'*Pi)*YiPi
		//Matrix 2-norm;
		//Repeat compution; use biInv;
		//normYiPi = Mat2NormSq(YiPi,Mi,N); //MixN
		//YiPi2Norm
		
		//printf("check point 4: YiPi2Norm\n");
		
//normYiPi = largestEigVal(YiPi2Norm, Mi,verbose);  //YiPi2Norm make a copy inside largestEigVal, YiPi2Norm will be intermediate value of biInv;
		//transa = 'N';
		transb = 'N';
		
		// dgeev)(&transa, &transb,&Mi, biInv, &Mi, wr, wi, vl, &ldvl,vr, &ldvr, work, &lwork, &info);
		lda = M;
		 dsyevd(&jobz, &uplo,&M, YiPi2Norm, &lda, w, work, &lwork, iwork, &liwork,&info);
		normYiPi = w[M -1];
	

	free(Y);
	free(X);
	free(YiPi);
	free(YiPi2Norm);
	//
	free(Hi);
	free(Yi);
	free(xi);
	free(xii);
	free(yi);
	free(meanY);
	free(meanX);

	free(w);
	free(iwork);
	free(work);
	return(normYiPi);
}


//combine lasso and zero_lasso for CV_selecting lambda
double Weighted_LassoSf_MLf_adaEN(double * Wori, double *BL, double *fL, double *Ycopy,double *Xcopy,
		double *Q, double lambda_factor, double lambda_factor_prev, double sigma2, int max_iter,
		int M, int N, int verbose, 			double *BC, double *fC, double *mue,double *QIBinv,
		double *IBinvZero,double lambda_max, int rank, int size,
		double alpha_factor)
{
//already alloc globally: QIBinv, W,BL,fL: Bcast
//for input variable needed in parallel: alloc memory; 

//---------------------------MPI::
	int MN = M*N;
	int MM = M*M;
	//MPI_Bcast (W, MM, MPI_DOUBLE,0,MPI_COMM_WORLD);		//allocated from mainSML
	MPI_Bcast (QIBinv, MM, MPI_DOUBLE,0,MPI_COMM_WORLD);	//ptr to a large chunck at cv_support
	MPI_Bcast (IBinvZero, MM, MPI_DOUBLE,0,MPI_COMM_WORLD);	////ptr to a large chunck at cv_support
//	MPI_Bcast (BL, MM, MPI_DOUBLE,0,MPI_COMM_WORLD);	
	MPI_Bcast (fL, M, MPI_DOUBLE,0,MPI_COMM_WORLD);	
//--------------------------mpi::	


//printf("check point 1: processor %d of %d \n", rank,size);

	//SET TO PART1: LASSO
	double *B, *f;
	B = &BL[0];
	f = &fL[0];
	//mue is used for calculation when Y, X are not centered: for testing set;

	int i,j,index,ldk,ldM;
	//lda = M;
	//ldb = M;ldb,
	ldM = M;//fixed
	// return lambda;
	double *meanY, *meanX;
	
if(rank==0)
{	
	meanY = (double* ) calloc(M, sizeof(double));
	meanX = (double* ) calloc(M, sizeof(double));
}	
	//copy Y, X; 
	double *Y, *X;

	Y = (double* ) calloc(MN, sizeof(double));
	X = (double* ) calloc(MN, sizeof(double));
	
	//F77_NAME(dcopy)(const int *n, const double *dx, const int *incx,
		//double *dy, const int *incy);
	int inci,incj,inc0;
	inci	= 1;
	incj 	= 1;
	inc0 	= 0;
if(rank==0)
{	
	 dcopy(&MN,Ycopy,&inci,Y,&incj);
	 dcopy(&MN,Xcopy,&inci,X,&incj);
	centerYX(Y,X, meanY, meanX,M, N);
}

	//MPI::Bcast
	MPI_Bcast (Y, MN, MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast (X, MN, MPI_DOUBLE,0,MPI_COMM_WORLD);
			
	//return value
	//double sigma2 			= SIGMA2[0];
	double lambda;//lambda_max,
	//lambdaMax
	//lambda_max 				= lambdaMax(Y,X,W,M, N);
	
if(rank==0)
{	
	if(verbose>4) printf("\t\t\t\tEnter Function: weighted_LassoSf. The maximum lambda is: %f\n\n",lambda_max);
}
	lambda 					= lambda_factor*lambda_max;
	
	//none zeros
	double alpha,beta;
	beta = 0;
	double deltaLambda;
//------adaEN Apr. 2013		
	double *s, *S,*Wcopy,*W; //global copy of W
	S = (double* ) calloc(MM, sizeof(double));
	s = (double* ) calloc(M, sizeof(double));
	W = (double* ) calloc(MM, sizeof(double));
	dcopy(&MM,Wori,&inci,W,&incj);
	dscal(&MM,&alpha_factor,W,&inci); //MAKE sure alpha_factor is bcasted
//------adaEN Apr. 2013		
if(rank==0)
{	
	Wcopy = (double* ) calloc(MM, sizeof(double));
	 dcopy(&MM,W,&inci,Wcopy,&incj);

	deltaLambda 			= (2*lambda_factor - lambda_factor_prev)*lambda_max;	
	 dscal(&MM,&deltaLambda,Wcopy,&inci); //wcopy = deltaLambda*W
}	
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
	
if(rank==0)
{	
	for(i=0;i<M;i++)
	{
		for(j=0;j<M;j++)
		{
			//W[i,j]
			index = j*M  +i;
// version 1_3: Qij			
			//if(fabs(Q[index])>= Wcopy[index] && i!= j)
			//if(fabs(Q[index] -(1-alpha_factor)*lambda*B[index])>= Wcopy[index] && i!= j)	
			if(fabs(Q[index] -2*(1-alpha_factor)*lambda*B[index]*Wori[index]*Wori[j*M + i])>= Wcopy[index] && i!= j)	
// version 1_3: Qij	
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
	
}
	//MPI::Bcast	
	MPI_Bcast (BL, MM, MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast (S, MM, MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast (s, M, MPI_DOUBLE,0,MPI_COMM_WORLD);


	char transa = 'N'; 
/*	ldk = M;
	//lda = M;
	 dgemv)(&transa, &M, &ldk,&alpha, S, &ldM, eye, &inci, &beta,s, &incj);
*/	
//printMat(W,M,M);
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
	
if(rank==0)
{	
	for(i=0;i<M;i++)
	{
		readPtr = &X[i];
		// dcopy)(&N,readPtr,&M,xi,&inci);
		readPtr2 = &Y[i];
		// dcopy)(&N,readPtr,&M,y_j,&inci);

		//dot product
		//XYi =  ddot)(&N, xi, &inci,y_j, &incj);
		XYi =  ddot(&N, readPtr, &M,readPtr2, &M);
		//XXi =  ddot)(&N, xi, &inci,xi, &incj);
		//norm2 =  dnrm2)(&N,xi,&inci);
		//XXi 	= pow(norm2,2);
		XXi =  ddot(&N, readPtr, &M,readPtr, &M);
		f0[i] 	= XYi/XXi;
		F1ptr	= &F1[M*i];//start from ith column
		//Y*X(i,:)' 		y := alpha*A*x + beta*y, alpha*Y *xi + beta*F1
		alpha = 1/XXi;
		 dgemv(&transa, &M, &N,&alpha, Y, &ldM, readPtr, &M, &beta,F1ptr, &incj);
	}
}
	//MPI::Bcast	
	MPI_Bcast (F1, MM, MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast (f0, M, MPI_DOUBLE,0,MPI_COMM_WORLD);

	
	
	//printMat(f0,M,1);

	// entering loop
	double *IBinv,*zi,*a_iT;// y_j: one row of Y: Nx1
if(rank==0)
{	
	IBinv 	= (double* ) calloc(MM, sizeof(double));
}
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
	
if(rank==0)
{	
	index = M*(M  +1);
	BfOld = (double* ) calloc(index, sizeof(double));
	BfNew = (double* ) calloc(index, sizeof(double));
	BfChange = (double* ) calloc(index, sizeof(double));
}	
	//	//linear system
	//int *ipiv;
	//ipiv = (int *) R_alloc(N,sizeof(int));
	//ipiv = (int *) calloc(M,int);
	//int nrhs = 1;
	//int info = 0;
	//for(i=0;i<M;i++) ipiv[i] = i+1;
	//
	//double *aiTQuad; 
	//aiTQuad 	= (double* ) calloc(N, double);

	//alpha =   dnrm2)(&index,BfOld,&inci);	
//printf("BfOldnorm:%f\n",alpha);	
	//int anyUpdateRow = 0;
	
//MPI:: ------------------------------------------ MESSAGE, BLOCKS
	int low,high,count,MPIiter;
	low = BLOCK_LOW(rank,size,M);
	high = BLOCK_HIGH(rank,size,M);
	count = BLOCK_SIZE(rank,size,M); 	//number of rows to compute
	count = count * M; 					// number of elements to communicate

//MPI loop
	int *receive_counts;
    int *receive_displacements;
	receive_counts = (int *) calloc(size,sizeof(int));
	receive_displacements = (int *) calloc(size,sizeof(int));
	//number of elements for each process	
	MPI_Gather(&count, 1, MPI_INT, receive_counts,1, MPI_INT, 0, MPI_COMM_WORLD);
	receive_displacements[0] = 0;
	for(i=1;i<size;i++)
	{	
		receive_displacements[i] = receive_counts[i-1] + receive_displacements[i-1];
	}
	
	// MPI message vector for for loop
	double *message;
	message = (double* ) calloc(count, sizeof(double));

	//--------------------------mpi::	
//printf("check point 2: processor %d of %d \n", rank,size);	

	
	while(iter < max_iter)
	{
		iter = iter + 1;
		//converge Bfold = [B f];
if(rank==0)
{		
		 dcopy(&MM,B,&inci,BfOld,&incj);
		//last column
		F1ptr = &BfOld[MM];
		 dcopy(&M,f,&inci,F1ptr,&incj);
 }
	//alpha =   dnrm2)(&index,BfOld,&inci);	
//printf("BfOldnorm:%f\n",alpha);
		//printMat(BfOld,M,M+1);
		
		
		




		
		
		
		//BEGIN: MPI parallel computation
		//
		//for(i=0;i<M;i++)
		MPIiter = 0;
		for(i=low;i<=high;i++)
		{
			if(s[i] >0)
			{ 	//
if(rank==0)
{			
				if(verbose>6) printf("\t\t\t\t\t updating gene %d \n",i);
}
				//
				ei[i] = 1;
				//zi   IBinv = I -B
				// dcopy)(&MM,B,&inci,IBinv,&incj);
				//alpha = -1; 
				// dscal)(&MM,&alpha,IBinv,&inci); // dscal(n, a, x, incx) x = a*x
				//diagonal + 1
				//for(j=0;j<M;j++) 
				//{
				//	index = j*M + j;
				//	IBinv[index] = 1 + IBinv[index];
				//}

				//call matrix inverse
				//MatrixInverse(IBinv,M);
				//zi is ith column of IBinv
				//for(j =0;j<M;j++) zi[j] 	= IBinv[i*M+j];
				
				// by linear solver: zi = (I - B)\ei;
				//copy ei to zi;
				zi = &QIBinv[i*M];
				// dcopy)(&M,ei,&inci,zi,&incj);
				//
				//if (i==0) printMat(zi,M,1);
				//
				//call linearSystem(double *a, int N,double *B) a is NxN
//linearSystem(IBinv, M,zi);  //zi is updated.

				// dgesv)(&M, &nrhs, IBinv, &ldM, ipiv, zi, &ldb, &info);
				//anyUpdateRow = 1;
//j reserved				//for j = js_i
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
							//eiB = ei-BiT			//daxpy(n, a, x, incx, y, incy) 		y := a*x + y
							alpha = 1;
							 daxpy(&M, &alpha,BiT, &inci,eiB, &incj);
							//a_iT      = (ei'-BiT)*Y-f(i)*X(i,:);
							readPtr = &X[i];
							 dcopy(&N,readPtr,&M,a_iT,&inci);	

							//a_iT = -f[i]*xi 		dscal(n, a, x, incx) 		x = a*x
							alpha = -f[i];
							 dscal(&N,&alpha,a_iT,&inci);							

							transa='T'; //y := alpha*A**T*x + beta*y, 		 dgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
							beta = 1;
							alpha = 1;
							 dgemv(&transa, &M, &N,&alpha, Y, &ldM, eiB, &inci, &beta,a_iT, &incj);
							
							//r_ij                = y_j'*y_j;	
r_ij =  ddot(&N, y_j, &inci,y_j, &incj);
//------adaEN Apr. 2013	
//r_ij = r_ij + (1 -alpha_factor)*lambdaW*2;
r_ij = r_ij + (1 -alpha_factor)*lambda*Wori[j*M + i]*2*Wori[j*M + i];
//------adaEN Apr. 2013		
							//norm2 =  dnrm2)(&N,y_j,&inci);
							//r_ij 	= pow(norm2,2);
							
							//beta_ij             = a_iT*y_j;
							beta_ij =  ddot(&N, y_j, &inci,a_iT, &incj);
							
							if (fabs(m_ij)<1e-10) //go to the linear equation 
							{
if(rank==0)
{							
								//
								if(verbose>7) printf("\t\t\t\t\t\t\t gene %d \t interact with gene %d.\tLinear equation\n",i,j);
								//
}
								Bij = (beta_ij-lambdaW)/r_ij;
								//printf("\t\t gene (%d,\t%d): linear Bij %f\n",i,j,Bij);
				//
				//if (i==0) 
				//{
				//	printf("\t\t\t beta_ij: %f;\t r_ij:%f; lambdaW: %f\n ",beta_ij,r_ij,lambdaW);
				//	printMat(a_iT,N,1);
					//printMat(y_j,N,1);
				//	printMat(eiB,M,1);
				//}
				//
								if(Bij>0) 
								{
									B[j*M+i] = Bij;//B(i,j)      = Bij;
									//message[i,j]
									message[MPIiter*M + j] = Bij;
									
								}else
								{
									Bij         = (beta_ij+lambdaW)/r_ij;
									if(Bij<0)
									{
										B[j*M+i] = Bij;//B(i,j)      = Bij;
										message[MPIiter*M + j] = Bij;
										
									}else
									{
										B[j*M+i] = 0;
										message[MPIiter*M + j] = 0;
									}
								}//B_ij>0 
							}else //m_ij ~=0 go to the quadratic equation
							{
if(rank==0)
{							
								//
								if(verbose>7) printf("\t\t\t\t\t\t\t gene %d \t interact with gene %d.\tQuadratic equation\n",i,j);
								//
}
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
								//Lss = quadraticLss(sigma2, N, d_ij, candsBij, r_ij, lambdaW,beta_ij);								
								//Lss = quadraticLssJuan(sigma2,N, d_ij, candsBij, a_iT,y_j,lambdaW);
//beta =  ddot)(&N, a_iT, &inci,a_iT, &incj);
								//norm2 =  dnrm2)(&N,a_iT,&inci);
								//beta 	= pow(norm2,2);
							
								//Lss = sigma2*N*log(fabs(d_ij - candsBij)+1e-16) - beta/2 -lambdaW*fabs(candsBij);
								//Lss = sigma2*N*log(fabs(d_ij)+1e-16) - beta/2;
								Lss = sigma2*N*log(fabs(d_ij)+1e-16);
								
								// a_iT = a_iT - candsBij*y_j 		daxpy(n, a, x, incx, y, incy) 	y := a*x + y							
								if (Bijpp>0)
								{
									//LssCands = quadraticLss(sigma2, N, d_ij, Bijpp, r_ij, lambdaW,beta_ij);
									//LssCands = quadraticLssJuan(sigma2,N, d_ij, Bijpp, a_iT,y_j,lambdaW);	
									//aiTQuad dcopy(n, x, incx, y, incy)  y= x
									// dcopy)(&N,a_iT,&inci,aiTQuad,&incj);
									//aiTQuad = aiTQuad - candsBij*y_j  daxpy(n, a, x, incx, y, incy) 	y := a*x + y
									//alpha = -Bijpp;
									// daxpy)(&N, &alpha,y_j, &inci,aiTQuad, &incj);
//beta =  ddot)(&N, aiTQuad, &inci,aiTQuad, &incj);
									//norm2 =  dnrm2)(&N,aiTQuad,&inci);
									//beta 	= pow(norm2,2);
									//LssCands = sigma2*N*log(fabs(d_ij - Bijpp)+1e-16) - beta/2 -lambdaW*fabs(Bijpp);
									LssCands = sigma2*N*log(fabs(d_ij - Bijpp)+1e-16) - r_ij*pow(Bijpp,2)/2 + beta_ij*Bijpp -lambdaW*fabs(Bijpp); 
									
									if(LssCands>Lss) 
									{
										candsBij = Bijpp;
										Lss 	= LssCands;
									}	
								}
								if (Bijpm>0)
								{
									//LssCands = quadraticLss(sigma2, N, d_ij, Bijpm, r_ij, lambdaW,beta_ij);
									//LssCands = quadraticLssJuan(sigma2,N, d_ij, Bijpm, a_iT,y_j,lambdaW);								
									// dcopy)(&N,a_iT,&inci,aiTQuad,&incj);
									//aiTQuad = aiTQuad - candsBij*y_j  daxpy(n, a, x, incx, y, incy) 	y := a*x + y
									//alpha = -Bijpm;
									// daxpy)(&N, &alpha,y_j, &inci,aiTQuad, &incj);
//beta =  ddot)(&N, aiTQuad, &inci,aiTQuad, &incj);
									//norm2 =  dnrm2)(&N,aiTQuad,&inci);
									//beta 	= pow(norm2,2);
									//LssCands = sigma2*N*log(fabs(d_ij - Bijpm)+1e-16) - beta/2 -lambdaW*fabs(Bijpm);
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
									//LssCands = quadraticLss(sigma2, N, d_ij, Bijmm, r_ij, lambdaW,beta_ij);
									//LssCands = quadraticLssJuan(sigma2,N, d_ij, Bijmm, a_iT,y_j,lambdaW);								
									// dcopy)(&N,a_iT,&inci,aiTQuad,&incj);
									//aiTQuad = aiTQuad - candsBij*y_j  daxpy(n, a, x, incx, y, incy) 	y := a*x + y
									//alpha = -Bijmm;
									// daxpy)(&N, &alpha,y_j, &inci,aiTQuad, &incj);
//beta =  ddot)(&N, aiTQuad, &inci,aiTQuad, &incj);
									//norm2 =  dnrm2)(&N,aiTQuad,&inci);
									//beta 	= pow(norm2,2);
									//LssCands = sigma2*N*log(fabs(d_ij - Bijmm)+1e-16) - beta/2 -lambdaW*fabs(Bijmm);
									LssCands = sigma2*N*log(fabs(d_ij - Bijmm)+1e-16) - r_ij*pow(Bijmm,2)/2 + beta_ij*Bijmm -lambdaW*fabs(Bijmm);  
									if(LssCands>Lss) 
									{
										candsBij = Bijmm;
										Lss 	= LssCands;
									}	
								}
								if (Bijmp<0)
								{
									//LssCands = quadraticLss(sigma2, N, d_ij, Bijmp, r_ij, lambdaW,beta_ij);
									//LssCands = quadraticLssJuan(sigma2,N, d_ij, Bijmp, a_iT,y_j,lambdaW);									
									// dcopy)(&N,a_iT,&inci,aiTQuad,&incj);
									//aiTQuad = aiTQuad - candsBij*y_j  daxpy(n, a, x, incx, y, incy) 	y := a*x + y
									//alpha = -Bijmp;
									// daxpy)(&N, &alpha,y_j, &inci,aiTQuad, &incj);
//beta =  ddot)(&N, aiTQuad, &inci,aiTQuad, &incj);
									//norm2 =  dnrm2)(&N,aiTQuad,&inci);
									//beta 	= pow(norm2,2);
									//LssCands = sigma2*N*log(fabs(d_ij - Bijmp)+1e-16) - beta/2 -lambdaW*fabs(Bijmp);			
									LssCands = sigma2*N*log(fabs(d_ij - Bijmp)+1e-16) - r_ij*pow(Bijmp,2)/2 + beta_ij*Bijmp -lambdaW*fabs(Bijmp); 
									if(LssCands>Lss) 
									{
										candsBij = Bijmp;
										Lss 	= LssCands;
									}	
								}
								B[j*M+i] = candsBij;
								message[MPIiter*M + j] = candsBij;
							}//m_ij
						}//if(j!= i)
						dB = B_old - B[j*M +i];
						//update c_ij
						ziDb = 1/(1 + dB*m_ij);
						 dscal(&M,&ziDb,zi,&inci);
						
						//update QIBinv ith column updated; jth row: 
						//readPtr = &QIBinv[j];
						// dscal)(&M,&ziDb,readPtr,&M);
						//QIBinv[j*M + i] = QIBinv[j*M + i]*ziDb;
						// adjugate of QIBinv = ziDb*(QIBinvadj - (-1)^kl) + (-1)^kl
						//for(k=0;k<M;k++) // kth row
						//{
						//	for(l=0;l<M;l++)//lth column
						//	{
						//		if(k!=j && l!=i) 
						///		{
						//			alpha = pow(-1.0,k+l);
						//			QIBinv[l*M+k] = ziDb*(QIBinv[l*M+k] -alpha) + alpha;
						//		}	
						//	}
						//}
						
					}//js_i >0
				}//j = 1:M					
				//f
				//BF1 = B(i,:)*F1(:,i)
				readPtr = &B[i];
				 dcopy(&M,readPtr,&ldM,BiT,&inci);

				F1ptr = &F1[M*i];
				BF1 =  ddot(&M, BiT, &inci,F1ptr, &incj);

				f[i] = f0[i] - BF1;
				
				
				
				//message
				message[MPIiter*M + i] = f[i];
				
				ei[i] = 0; // re-set ei for next i
			}else//s[i]  no un-zero weight in this gene
			{
				readPtr = &B[i];
				// dscal)(&M,&toyZero,readPtr,&ldM);
				 dcopy(&M,&toyZero,&inc0,readPtr,&ldM);
				f[i] = f0[i];
				
				//message : if B is "Bcast" each while loop, the update is not needed
				readPtr= &message[MPIiter*M];
				dcopy(&M,&toyZero,&inc0,readPtr,&inci);
				message[MPIiter*M + i] = f0[i];
				
			} // s[i] iter
			MPIiter = MPIiter  +1;
		}//i= 1:M
		
		MPI_Gatherv(message, count, MPI_DOUBLE, B,receive_counts,receive_displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);		
if(rank==0)
{
	//transpose B
	transposeB(B,M,M);
	//printMat(B,M,M);
	for(i=0;i<M;i++)
	{
		f[i] = B[i*M + i];
		B[i*M + i] = 0;
	}
}
		MPI_Bcast (B, MM, MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Bcast (f, M, MPI_DOUBLE,0,MPI_COMM_WORLD);

		//END: MPI parallel computation
				
		
		
		
if(rank==0)
{		
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
		
//printf("FnormOld: %f; \t FnormChange: %f\n",FnormOld,FnormChange);
//MASTER IS HERE	IBinv will be used in Qij	
		// BLOCK COORDINATE ASCEND: Update IBinv
		UpdateIBinv(QIBinv, B,M);
		
		if(verbose>5) printf("\t\t\t\t\t\tdelta_BF: %f\n",delta_BF);
}
		//MPI 
		MPI_Bcast (&delta_BF, 1, MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Bcast (QIBinv, MM, MPI_DOUBLE,0,MPI_COMM_WORLD);


		if(delta_BF<1e-3)		//break out
		{
			break;
		}
		
		
	}//while

//printf("check point 3: processor %d of %d \n", rank,size);
	//QIBinv: compute not updated ones
	//F77_NAME(dgetrs)(const char* trans, const int* n, const int* nrhs,
	//	 const double* a, const int* lda, const int* ipiv,
	//	 double* b, const int* ldb, int* info);
	//printMat(IBinv,M,M);
	//if(anyUpdateRow>0) //QIBinv --> = Cij/det(I-B) != eye(M); ipiv not zeros
	//{	
	//	transa = 'N';
	//	for(i=0;i<M;i++)
	//	{
	//	//	if(s[i] ==0)
	//		{ 	
	//			info = 0;
	//			ei[i] = 1;
	//			zi = &QIBinv[i*M];
	//			 dcopy)(&M,ei,&inci,zi,&incj);
	//			// dgetrs
	//			 dgetrs)(&transa, &M, &nrhs,IBinv,&M,ipiv,zi,&ldb,&info); //column i updated
	//			ei[i] = 0;
	//		}
	//	}
	//}else
	//{
		//initialize IBinv
	//	 dcopy)(&MM,&beta,&inc0,QIBinv,&incj);
	//	for(i=0;i<M;i++) QIBinv[i*M+i] =1;
	//}
	
	//printMat(QIBinv,M,M);
	
if(rank==0)
{	
	if(verbose>3) printf("\t\t\t\tCurrent lambda: %f;\t number of iteration is: %d.\tExiting Weighted_LassoSf\n\n",lambda, iter);
}	
	//mue          = (IM-B)*meanY-bsxfun(@times,f,meanX);

	//IBinv = I -B
	// dcopy)(&MM,B,&inci,IBinv,&incj);
	//alpha = -1; 
	// dscal)(&MM,&alpha,IBinv,&inci); // dscal(n, a, x, incx) x = a*x
	//diagonal + 1
	//for(j=0;j<M;j++) 
	//{
	//	index = j*M + j;
	//	IBinv[index] = 1 + IBinv[index];
	//	mueL[j] = -f[j]*meanX[j];
	//}
	//mue[i] = -f[i]*meanX[i];
	
	//	//dgemv mue = Ax + beta*mue
	//transa = 'N';
	//alpha = 1;
	//beta = 1;
	//ldk = M;
	// dgemv)(&transa, &M, &ldk,&alpha, IBinv, &ldM, meanY, &inci, &beta,mueL, &incj);

	//----------------------------------------------------------------------------------END OF LASSO
		//SET TO PART2: LASSO with lambda zero
	//double *B, double *f;
	 dcopy(&MM,BL,&inci,BC,&incj);
	 dcopy(&M,fL,&inci,fC,&incj);
	B = &BC[0];
	f = &fC[0];
if(rank ==0)
{	
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
}
	MPI_Bcast (S, MM, MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast (s, M, MPI_DOUBLE,0,MPI_COMM_WORLD);


	beta = 1;
	//f0, F1
	//ei = 0	
	// entering loop
	//loop starts here
	iter = 0;
	//double js_i, m_ij,B_old, beta_ij,r_ij;
	//int *ipiv;
	//ipiv = (int *) R_alloc(N,sizeof(int));
	//info = 0;
	
	//dynamic variable keep intermidiate values eiB = ei-BiT

	//quadratic function
	double theta_ij,k_ij,q_ij,Bijp, Bijm; //case (14)
	
	//converge of gene i
	
	//converge of while
	max_iter = max_iter/5;
	//linear system

	//
	//double *aiTQuad; 
	//aiTQuad 	= (double* ) calloc(N, double);
	//double *zzi;
	//zzi 		= (double* ) calloc(M, double);
	//zi = &zzi[0];
	
//printf("check point 4: processor %d of %d \n", rank,size);		
	
	while(iter < max_iter)
	{
		iter = iter + 1;
		//converge Bfold = [B f];
if(rank==0)
{	
		 dcopy(&MM,B,&inci,BfOld,&incj);
		//last column
		F1ptr = &BfOld[MM];
		 dcopy(&M,f,&inci,F1ptr,&incj);
}
		 //
//	printMat(ei,M,1);
//	printMat(F1,M,M);
		//
		
		// inner loop
		MPIiter = 0;
		//for(i=0;i<M;i++)
		for(i=low;i<=high;i++)		
		{
			if(s[i] >0)
			{
if(rank==0)
{			
				//
				if(verbose>6) printf("\t\t updating gene %d \n",i);
				//
}
			ei[i] = 1;
				//zi
				// dcopy)(&MM,B,&inci,IBinv,&incj);
				//alpha = -1; 
				// dscal)(&MM,&alpha,IBinv,&inci); // dscal(n, a, x, incx) x = a*x
				//diagonal + 1
				//for(j=0;j<M;j++) 
				//{
				//	index = j*M + j;
				//	IBinv[index] = 1 + IBinv[index];
				//}
				//call matrix inverse
				//MatrixInverse(IBinv,M);
				//zi is ith column of IBinv
				//for(j =0;j<M;j++) zi[j] 	= IBinv[i*M+j];
				
				//by linear solver: 
				//copy ei to zi;
				zi = &IBinvZero[i*M];
				// dcopy)(&M,ei,&inci,zi,&incj);
				//call linearSystem(double *a, int N,double *B) a is NxN
				//linearSystem(IBinv, M,zi);  //zi is updated.

//printf("M: %d; irhs: %d; ldM: %d; ldb: %d; info: %d\n",M,nrhs,ldM,ldb,info);
				// dgesv)(&M, &nrhs, IBinv, &ldM, ipiv, zi, &ldb, &info);
								
				//printMat(zi,M,1);
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
						//a_iT      = (ei'-BiT)*Y-f(i)*X(i,:);
						//eiB = ei-BiT			//daxpy(n, a, x, incx, y, incy) 		y := a*x + y
						alpha = 1;
						 daxpy(&M, &alpha,BiT, &inci,eiB, &incj);
						//a_iT      = (ei'-BiT)*Y-f(i)*X(i,:);
						readPtr = &X[i];
						 dcopy(&N,readPtr,&M,a_iT,&inci);
						//a_iT = -f[i]*xi 		dscal(n, a, x, incx) 		x = a*x
						alpha = -f[i];
						 dscal(&N,&alpha,a_iT,&inci);							

						transa='T'; //y := alpha*A**T*x + beta*y, 		 dgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
						//beta = 1;
						alpha = 1;
						 dgemv(&transa, &M, &N,&alpha, Y, &ldM, eiB, &inci, &beta,a_iT, &incj);

						//r_ij                = y_j'*y_j;	
						r_ij =  ddot(&N, y_j, &inci,y_j, &inci);
						//norm2 =  dnrm2)(&N,y_j,&inci);
						//r_ij 	= pow(norm2,2);	
						//beta_ij             = a_iT*y_j;
						beta_ij =  ddot(&N, y_j, &inci,a_iT, &incj);
						
						if (fabs(m_ij)<1e-10) //go to the linear equation 
						{
if(rank==0)
{						
							//
							if(verbose>7) printf("\t\t\t gene %d \t interact with gene %d.\tLinear equation\n",i,j);
							//
}
							//Bij = (beta_ij+lambdaW)/r_ij; 
							B[j*M+i] = beta_ij/r_ij;
							message[MPIiter*M + j] = beta_ij/r_ij;
							
							//printf("\t\t\t beta_ij: %f;\t r_ij:%f\n ",beta_ij,r_ij);
							//printMat(a_iT,M,1);
						}else //m_ij ~=0 go to the quadratic equation
						{
if(rank==0)
{						
							//
							if(verbose>7) printf("\t\t\t gene %d \t interact with gene %d.\tQuadratic equation\n",i,j);
							//					
}						
							//assume Bij >0
							d_ij = 1/m_ij + B[j*M+i];
							theta_ij = r_ij*d_ij + beta_ij;
							k_ij = d_ij*beta_ij - N*sigma2;
								
							q_ij = theta_ij*theta_ij - 4*r_ij* k_ij;
							Bijp = (1/(2*r_ij))*(theta_ij + sqrt(q_ij));
							Bijm = (1/(2*r_ij))*(theta_ij - sqrt(q_ij));
								
							candsBij = 0;
							//Lss = quadraticLss(sigma2, N, d_ij, candsBij, r_ij, lambdaW,beta_ij);
							//Lss = quadraticLssJuan(sigma2,N, d_ij, candsBij, a_iT,y_j,lambdaW);
							//beta =  ddot)(&N, a_iT, &inci,a_iT, &inci);
							//norm2 =  dnrm2)(&N,a_iT,&inci);
							//beta 	= pow(norm2,2);
							//Lss = sigma2*N*log(fabs(d_ij - candsBij)+1e-16) - beta/2 -lambdaW*fabs(candsBij);
							//Lss = sigma2*N*log(fabs(d_ij)+1e-16) - beta/2;															
							Lss = sigma2*N*log(fabs(d_ij)+1e-16);
							//Bijp
							//LssCands = quadraticLss(sigma2, N, d_ij, Bijp, r_ij, lambdaW,beta_ij);
							//LssCands = quadraticLssJuan(sigma2,N, d_ij, Bijp, a_iT,y_j,lambdaW);							
							// dcopy)(&N,a_iT,&inci,aiTQuad,&incj);
							//aiTQuad = aiTQuad - candsBij*y_j  daxpy(n, a, x, incx, y, incy) 	y := a*x + y
							//alpha = -Bijp;
							// daxpy)(&N, &alpha,y_j, &inci,aiTQuad, &incj);
							//beta =  ddot)(&N, aiTQuad, &inci,aiTQuad, &incj);
							//norm2 =  dnrm2)(&N,aiTQuad,&inci);
							//beta 	= pow(norm2,2);							
							//LssCands = sigma2*N*log(fabs(d_ij - Bijp)+1e-16) - beta/2;
							LssCands = sigma2*N*log(fabs(d_ij - Bijp)+1e-16) - r_ij*pow(Bijp,2)/2 + beta_ij*Bijp;
							if(LssCands>Lss) 
							{
								candsBij = Bijp;
								Lss 	= LssCands;
							}	
							//Bijm>
							//LssCands = quadraticLss(sigma2, N, d_ij, Bijm, r_ij, lambdaW,beta_ij);
							//LssCands = quadraticLssJuan(sigma2,N, d_ij, Bijm, a_iT,y_j,lambdaW);							
							// dcopy)(&N,a_iT,&inci,aiTQuad,&incj);
							//aiTQuad = aiTQuad - candsBij*y_j  daxpy(n, a, x, incx, y, incy) 	y := a*x + y
							//alpha = -Bijm;
							// daxpy)(&N, &alpha,y_j, &inci,aiTQuad, &incj);
							//beta =  ddot)(&N, aiTQuad, &inci,aiTQuad, &inci);
							//norm2 =  dnrm2)(&N,aiTQuad,&inci);
							//beta 	= pow(norm2,2);
							//LssCands = sigma2*N*log(fabs(d_ij - Bijm)+1e-16) - beta/2;							
							LssCands = sigma2*N*log(fabs(d_ij - Bijm)+1e-16) - r_ij*pow(Bijm,2)/2 + beta_ij*Bijm;
							if(LssCands>Lss) 
							{
								candsBij = Bijm;
								Lss 	= LssCands;
							}	

							B[j*M+i] = candsBij;
							message[MPIiter*M + j] = candsBij;
							
						}//m_ij

						dB = B_old - B[j*M +i];
						//update c_ij
						ziDb = 1/(1 + dB*m_ij);
						 dscal(&M,&ziDb,zi,&inci);	

						//update IBinvZero ith column updated; jth row: 
						//readPtr = &IBinvZero[j];
						// dscal)(&M,&ziDb,readPtr,&M);
						//IBinvZero[j*M + i] = IBinvZero[j*M + i]*ziDb;
						// adjugate of QIBinv = ziDb*(QIBinvadj - (-1)^kl) + (-1)^kl
						//for(k=0;k<M;k++) // kth row
						//{
						//	for(l=0;l<M;l++)//lth column
						//	{
						//		if(k!=j && l!=i) 
						//		{
						//			alpha = pow(-1.0,k+l);
						//			IBinvZero[l*M+k] = ziDb*(IBinvZero[l*M+k] -alpha) + alpha;
						//		}	
						//	}
						//}


						
					}else //js_i >0
					{
						B[j*M+i] = 0;
						message[MPIiter*M + j] = 0;
					}
					
				}//j = 1:M	
			
				//f
				//BF1 = B(i,:)*F1(:,i)
				readPtr = &B[i];
				 dcopy(&M,readPtr,&ldM,BiT,&inci);
				F1ptr = &F1[M*i];
				BF1 =  ddot(&M, BiT, &inci,F1ptr, &incj);

				f[i] = f0[i] - BF1;
				
				//message
				message[MPIiter*M + i] = f[i];
				
				ei[i] = 0; // re-set ei for next i
				
			}else//[si]
			{
				readPtr = &B[i];
				// dscal)(&M,&toyZero,readPtr,&ldM);
				 dcopy(&M,&toyZero,&inc0,readPtr,&ldM);
				f[i] = f0[i];
				
				//message : if B is "Bcast" each while loop, the update is not needed
				readPtr= &message[MPIiter*M];
				dcopy(&M,&toyZero,&inc0,readPtr,&inci);
				message[MPIiter*M + i] = f0[i];
				
			} //s[i]
			MPIiter = MPIiter  +1;
			
		}//i= 1:M
		MPI_Gatherv(message, count, MPI_DOUBLE, B,receive_counts,receive_displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);		
if(rank==0)
{
	//transpose B
	transposeB(B,M,M);
	//printMat(B,M,M);
	for(i=0;i<M;i++)
	{
		f[i] = B[i*M + i];
		B[i*M + i] = 0;
	}
}
		MPI_Bcast (B, MM, MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Bcast (f, M, MPI_DOUBLE,0,MPI_COMM_WORLD);	
	
		//END: MPI parallel computation

if(rank==0)
{		
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
		
		
		// BLOCK COORDINATE ASCEND: Update IBinv
		UpdateIBinv(IBinvZero, B,M);
}
		//MPI 
		MPI_Bcast (&delta_BF, 1, MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Bcast (IBinvZero, MM, MPI_DOUBLE,0,MPI_COMM_WORLD);
			
		if(delta_BF<1e-2)		//break out
		{
			break;
		}

		

	}//while
		
if(rank==0)
{	
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
	
	
	
	
	//mue                     	= (IM-B)*meanY-bsxfun(@times,f,meanX); diag(f)*meanX

	free(meanY);
	free(meanX);
	free(Wcopy);
	free(IBinv);	
	free(BfOld);
	free(BfNew);
	free(BfChange);
}
	free(Y);
	free(X);
	
	free(S);
	free(s);
	free(f0);
	free(F1);

	
	//free(xi);
	free(y_j);
	
	free(ei);

	//free(zzi);
	free(a_iT);
	
	free(eiB);
	free(BiT);

	
	//free(ipiv);
	
	//free(aiTQuad);
	//free(eye);
	
	
	free(message);
	free(receive_counts);
	free(receive_displacements);
	//------adaEN Apr. 2013	
	free(W);
	//------adaEN Apr. 2013		
	return lambda;
	//sigma2 remains the same
}//weighted_LassoSf

//--------------------------------------------------------------------------------END WEIGHTED_LASSOSF

//----------------------------------------------- TEST FUNCTION
int cv_gene_nets_support_adaEN(double *Y, double *X, int Kcv,double *lambda_factors, double *rho_factors, 
			int maxiter, int M, int N,int Nlambdas, int Nrho,int verbose,double *W, 			//double sigma2learnt,
			double *sigmaLasso,int rank, int size,		//,double *IBinv
			int i_alpha, double alpha_factor,double * ErrorEN,double *sigma2learnt_EN)		
//------adaEN Apr. 2013	
// if i_alpha ==0: calculate W
// keep record of ErrorEN = errs_mean_min (the error in selecting lambda)
// i_alpha goes global, like the function call of cv_gene_nets_support_adaEN
//------adaEN Apr. 2013				
{	//sigma2_ms ilambda_ms
	// 
	// return sigmaLASSO = sigma_RidgeCVf for lasso in main
	//return ilambda_cv_ms: index of lambda_factors
	//0. lambda(i) path:
	//1. 	cross validation by ridge_cvf: rho_factor 		--> ONLY  for ilambda = 1; save Kcv set;
	//2. 	constrained_ridge_eff: weights W
	//3. 	weighted_lassoSf: none-zeros with lambda(i)
	//4. 	constrained_MLf: likelihood
	//5. return lambda
	int MM = M*M;
	int MN = M*N;
	// to save for each lambda
	double *Q, *BL, *fL,*mueL; 
	int i,j,index;
	double *BC, *fC;	
	
	int Ntest = N/Kcv; 		//C takes floor
	int Nlearn = N - Ntest; 
	double * Errs, *Sigmas2, *ErrorMean, *ErrorCV;
	
	double *Ylearn, *Xlearn, *Ytest,*Xtest;
	int NlearnM = Nlearn*M;
	int NtestM = Ntest*M;
	double *Ylearn_centered, *Xlearn_centered, *meanY, *meanX;
	double *SL;				//zero or not-zero
	
	//lasso 	MPI
	BL = (double* ) calloc(MM, sizeof(double));//ptr BL
	fL = (double* ) calloc(M, sizeof(double)); //ptr fL	
	BC = (double* ) calloc(MM, sizeof(double));
	fC = (double* ) calloc(M, sizeof(double));
	
if(rank==0)
{	
	Q = (double* ) calloc(MM, sizeof(double)); //ptr Q
	mueL = (double* ) calloc(M, sizeof(double));
	//
	if(Nrho>Nlambdas)
	{
		i = Nrho*Kcv;
	}else
	{
		i= Nlambdas*Kcv;
	}
	
	Errs = (double* ) calloc(i, sizeof(double));
	Sigmas2 = (double* ) calloc(Nlambdas, sizeof(double));
	ErrorMean= (double* ) calloc(Nlambdas, sizeof(double));
	i = Nlambdas*Kcv;
	ErrorCV = (double* ) calloc(i, sizeof(double));
	
	//parameters inside the loop

	Ylearn = (double* ) calloc(NlearnM, sizeof(double));
	Xlearn = (double* ) calloc(NlearnM, sizeof(double));
	Ytest = (double* ) calloc(NtestM, sizeof(double));
	Xtest = (double* ) calloc(NtestM, sizeof(double));

	//centerYX function

	Ylearn_centered = (double* ) calloc(NlearnM, sizeof(double));
	Xlearn_centered = (double* ) calloc(NlearnM, sizeof(double));
	meanY = (double* ) calloc(M, sizeof(double));
	meanX = (double* ) calloc(M, sizeof(double));
	//SL
	SL = (double* ) calloc(MM, sizeof(double));
}

	//main loop
	double err_mean;
	//initialize
	err_mean = 1e5;

	int ilambda, cv,testStart, testEnd;
	//min_attained,min_attained = 0;
	ilambda = 0;
	
	//Weighted_LassoSf
	double lambda,lambda_factor,lambda_factor_prev;
	lambda_factor_prev = 1;


//	double sigma2Ridge; 	//check which one is better: sigma2Ridge of sigma_hat(lambda)



	//convergence 
if(rank==0)
{	
	if(verbose>1) printf("\t\tEnter Function: cv_support. Nlambdas: %d; \t %d-fold cross validation.\n", Nlambdas,Kcv);
	if(verbose>4) printf("\n\t\t\t\t\tEnter Function: ridge_cvf. %d-fold cross validation.\n", Kcv);
	
}	
	int inci = 1;
	int incj = 1;
	int inc0 = 0;
	int lda,ldb,ldc,ldk;
	double *Xptr, *XsubPtr; //submatrix
	double alpha, beta;
//printf("Initialized.\n");		


//-------------------------------------------------------------ridge cv
	double sigma2learnt; // sigma of ridge regression for lasso_cv_support
	int irho = 0;
	int min_attained = 0;
	double err_mean_prev = 0;
	double sigma2R,i_rho_factor;
	double *BR, *fR, *mueR;
	BR =&BL[0];
	fR = &fL[0];
	mueR= &mueL[0];//BORROWED; no return: reset in lambda CV
	
	//testNoise, ImB: shared with Ridge & Lasso CV
	double *testNoise,*ImB,*NOISE;
if(rank==0)
{	
	NOISE =(double* ) calloc(MN, sizeof(double));
	//testNoise =(double* ) calloc(NtestM, double);
	testNoise = &NOISE[0];
	ImB = (double* ) calloc(MM, sizeof(double));
}
	char transa = 'N'; 
	char transb = 'N';
	lda = M;
	ldb = M;
	ldc = M;
	ldk = M;
	double testNorm;
	//result of CV 
	double rho_factor; // no pointers needed
	// version 14beta: 
	
	
//------adaEN Apr. 2013
//sigma2R is not used: here: used in selection path
//sigma2learnt is the residual need to be reused;

	if(i_alpha ==0)	
	{	
		//cross validation
		for(cv=0;cv<Kcv;cv++)
		{
if(rank==0) 	//create Xtest, Xlearn, Ytest, Ylearn;
{		
			if(verbose>5) printf("\t\t\t\t\t\t crossValidation %d/Kcv\n\n",cv);
			// start and end point
			testStart = Ntest*cv + 1;
			testEnd = Ntest*(cv+1);
//printf("\t\t\t testStart: %d \t End %d\n",testStart,testEnd);
			//assign submatrices
			//SubMatrix(X,Xtest,Xlearn,M,N,testStart,testEnd);
			Xptr = &X[(testStart-1)*M];//index
			i = (testEnd - testStart + 1)*M;//length
			 dcopy(&i,Xptr,&inci,Xtest,&incj);
			if(testStart ==1)
			{
				Xptr = &X[testEnd*M]; //index
				i = (N - testEnd)*M;//length
				 dcopy(&i,Xptr,&inci,Xlearn,&incj);
			}else if(testEnd!=N) //two segments
			{
				i = (testStart-1)*M;//length
				 dcopy(&i,X,&inci,Xlearn,&incj);
				j = (N - testEnd)*M;//length
				Xptr = &X[testEnd*M]; //index
				XsubPtr = &Xlearn[i];//index
				 dcopy(&j,Xptr,&inci,XsubPtr,&incj);
			}else // 1- start is learn
			{
				i = (testStart-1)*M;//length
				 dcopy(&i,X,&inci,Xlearn,&incj);
			}
			
			//SubMatrix(Y,Ytest,Ylearn,M,N,testStart,testEnd);
			Xptr = &Y[(testStart-1)*M];//index
			i = (testEnd - testStart + 1)*M;//length
			 dcopy(&i,Xptr,&inci,Ytest,&incj);
			if(testStart ==1)
			{
				Xptr = &Y[testEnd*M]; //index
				i = (N - testEnd)*M;//length
				 dcopy(&i,Xptr,&inci,Ylearn,&incj);
			}else if(testEnd!=N) //two segments
			{
				i = (testStart-1)*M;//length
				 dcopy(&i,Y,&inci,Ylearn,&incj);
				j = (N - testEnd)*M;//length
				Xptr = &Y[testEnd*M]; //index
				XsubPtr = &Ylearn[i];//index
				 dcopy(&j,Xptr,&inci,XsubPtr,&incj);
			}else // 1- start is learn
			{
				i = (testStart-1)*M;//length
				 dcopy(&i,Y,&inci,Ylearn,&incj);
			}
}			
			//version 14

			//SubMatrixInt(Missing,Missing_test,Missing_learn,M,N,testStart,testEnd);
		
			
			// ridge SEM		sigma2R =	
			 constrained_ridge_cffSVD(Ylearn, Xlearn, rho_factors, M, Nlearn,
						verbose,rank,size,Ytest,Xtest,Errs,cv, Nrho,Ntest,mueR);
			//on return, Errs[:,cv] is written.

		}//cv = 0: Kcv
		//err_mean = err_mean/Kcv;   	calculate sum instead

		//find minimum error

if(rank==0)
{

printMat(Errs,Nrho,Kcv);


	err_mean_prev = 1e10;
	for(irho=0;irho<Nrho;irho++)
	{
		Xptr = &Errs[irho]; 
		err_mean =  dasum(&Kcv, Xptr, &Nrho);

		printf("\t\t\t\t\t irho %d/(Nrho) errorMean: %f\n",irho,err_mean);
		if(verbose>5) printf("\t\t\t\t\t irho %d/(Nrho) errorMean: %f\n",irho,err_mean);		
		if(err_mean<err_mean_prev)
		{
			err_mean_prev 	= err_mean;
			min_attained 	= irho;		
		}
	}
	sigma2R = err_mean_prev/(MN -1);
	if(verbose>2) printf("optimal irho index: %d/(Nrho) errorMean: %f\n",min_attained,err_mean_prev);		
 	if(verbose==0) printf("Step 2: ridge CV; find rho : %d\n", min_attained);
}

	irho = min_attained;
	//MPI::
	MPI_Bcast (&irho, 1, MPI_INT,0,MPI_COMM_WORLD);
	
	
	//sum(sum(1-Missing))
	//int Nmiss = 0;
	//for(i=0;i<MN;i++) Nmiss = Nmiss + Missing[i];

	
	//int Npresent = MN- Nmiss;

	//rho_factor_m            = rho_factors(irho)*N/(N-Ntest);
	rho_factor = rho_factors[irho]*N/(N-Ntest);
	MPI_Bcast (&rho_factor, 1, MPI_DOUBLE,0,MPI_COMM_WORLD);

//if(rank==0)
//{	
//	if(verbose>4) printf("sigma2learnt: %f\n",sigma2R);	
	//sigma2R is for LASSO CV;
	//rho_factor_m: for ridge regression: use the whole dataset
//	if(verbose==0) printf("Step 2: ridge CV; find rho : %f\n", rho_factor);
//}

	
	sigma2learnt = constrained_ridge_cff(Y, X, rho_factor, M, N,BR,fR,mueR,verbose,rank,size);	

	
	
	
	
	
	if(rank==0)
{
	if(verbose==0) printf("Step 3: ridge; calculate weights.\n");

	// weight W: for CV(this function) and selection path (Main)
	for(i=0;i<MM;i++) W[i] = 1/fabs(BL[i]+ 1e-10);
	//printMat(W,M,1);
}	
	//sigma2learnt = sigma2cr; //for lambda CV
	//sigma2R is used for path in Main
	sigmaLasso[0] = sigma2R;
	sigma2learnt_EN[0] = sigma2learnt;
	}else	//i_alpha ==0
	{
		sigma2R 		= sigmaLasso[0];
		sigma2learnt 	= sigma2learnt_EN[0];
	}
//------adaEN Apr. 2013	
//------------------------------------------------------------ridge cv end return sigma2R, weight	

MPI_Bcast (&sigma2learnt, 1, MPI_DOUBLE,0,MPI_COMM_WORLD);
MPI_Bcast (W, MM, MPI_DOUBLE,0,MPI_COMM_WORLD);		//allocated from mainSML
//------------------------------------------------------------------------------------------------
	int ilambda_ms =0;
	double *IBinv,*IBinvZero,*lambda_Max;
	double *IBinvPath,*IBinvPathZero;
	double lambda_max_cv;

	double *errorMeanCpy;
	double minimumErr;
	
	irho  = MM*Kcv;
	IBinvPath = (double* ) calloc(irho, sizeof(double));
	IBinvPathZero = (double* ) calloc(irho, sizeof(double));	
	
if(rank==0)
{

	//IBinv: update in path


	lambda_Max = (double* ) calloc(Kcv, sizeof(double));
	
	beta = 0;
	//initialized
	
	for(cv=0;cv<Kcv;cv++)
	{
		IBinv = &IBinvPath[cv*MM];
		 dcopy(&MM,&beta,&inc0,IBinv,&incj);
		for(i=0;i<M;i++) IBinv[i*M+i] =1;
	}
	 dcopy(&irho,IBinvPath,&inci,IBinvPathZero,&incj);
}


MPI_Barrier (MPI_COMM_WORLD);



	while(ilambda < Nlambdas)
	{	
		//ilambda = ilambda  +1;
		//err_mean_prev = err_mean;
		//err_sigma_prev = err_sigma;
		err_mean = 0;
		for(cv=0;cv<Kcv;cv++)
		{
if(rank==0)
{		
			if(verbose>3) printf("\t\t %d/Kcv cross validation.\n", cv);
			// test, learn
			// start and end point
			testStart = Ntest*cv + 1;
			testEnd = Ntest*(cv+1);
			//assign submatrices
			//SubMatrix(X,Xtest,Xlearn,M,N,testStart,testEnd);
			//SubMatrix(Y,Ytest,Ylearn,M,N,testStart,testEnd);
			//SubMatrixInt(Missing,Missing_test,Missing_learn,M,N,testStart,testEnd);
			//SubMatrix(X,Xtest,Xlearn,M,N,testStart,testEnd);
			Xptr = &X[(testStart-1)*M];//index
			i = (testEnd - testStart + 1)*M;//length
			 dcopy(&i,Xptr,&inci,Xtest,&incj);
			if(testStart ==1)
			{
				Xptr = &X[testEnd*M]; //index
				i = (N - testEnd)*M;//length
				 dcopy(&i,Xptr,&inci,Xlearn,&incj);
			}else if(testEnd!=N) //two segments
			{
				i = (testStart-1)*M;//length
				 dcopy(&i,X,&inci,Xlearn,&incj);
				j = (N - testEnd)*M;//length
				Xptr = &X[testEnd*M]; //index
				XsubPtr = &Xlearn[i];//index
				 dcopy(&j,Xptr,&inci,XsubPtr,&incj);
			}else // 1- start is learn
			{
				i = (testStart-1)*M;//length
				 dcopy(&i,X,&inci,Xlearn,&incj);
			}
			
			//SubMatrix(Y,Ytest,Ylearn,M,N,testStart,testEnd);
			Xptr = &Y[(testStart-1)*M];//index
			i = (testEnd - testStart + 1)*M;//length
			 dcopy(&i,Xptr,&inci,Ytest,&incj);
			if(testStart ==1)
			{
				Xptr = &Y[testEnd*M]; //index
				i = (N - testEnd)*M;//length
				 dcopy(&i,Xptr,&inci,Ylearn,&incj);
			}else if(testEnd!=N) //two segments
			{
				i = (testStart-1)*M;//length
				 dcopy(&i,Y,&inci,Ylearn,&incj);
				j = (N - testEnd)*M;//length
				Xptr = &Y[testEnd*M]; //index
				XsubPtr = &Ylearn[i];//index
				 dcopy(&j,Xptr,&inci,XsubPtr,&incj);
			}else // 1- start is learn
			{
				i = (testStart-1)*M;//length
				 dcopy(&i,Y,&inci,Ylearn,&incj);
			}
			
			
			//SubMatrixInt(Missing,Missing_test,Missing_learn,M,N,testStart,testEnd);
			//Missing_test = &Missing[(testStart-1)*M];
			//Learn matrix
			
			//Ylearn_centered
			 dcopy(&NlearnM,Xlearn,&inci,Xlearn_centered,&incj);
			 dcopy(&NlearnM,Ylearn,&inci,Ylearn_centered,&incj);

			centerYX(Ylearn_centered,Xlearn_centered,meanY, meanX,M, Nlearn);
			//first ilambda

//printf("Initialized.\n");			
			if(ilambda == 0)
			{
				//if(verbose>3) printf("\t\t\t step 0 for first lambda: Ridge cross validation; \tRidge weights.\n");

				//for(i=0;i<M;i++) fL[i] = 1;
				alpha = 1; 
				 dcopy(&M,&alpha,&inc0,fL,&incj); // call dcopy(n, x, inci, y, incy)
				alpha = 0;
				 dcopy(&MM,&alpha,&inc0,BL,&incj); 
				// dscal)(&MM,&alpha,BL,&inci);
				// Q(lambda)S4
				QlambdaStart(Ylearn_centered,Xlearn_centered, Q, sigma2learnt,M, Nlearn);
				// set BL, fL to  zeros: not necessary; will test this after Debug
			
				lambda_Max[cv]	= lambdaMax_adaEN(Ylearn_centered,Xlearn_centered,W,M, Nlearn,alpha_factor);
			}//ilambda ==0
			lambda_max_cv = lambda_Max[cv];
}			


			//MPI::Bcast
			MPI_Bcast (&lambda_max_cv, 1, MPI_DOUBLE,0,MPI_COMM_WORLD);
	
			
			
			//if(ilambda > 0) sigma2learnt = Sigmas2[cv*Nlambdas];//in Juan: 5 sigma2learnt saved: one for each fold
			//Weighted_LassoSf
			lambda_factor = lambda_factors[ilambda];
			//SIGMA2[0] = sigma2learnt;
			
			//lambda = Weighted_LassoSf(W, BL, fL, Ylearn, Xlearn, Q, lambda_factor,
			//				lambda_factor_prev, sigma2learnt, maxiter, M, Nlearn, verbose,IBinv);

			
			IBinv = &IBinvPath[cv*MM];
			IBinvZero= &IBinvPathZero[cv*MM];
//MPI_Barrier (MPI_COMM_WORLD);

			lambda = Weighted_LassoSf_MLf_adaEN(W, BL, fL, Ylearn,Xlearn, Q, lambda_factor, 
							lambda_factor_prev, sigma2learnt, maxiter,M, Nlearn, verbose,
							BC, fC, mueL,IBinv,IBinvZero,lambda_max_cv,rank,size,
							alpha_factor);
//if(ilambda==0 && cv==0)
//{
//	printf("In cv support: \n");
//	printMat(IBinv,M,M);
//	printMat(fL,1,M);
//printMat(BL,M,M);
	//printMat(fC,1,M);
	//printMat(mueL,1,M);
//}

if(rank==0)
{							
			if(verbose>3) printf("\t\t\t step 1 SML lasso regression, lambda: %f.\n",lambda);
			//sign of B SL		//SL int:	
			// dscal)(&MM,&alpha,SL,&inci);

			//Q(lambda)
			//QlambdaMiddle(Ylearn,Xlearn, Q,BL,fL, mueL, sigma2learnt,M, Nlearn);
			QlambdaMiddleCenter(Ylearn_centered,Xlearn_centered, Q,BL,fL,sigma2learnt,M, Nlearn,IBinv);

//if(ilambda==0 && cv==0)
//{
	//printf("In cv support: \n");
	//printMat(Q,M,M);
//	printMat(fL,1,M);
	//printMat(BC,M,M);
	//printMat(fC,1,M);
	//printMat(mueL,1,M);
//}			
			
			// constrained_MLf
			if(verbose>3) printf("\t\t\t step 2 SML ZeroRegression.\n");
			

			//constrained_MLf(BC, fC, SL, Ylearn,Xlearn, sigma2learnt, maxiter, mueL,M, Nlearn, verbose);

			//noise error Errs[ilambda, cv]
			
			//error function: norm((1-Missing).*(A*Y-fC*X)-mueC),'fro')^2;
//Problem in JUAN'S CODE: FOR CV mean: need to normalize, ow. it is not a standard error.
//Errs[index] = errorFunc(BC,Ytest, fC, Xtest, mueL, M, Ntest);
			
			 dcopy(&MM,BC,&inci,ImB,&incj);
			alpha = -1;
			 dscal(&MM,&alpha,ImB,&inci);
			for(i=0;i<M;i++) 
			{
				index = i*M + i;
				ImB[index] = 1 + ImB[index];
			} //I-BR
			alpha = 1; 
			beta = 0;
			testNoise = &NOISE[NtestM*cv];
			 dgemm(&transa, &transb,&M, &Ntest, &ldk,&alpha, ImB, &lda, Ytest, &ldb, &beta, testNoise, &ldc);
			for(i=0;i<M;i++)
			{
				// row i of X
				Xptr = &Xtest[i];
				//XsubPtr = &testNoise[i];
				XsubPtr = &NOISE[NtestM*cv+i];
				// dcopy)(&N,readPtr,&M,xi,&inci);
				//row i of noise ldM
				alpha = -fC[i];
				 daxpy(&Ntest, &alpha,Xptr, &M,XsubPtr, &ldk);
				//NOISE[i,1:N]	
			}//row i = 1:M
			
			// y = ax + y   daxpy)(&MNtest, &alpha,Xptr, &inci,XsubPtr, &incj);
			alpha = -1;
			for(i=0;i<Ntest;i++)
			{
				//Xptr = &testNoise[i*M];
				Xptr = &NOISE[NtestM*cv+i*M];
				 daxpy(&M, &alpha,mueL, &inci,Xptr, &incj);
			}

			//testNorm = FrobeniusNorm(testNoise, M, N);
			// TEST_NOISE computed;
			
			
			testNorm =  ddot(&NtestM, testNoise, &inci,testNoise, &incj);
			//testNorm =  dnrm2)(&NtestM,testNoise,&inci); //just dotproduct will work
			//testNorm = testNorm*testNorm;
			index = cv*Nlambdas+ilambda;
			
			Errs[index] = testNorm;
	
	
//			Sigmas2[index] = sigma2learnt;
			ErrorCV[index] = Errs[index]/NtestM;
			err_mean = err_mean + Errs[index];
			if(verbose>3) printf("\t\t\t cv: %d \tend; err_mean = %f.\n", cv, Errs[index]);
}				
		}//cv
		
if(rank==0)
{		
		//err
		err_mean = err_mean/MN;
		ErrorMean[ilambda] = err_mean;
		
		//mean, sigma
		for(i=0;i<MN;i++)
		{
			NOISE[i] = pow(NOISE[i],2);
		}
		alpha = -1;
		 daxpy(&MN,&alpha,&err_mean,&inc0,NOISE,&inci);
		testNorm =  dnrm2(&MN,NOISE,&inci);
		Sigmas2[ilambda] = testNorm/sqrt(Kcv*(MN -1));
		
		
/*		if(minimumErr>err_mean) 
		{
			minimumErr = err_mean;
			ilambda_min = ilambda;
		}
*/		//check for convergence
		//ilambda = ilambda + 1; //don't move this line
		//if(ilambda>3 && min_attained==0)
		//{
		//	if ((err_mean_prev + err_sigma_prev) < err_mean)
		//	{
		//		min_attained = 1;
		//		ilambda_min = ilambda; //number; not for indexing
		//	}
		//}
		lambda_factor_prev = lambda_factor;
		
		if(verbose>2) printf("\t\t\t %d/Nlambdas. %d fold cv; \t Err_Mean: %f; std:%f; \t sigma2learnt:%f.\n", ilambda,Kcv,err_mean,Sigmas2[ilambda],sigma2learnt);
}
		ilambda = ilambda + 1; 
		
	}//while ilambda

	// select ilambda_ms
	
	// step1. calculate sigma2
	//ErrorCV = ErrorCV - ErrorMean
//printMat(Sigmas2,Nlambdas,1);	
	// step2. index of minimal ErrorMean
	//double *errorMeanCpy;
	
if(rank==0)	
{
	errorMeanCpy = (double* ) calloc(Nlambdas, sizeof(double));
	//int inc0  = 0;
	minimumErr = -1e5 - MN;
	 dcopy(&Nlambdas,&minimumErr,&inc0,errorMeanCpy,&inci);
	alpha = 1;
	 daxpy(&Nlambdas,&alpha,ErrorMean,&inci,errorMeanCpy,&incj); // y = ax + y
	
	ilambda_ms =  idamax(&Nlambdas, errorMeanCpy, &inci);//index of the max(errs_mean)<--min(mean)
	index = ilambda_ms - 1;
	minimumErr = ErrorMean[index] + Sigmas2[index]; //actually max
//printMat(ErrorMean,1,Nlambdas);
//printMat(Sigmas2,1, Nlambdas);

//------adaEN Apr. 2013	
	//double lowBound;
	for(i=index-1;i>0;i--) 
	{
		if(ErrorMean[i] < minimumErr)
		{
			ilambda_ms = i + 1;
		}else
		{
			break;
		}
	}
	
	index = ilambda_ms - 1;
	ErrorEN[i_alpha] = ErrorMean[index];
//------adaEN Apr. 2013		
	
	
	//return index of lambda_factors
	if(verbose>1) printf("\t\tExit Function: cv_support. optimal lambda index: %d.\n\n", ilambda_ms);

//	free(Ws);
	free(NOISE);
	free(ImB);
	
	
	free(Q);

	free(mueL);

	
	free(Errs);
	free(Sigmas2);
	free(ErrorMean);
	free(ErrorCV);
	free(errorMeanCpy);
	
	free(Ylearn);
	free(Xlearn);
	free(Ytest);
	free(Xtest);	

	//free(Missing_learn);
	//free(Missing_test);
	free(Ylearn_centered);
	free(Xlearn_centered);	
	free(meanY);
	free(meanX);
	free(SL);
//	free(errs_mean);	
//	free(errs_sigma);	

//	free(rho_factor_ptr);


	free(IBinvPath);
	free(IBinvPathZero);
	
	free(lambda_Max);

	
}
	free(BL);
	free(fL);
	free(BC);
	free(fC);
	
	//MPI_Barrier (MPI_COMM_WORLD);
	MPI_Bcast (&ilambda_ms, 1, MPI_DOUBLE,0,MPI_COMM_WORLD);
	return ilambda_ms;
	//index + 1 -->position; return position
	
	
	
} //end of cv_gene_nets_support		

			
			
			
//cv_gene_nets_support: return ilambda_cv_ms Missing


// ---------------------------- Main function that calls by R (interface)
//-----------------------------main function calls:
// centerYX
// cv_gene_nets_support
// ridge_cvf
// constraint - ridge_cff
// weighted_lassoSf


//--------------------------------not elastic at this moment
void mainSML_adaEN(double *Y, double *X, int *m, int *n, int *Missing,double*B, double *f,double*stat,int*VB,int rank, int size)
{
	//stat contains: correct postive, true positve; false postive, positve detected; power; fdr 6x1
	// assume B is the true imput
	int M, N, i, j,index,verbose;
	int inci = 1;
	int incj = 1;
	int inc0 = 0;
	M 			= m[0];
	N 			= n[0];	
	verbose 	= VB[0];
	int MN 		= M*N;
	int MM 		= M*M;
	double *Strue;
	double alpha = 1;	
if(rank==0)
{	
	Strue 		= (double* ) calloc(MM, sizeof(double));
	 dcopy(&MM,B,&inci,Strue,&incj);
	stat[1] 	= 0;
	for(i=0;i<M;i++)
	{
		for(j=0;j<M;j++)
		{
			index = j*M  +i;
			//Strue[index] = B[index];
			if(i!=j && B[index]!=0)
			{	stat[1] = stat[1] + 1;} //stat[1] total positive
		}
	}
	
	 dcopy(&M,&alpha,&inc0,f,&inci); 		//initialize f
	//for(i=0;i<M;i++) f[i] = 1;

	alpha = 0;
	// dscal)(&MM,&alpha,B,&inci);
	 dcopy(&MM,&alpha,&inc0,B,&inci);		//initialize B
	//assume missing values are from X		//set missing counterpart in Y to zero

	for(i=0;i<MN;i++)
	{
		if(Missing[i] == 1) Y[i] = 0;
	}
}	

	//call cv_gene_nets_support ------------------------SYSTEM PARAMETERS
	int maxiter 	= 500;
	int Kcv 		= 5;
	int L_lambda 	= 20; // number of lambdas in lambda_factors	stop at 0.001
	double *lambda_factors;
	double step 	= -0.2;

	lambda_factors 	= (double* ) calloc(L_lambda, sizeof(double));
	for(i=0;i<L_lambda;i++) 
	{
		lambda_factors[i] 	= pow(10.0,step);
		step 				= step - 0.2;
	}
	
	//rho factor
	step 					= -6;
	double * rho_factors;
	int L 					= 31; // number of rho_factors
	
	rho_factors 			= (double* ) calloc(L, sizeof(double));
	for(i=0;i<L;i++) 
	{
		rho_factors[i] 		= pow(10.0,step);
		step 				= step + 0.2;
	}
//------adaEN Apr. 2013	
	//adaEN parameter
	double *alpha_factors, *ErrorEN,*lambdaEN, sigma2learnt;
	int L_alpha  		= 19; 
	alpha_factors 		= (double* ) calloc(L_alpha, sizeof(double)); //variable in all ranks
	ErrorEN 			= (double* ) calloc(L_alpha, sizeof(double));//rank0 variable actually.
	lambdaEN 			= (double* ) calloc(L_alpha, sizeof(double));
	
	step 					= 0.05;
	for(i=0;i<L_alpha;i++) 
	{
		alpha_factors[i]	= 0.95 - step*i;
	}

//------adaEN Apr. 2013	
	//---------------------------------------------END  SYSTEM PARAMETERS
	int Nlambdas,Nrho;
	Nlambdas 				= L_lambda;
	Nrho 					= L;
	
	
	//Nlambdas 				= 1;
	//Nrho 					= L;
	
	
	
	
	
	
	
//call ridge_cvf
	double sigma2; //return value;

	//double *mueL;
	//mueL = (double* ) calloc(M, double); 

	// weight W
	double *W,*QIBinv;  //weight on diagonal?
	

	double beta = 0;
	
	// IBinv: SAVE COMPUTATION: get IBinv from lasso: 1) CV_support; 2) selection path
	W = (double* ) calloc(MM, sizeof(double));
	//IBinv = (double* ) calloc(MM, double);
	QIBinv = (double* ) calloc(MM, sizeof(double));
	
if(rank==0)
{
	 dcopy(&MM,&beta,&inc0,QIBinv,&incj);
	for(i=0;i<M;i++) QIBinv[i*M+i] =1;
}		
	//
	int ilambda_cv_ms; //index + 1
	
	//ilambda_cv_ms = cv_gene_nets_support(Y, X, Kcv,lambda_factors, 
	//				rho_factors, maxiter, M, N,Nlambdas, Nrho,verbose,W,sigma2cr);
//------adaEN Apr. 2013		

	int i_alpha;
	double alpha_factor;
	for(i_alpha=0;i_alpha<L_alpha;i_alpha++)
	{	
		alpha_factor 		= alpha_factors[i_alpha];
		ilambda_cv_ms = cv_gene_nets_support_adaEN(Y, X, Kcv,lambda_factors, rho_factors, 
			maxiter, M, N,Nlambdas, Nrho,verbose,W, &sigma2, rank,size,
			i_alpha,alpha_factor,ErrorEN, &sigma2learnt);	
			
		lambdaEN[i_alpha] 	= ilambda_cv_ms;
	}		
	
	//find the min of ErrorEN;
	double minEN 			= ErrorEN[0];
	int ind_minEN 			= 0;
if(rank==0)
{	
	for(i_alpha = 1;i_alpha<L_alpha;i_alpha++)
	{
		if(minEN > ErrorEN[i_alpha])
		{
			minEN 			= ErrorEN[i_alpha];
			ind_minEN 		= i_alpha;
		}	
	}
	//sigma2 					= sigmaEN[ind_minEN];
	ilambda_cv_ms 			= lambdaEN[ind_minEN];
	alpha_factor  			= alpha_factors[ind_minEN];
	printf("\tAdaptive_EN %d-fold CV, alpha: %f.\n", Kcv,alpha_factor);
	printMat(ErrorEN,L_alpha,1);
}	
	MPI_Bcast (&alpha_factor, 1, MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast (&ilambda_cv_ms, 1, MPI_INT,0,MPI_COMM_WORLD);
//------adaEN Apr. 2013	

	//ilambda_cv_ms = 17;
	//sigma2 = 0.0156827270598132;
	//step = 1;
	// dcopy)(&MM,&step,&inc0,W,&inci);
	
	double *meanY, *meanX, *Ycopy, *Xcopy,*Q;	
	double lambda_factor_prev = 1.0;
	double lambda_factor;
	int ilambda;
	double lambda;
	double lambda_max;
if(rank== 0)
{	
	if(verbose==0) printf("Step 1: CV support; return number of lambda needed: %d\n", ilambda_cv_ms);
	
	//call centerYX;

	meanY = (double* ) calloc(M, sizeof(double));
	meanX = (double* ) calloc(M, sizeof(double));
	Ycopy = (double* ) calloc(MN, sizeof(double));
	Xcopy = (double* ) calloc(MN, sizeof(double));
	
	
	//copy Y,X
	 dcopy(&MN,X,&inci,Xcopy,&incj);
	 dcopy(&MN,Y,&inci,Ycopy,&incj);
	
	centerYX(Ycopy,Xcopy, meanY, meanX,M, N);
	// call Q_start

	Q = (double* ) calloc(MM, sizeof(double));
	QlambdaStart(Ycopy,Xcopy, Q, sigma2, M, N);
	
	//selection path

	lambda_max = lambdaMax_adaEN(Ycopy,Xcopy,W,M, N,alpha_factor);
	//
	//for(i=0;i<M;i++) f[i] = 1;
	// dscal)(&MM,&alpha,B,&inci);
	if(verbose==0) printf("Step 4: lasso selection path.\n");
}

	//MPI:Bcast
	MPI_Bcast (&lambda_max, 1, MPI_DOUBLE,0,MPI_COMM_WORLD);	
	MPI_Bcast (&sigma2, 1, MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast (W, MM, MPI_DOUBLE,0,MPI_COMM_WORLD);	


	//printMat(B,M,M);
	//ilambda_cv_ms = 0;
//printMat(IBinv,M,M);	
	for(ilambda = 0;ilambda<ilambda_cv_ms;ilambda++)
	//for(ilambda = 0;ilambda<1;ilambda++)
	{
	
		lambda_factor = lambda_factors[ilambda];
if(rank==0)
{		
		if(verbose>0) printf("\t%d/%d lambdas. \tlambda_factor: %f\n", ilambda, ilambda_cv_ms,lambda_factor);
		// call Weighted_LassoSf		Missing,
}		
		lambda = Weighted_LassoSf_adaEN(W, B, f, Y,X, Q, lambda_factor,lambda_factor_prev, 
					sigma2, maxiter, M, N, verbose,QIBinv,lambda_max,rank,size,
					alpha_factor); 	// mueL not calculated
//printMat(IBinv,M,M);							
		// call QlambdaMiddle
		//QlambdaMiddle(Y,X, Q,B,f, mueL, sigma2,M,N);
if(rank==0)
{		
		QlambdaMiddleCenter(Ycopy,Xcopy, Q,B,f,sigma2,M, N,QIBinv); //same set of Y,X 		<-- mueL not needed
}		
		lambda_factor_prev = lambda_factors[ilambda];
		
//printMat(IBinv,M,M);
	}//ilambda; selection path
	//return B,f; 
	//correct positive
	//printMat(B,M,M);
	
if(rank==0)
{	
	stat[0] = 0;// correct positive
	stat[2] = 0;//false positive
	stat[3] = 0;//positive detected
	for(i=0;i<M;i++)
	{
		for(j=0;j<M;j++)
		{
			index = j*M + i;
			if(Strue[index]==0 && B[index]!=0) stat[2] = stat[2] + 1;
			if(i!=j)
			{
				//stat[3]
				if(B[index]!=0) 
				{
					stat[3] = stat[3] + 1;
					//stat[0]
					if(Strue[index]!=0) stat[0] = stat[0] + 1;
				}
			}
		}
	}
	//power
	stat[4] = stat[0]/stat[1];
	stat[5] = stat[2]/stat[3];
	if(verbose==0) printf("Step 5: Finish calculation; detection power in stat vector.\n");
	free(Strue);
	free(meanY);
	free(meanX);

	free(Ycopy);
	free(Xcopy);
	//free(rho_factor_m);
	//free(mueL);

	//free(IBinv);
	free(Q);
}

	MPI_Barrier (MPI_COMM_WORLD);
	
	free(QIBinv);
	free(W);
	free(lambda_factors);
	free(rho_factors);
//------adaEN Apr. 2013		
	free(alpha_factors);
	free(ErrorEN);
	free(lambdaEN);
//------adaEN Apr. 2013	
	//-------------------------------- some function changes Y, X permantly.
}
// read: Y,X,Missing,B: B is contains true values
// write: B,f

//Input options:
//"-Nsample"
//"-Ngene"
//"-verbose"
//"-response"
//"-covariate"
//"-true"
//"-missing"

	  
	  
void main(int argc, char *argv[])
{

	// MPI setup
	int rank, size;
	int i;
	double elapsed_time;
    MPI_Init(&argc, &argv); // note that argc and argv are passed
                            // by address
	MPI_Barrier (MPI_COMM_WORLD);
	elapsed_time = - MPI_Wtime();
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
	//
	int M, N;
	int verbose = 3;
	char Yfile[256], Xfile[256], Bfile[256], Mfile[256],Ofile[256];
    *Yfile=0;
    *Xfile=0;
	*Bfile = 0;
	*Mfile = 0;
	int na=1;
if(rank==0)
{	
   while(na < argc)
   {
      if ( 0 == strcmp(argv[na],"-Nsample") ){
         N = atof(argv[++na]);
         if (na+1==argc) break;
         na++;
      }
      else if ( 0 == strcmp(argv[na],"-Ngene") ){
         M = atof(argv[++na]);
         if (na+1==argc) break;
         na++;
      }
       else if ( 0 == strcmp(argv[na],"-verbose") ){
         verbose = atof(argv[++na]);
         if (na+1==argc) break;
         na++;
      }	  
      else if ( 0 == strcmp(argv[na],"-response") ){
         strcpy(Yfile,argv[++na]);
         if (na+1==argc) break;
         na++;
      }
      else if ( 0 == strcmp(argv[na],"-covariate") ){
         strcpy(Xfile,argv[++na]);
         if (na+1==argc) break;
         na++;
      }
      else if ( 0 == strcmp(argv[na],"-true") ){
         strcpy(Bfile,argv[++na]);
         if (na+1==argc) break;
         na++;
      }
      else if ( 0 == strcmp(argv[na],"-missing") ){
         strcpy(Mfile,argv[++na]);
         if (na+1==argc) break;
         na++;
      }
      else if ( 0 == strcmp(argv[na],"-o") ){
         strcpy(Ofile,argv[++na]);
         if (na+1==argc) break;
         na++;
      }

      else{
         printf( "Unknown option \n");
         exit(0);
      }
   }
}
MPI_Bcast (&M, 1, MPI_INT,0,MPI_COMM_WORLD);
MPI_Bcast (&N, 1, MPI_INT,0,MPI_COMM_WORLD);
MPI_Bcast (&verbose, 1, MPI_INT,0,MPI_COMM_WORLD);
	FILE *yPtr,*xPtr,*missPtr,*bPtr;
	
	

if(rank==0)
{
	yPtr = fopen(Yfile,"r");
	if(yPtr==NULL)
	{
		printf("The Response File could not be opened!\n");
		exit(1);
	}
}
	double *Y,*X,*B,*f,*stat;;
	int *Missing;
	int MN = M*N;
	int MM = M*M;
	B = (double* ) calloc(MM, sizeof(double)); 	
	f=(double* ) calloc(M, sizeof(double)); 		
if(rank==0)
{
	// the copy Y,X,B for workers are local names; e.g., in CV, Bcast the training Y, X, B
	Y = (double* ) calloc(MN, sizeof(double)); 
	X = (double* ) calloc(MN, sizeof(double)); 

	Missing = (int* ) calloc(MN, sizeof(int)); 
	

	i = 0;

	printf("\nReading from Response File ...\n");

	while(i<MN)
	{
		fscanf(yPtr,"%lf",&Y[i]);
		i= i+ 1;
	}
	printf("Number of elements read in Y: %d\n",i);
	fclose(yPtr);
	//printMat(Y,30,1);
	
	//X
	xPtr = fopen(Xfile,"r");
	if(xPtr==NULL)
	{
		printf("The  Covariate File could not be opened!\n");
		exit(1);
	}else
	{	
		i = 0;
		printf("\nReading from Covariate File ...\n");
		while(i<MN)
		{
			fscanf(xPtr,"%lf",&X[i]);
			i= i+ 1;
		}
		printf("Number of elements read in X: %d\n",i);
		fclose(xPtr);
	}
	
	//B
	bPtr = fopen(Bfile,"r");
	if(xPtr==NULL)
	{
		printf("The True Effect File could not be opened!\n");
		exit(1);
	}else
	{	
		i = 0;
		printf("Reading from True File ...\n\n");
		//while(!feof(bPtr))
		while(i<MM)
		{
			fscanf(bPtr,"%lf",&B[i]);
			i= i+ 1;
		}
		printf("Number of elements read in B: %d\n",i);		
		fclose(bPtr);
	}
	
	//Missing
	missPtr = fopen(Mfile,"r");
	if(missPtr==NULL)
	{
		printf("The Missing Data File could not be opened!\n");
		exit(1);
	}else
	{	
		i = 0;
		printf("\nReading from Missing File ...\n");
		while(i<MN)
		{
			fscanf(missPtr,"%d",&Missing[i]);
			i= i+ 1;
		}
		printf("Number of elements read in Missing: %d\n",i);	
		fclose(missPtr);		
	}

	//printMat(Y,30,1);
	//printMat(B,30,30);
	//verbose


	stat = (double* ) calloc(6, sizeof(double)); 

	// stat[0] correct positive
	// stat[1] total positive
	//stat[2] false positive
	//stat[3] positive detected
	//stat[4] power
	// stat[5] fdr
}	

	mainSML_adaEN(Y, X, &M, &N, Missing,B, f,stat,&verbose,rank,size);

	
if(rank==0)
{
	printMat(stat,6,1);
	free(Missing);

	free(stat);

	//free
	free(Y);
	free(X);

}
	free(B);
	free(f);
	elapsed_time += MPI_Wtime();
	MPI_Finalize();
if(rank==0)
{
	printf("\n The computational time in the cluster is %f seconds\n",elapsed_time);
}
	
}































