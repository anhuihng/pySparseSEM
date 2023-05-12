"""

--------------------------------------------------------------------------
elasticNetSMLcv.py:
    Cross Validation (CV) function for the sparseSEM with lasso or elastic-net regularization.
    elasticNetSMLcv.py provides a wrapper to the sparseSEM CV routines. All
    variables in the arguments are keyword-only. (see examples below).
--------------------------------------------------------------------------
DESCRIPTION:
-----------
    Fit a Structural Equations Model with lasso or elastic-net regularization through cross validation grid search.
    The optimal (alpha, lambda) are chosen based on the optimal mean square error.


FUNCTION INTERFACE:
-----------
    import sparseSEM
    fit = sparseSEM.elasticNetSMLcv(X, Y, M, B, verbose = 1);


INPUT ARGUMENTS (in SEM: Y = BY + FX + e):
---------------
    X           The network node attribute matrix with dimension of M by N, with M being the number of nodes,
                and N being the number of samples. Theoretically, X can be L by N matrix, with L being the total
                node attributes. However, in current implementation, each node only allows one and only one attribute.
                If you have more than one attributes for some nodes,  please consider selecting the top one by either
                correlation or principal component methods.
                X is normalized inside the function.

    Y           The observed node response data with dimension of M by N. Y is normalized inside the function.

    B           For a network with M nodes, B is the M by M adjacency matrix.
                If data is simulated/with known true network topology (i.e., known adjacency matrix), the Power
                of detection (PD) and False Discovery Rate (FDR) is computed in the output parameter 'statistics'.

                If the true network topology is unknown, B is optional, and the PD/FDR in output parameter
                'statistics' should be ignored.

    Missing     M by N matrix corresponding to elements of Y. 0 denotes no missing, while 1 denotes missing.
                If a node j in sample i has a missing label (Missing[j,i] = 1), the node response Y[j,i] is set to 0.

    alpha_factors
                The set of candidate alpha values.  Default is seq(start = 0.95, to = 0.05, step = -0.05)

    lambda_factors
                The set of candidate lambda values. Default is 10^seq(start =1, to = 0.001, step = -0.2)

    Kcv         Kcv folds of cross validation.  Default is 5-fold CV (Kcv =5).



OUTPUT ARGUMENTS:
---------------
    Function output is a dictionary with the following keys:
set 1: CV parameters
    mseStd      the CV results.
                col1: alpha
                col2: lambda
                col3: mean of residual error in k-fold CV
                col4: standard error of residual error in k-fold CV

    alpha       the final alpha chosen

    lambda      the final lambda chosen

Set 2: Fit parameters (same as in function elasticNetSML.py)
    weight      the computed weights for the network topology. B[i,j] = 0 means there is no edge between node i and j;
                B[i,j]!=0 denotes an (undirected) edge between note i and j.

    f           f is 1 by M array keeping the weight for X (in SEM: Y = BY + FX + e). Theoretically, F can be M by L matrix,
                with M being the number of nodes, and L being the total node attributes. However, in current implementation,
                each node only allows one and only one attribute.
                If you have more than one attributes for some nodes, please consider selecting the top one by either
                correlation or principal component methods.

    statistics  statistics is 1x6 array keeping record of:
                1. correct positive
                2. total positive
                3. false positive
                4. positive detected
                5. Power of detection (PD) = correct positive/total positive
                6. False Discovery Rate (FDR) = false positive/positive detected

    runTime     the total computational time.


    Random Seed
                User is responsible to set the random seed before calling this function.

LICENSE:
-------
    GPL-2 | GPL-3
AUTHORS:
-------
    C code, R package (sparseSEM: https://cran.r-project.org/web/packages/sparseSEM/index.html)
    and this Python package were written by Anhui Huang (anhuihuang@gmail.com)

REFERENCES:
----------
    Huang A. (2014) Sparse Model Learning for Inferring Genotype and Phenotype Associations. Ph.D Dissertation,
    University of Miami, Coral Gables, FL, USA.
    Huang A. (2014) sparseSEM: Sparse-Aware Maximum Likelihood for Structural Equation Models. Rpackage
    (https://cran.r-project.org/web/packages/sparseSEM/index.html)
    Meinshausen, N. and P. Buhlmann, 2010 Stability selection. J. R. Stat. Soc. Series B. Stat. Methodol. 72: 417-473.

"""
import ctypes
import numpy as np
from loadSEMlib import loadSEMlib
import time

def elasticNetSMLcv(X, Y, Missing=None, B=None,
                    alpha_factors = np.linspace(1, 0.05, 20),
                    lambda_factors =10**np.linspace(-0.2, -4, 20),
                    kFold = 5,
                    verbose=0):
    M, N = X.shape;
    if B is None:
        B = np.zeros(M*M).reshape(M,M);
    if Missing is None:
        Missing = np.zeros(M*N).reshape(M,N);

    if B.shape == (M, M) and Y.shape == (M, N):
        if (verbose >= 0): print("\t sparseSEM", M, "nodes, ", N, "samples; Verbose: ", verbose, "\n\n")
    else:
        if (verbose >= 0): print("Error: sparseSEM currently support only the same dimension of X, Y");

    semlib = loadSEMlib();
    f = np.ones((M));
    stat = np.zeros(6);
    nAlpha = len(alpha_factors);
    nLambda = len(lambda_factors);
    mse = np.zeros(nAlpha*nLambda);
    mseSte = np.zeros(nAlpha*nLambda);
    mseStd = np.zeros(nLambda*2);

    parameters = np.zeros(nAlpha*nLambda*2).reshape(nAlpha*nLambda, 2);
    r = 0;
    for i in range(nAlpha):
        for j in range(nLambda):
            parameters[r][0] = alpha_factors[i];
            parameters[r][1] = lambda_factors[j];
            r += 1;

    B = np.asfortranarray(B);
    B = B.ctypes.data_as(ctypes.POINTER(ctypes.c_double));  # row based. need to use column based.
    X = np.asfortranarray(X);
    X = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double));
    Y = np.asfortranarray(Y);
    Y = Y.ctypes.data_as(ctypes.POINTER(ctypes.c_double));
    Missing = np.asfortranarray(Missing);
    Missing = Missing.ctypes.data_as(ctypes.POINTER(ctypes.c_int));
    f = np.asfortranarray(f);
    f = f.ctypes.data_as(ctypes.POINTER(ctypes.c_double));
    stat = np.asfortranarray(stat);
    stat = stat.ctypes.data_as(ctypes.POINTER(ctypes.c_double));

    #alpha_factors = np.asfortranarray(alpha_factors);
    alpha_factors = alpha_factors.ctypes.data_as(ctypes.POINTER(ctypes.c_double));
    _nAlpha = ctypes.c_int(nAlpha);
    lambda_factors = lambda_factors.ctypes.data_as(ctypes.POINTER(ctypes.c_double));
    _nLambda = ctypes.c_int(nLambda);
    mse = np.asfortranarray(mse);
    mse = mse.ctypes.data_as(ctypes.POINTER(ctypes.c_double));
    mseSte = np.asfortranarray(mseSte);
    mseSte = mseSte.ctypes.data_as(ctypes.POINTER(ctypes.c_double));
    mseStd = np.asfortranarray(mseStd);
    mseStd = mseStd.ctypes.data_as(ctypes.POINTER(ctypes.c_double));
    _kFold = ctypes.c_int(kFold);

    m = ctypes.c_int(M);
    n = ctypes.c_int(N);
    v = ctypes.c_int(verbose);
    startTime = time.time();

    semlib.mainSML_adaENcv(Y, X, m, n, Missing, B, f, stat,
                           alpha_factors, _nAlpha,
                           lambda_factors,_nLambda,
                           mse, mseSte, mseStd,
                           _kFold, v);

    endTime = time.time();
    runTime = endTime - startTime;

    if (verbose >= 0):
        print(f"\t sparseSEM CV running time: ", runTime, " seconds \n");
    output = dict();
    output['weight'] = np.ctypeslib.as_array(B, shape=(M, M));
    output["F"] = np.ctypeslib.as_array(f, shape=(1, M));

    output["statistics"] = [];
    stats = ["True positive",
             "Total positive",
             "False positive",
             "Positive detected",
             "Power of detection (PD)",
             "False Discovery Rate (FDR)",
             ]

    for i in range(len(stats)):
        output["statistics"].append((stats[i], stat[i]));

    mse = np.ctypeslib.as_array(mse, shape=(nAlpha*nLambda, 1));
    mseSte = np.ctypeslib.as_array(mseSte, shape=(nAlpha * nLambda, 1));
    output["cv"] = np.column_stack((parameters, mse, mse) );

    output["runTime"] = runTime;
    return output;


