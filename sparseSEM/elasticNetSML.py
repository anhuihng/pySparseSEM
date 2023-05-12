"""

--------------------------------------------------------------------------
elasticNetSML.py:
    Fit a Structural Equations Model with lasso or elastic-net regularization.
    elasticNetSML.py provides a wrapper to the sparseSEM C/C++ routines. All
    variables in the arguments are keyword-only. (see examples below).
--------------------------------------------------------------------------
DESCRIPTION:
-----------
    Fit a Structural Equations Model with lasso or elastic-net regularization by first grid searching the optimal
    hyperparameters through 5-fold CV.  The regularization path is computed for the lasso or elasticnet penalty
    at a grid of values for the regularization parameter alpha, lambda.

    Step 1:  5 fold CV (see parameters of the grid values)
    Step 2: fit model with (alpha, lambda) from Step 1.

    Note:
        Regularization path: the lasso-strong rule is applied, thus the path is run through lambda_max to lambda_opt,
        where lambda_max is the lambda that keeps only 1 non-zero edge,and lambda_opt is the optimal lambda from CV.

        The program implements the following components:
        Step 1. SEM-ridge regression (L2 penalty) with k-fold CV: this step find the optimal ridge hyperparameter rho
        Step 2. fit SEM reidge regression model (L2 penalty) with rho from Step 1, obtain the initial status (non-sparse)
                of network structure (B_ridge);
        Step 3.SEM-elastic net regression with k-fold CV: this step finds the optimal hyperparameter (alpha, lambda)
        Step 4. fit SEM-elastic net model with (alpha, lambda) from Step 3.
        Step 5. result calculation for PD, FDR, provide the output


FUNCTION INTERFACE:
-----------
    import sparseSEM
    fit = sparseSEM.elasticNetSML(X, Y, M, B, verbose = 1);


INPUT ARGUMENTS (in SEM: Y = BY + FX + e):
---------------
    X           The network node attribute matrix with dimension of M by N, with M being the number of nodes,
                and N being the number of samples. Theoretically, X can be L by N matrix, with L being the total
                node attributes. However, in current implementation, each node only allows one and only one attribute.
                If users have more than one attributes for some nodes,  please consider selecting the top one by either
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



OUTPUT ARGUMENTS:
---------------
    Function output is a dictionary with the following keys:
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

Hyperparameters:
--------
    Cross - Validation:
                By default, hyperparameters for elastic net (alpha, lambda) are selected by 5 fold cross validation
                (CV) with alpha from seq(start = 0.95, to = 0.05, step = -0.05), and lambda from lambda_max to
                lambda_max*0.001 in 19 steps (20 lambdas)

    If users would like to provide fine granular grid search parameters: see elasticNetSMLcv()

    Stability Selection
                Stability Selection (Meinshausen and Buhlmann 2010) can be an alternative approach to the cross
                validation.

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
def elasticNetSML(X, Y, Missing = None, B = None, verbose = 0):
    M, N = X.shape;
    if B is None:
        B = np.zeros(M,M);
    if Missing is None:
        Missing = np.zeros(M,N);

    if B.shape == (M,M) and Y.shape == (M,N):
        if(verbose >= 0): print("\t sparseSEM", M, "nodes, ", N, "samples; Verbose: ", verbose, "\n\n")
    else:
        if(verbose >= 0): print("Error: sparseSEM currently support only the same dimension of X, Y");

    semlib = loadSEMlib();
    f = np.ones((1,M) );
    stat = np.zeros(6);

    B = np.asfortranarray(B);
    B = B.ctypes.data_as(ctypes.POINTER(ctypes.c_double)); # row based. need to use column based.
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

    m = ctypes.c_int(M);
    n = ctypes.c_int(N);
    v = ctypes.c_int(verbose);
    startTime = time.time();
    semlib.mainSML_adaEN(Y, X, m, n, Missing, B, f, stat, v);
    #void mainSML_adaEN(double *Y, double *X, int *m, int *n, int *Missing,double*B, double *f,double*stat,int*VB)
    endTime = time.time();
    runTime = endTime - startTime;

    if (verbose >= 0):
        print(f"\t sparseSEM running time: ", runTime, " seconds \n");
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
        output["statistics"].append((stats[i], stat[i]) );

    output["runTime"] = runTime;
    return output;
