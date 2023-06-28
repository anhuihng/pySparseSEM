import ctypes
import numpy as np
import pandas as pd
from loadSEMlib import loadSEMlib
from elasticNetSMLcv import elasticNetSEMcv
import time

def elasticnetSEM_STS(i, Y_sts, X_sts, Missing_sts=None, B=None, STS_para=None, kFold=5, verbose=0):
    M = Y_sts.shape[0]
    N = Y_sts.shape[1]
    if Missing_sts is None:
        Missing_sts = np.zeros((M, N))
    if B is None:
        B = np.zeros((M, M))
    if X_sts.shape[0] != M:
        if verbose >= 0:
            print("error: sparseSEM currently supports only the same dimension of X, Y.")
        return None
    if STS_para is None:
        if verbose >= 0:
            print("error: enSEM_STS is an internal function with STS_para setup as 2-column matrix of [alphas, lambdas].")
        return None
    if verbose >= 1:
        if i != 0:
            print("\tbootstrapping:", i, "\n")

    alpha_factors = STS_para[:, 0]
    lambda_factors = STS_para[:, 1]
    nAlpha = len(alpha_factors)
    nLambda = len(lambda_factors)

    f = np.ones((M, 1))
    stat = np.zeros(6)
    mseStd = np.zeros(nLambda * 2)

    MM = M * M
    Bout = np.zeros((MM, nAlpha))
    nKeep = int(N / 2)
    keepSample = np.random.choice(N, nKeep, replace=False)
    Y = Y_sts[:, keepSample]
    X = X_sts[:, keepSample]
    Missing = Missing_sts[:, keepSample]
    N = Y.shape[1]
    semlib = loadSEMlib();
    tStart = time.process_time()
    # Call the mainSML_adaENstabilitySelection function
    # Use appropriate function name and arguments based on the available libraries in Python

    Y = np.asfortranarray(Y);
    Y = Y.ctypes.data_as(ctypes.POINTER(ctypes.c_double));
    X = np.asfortranarray(X);
    X = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double));

    Missing = np.asfortranarray(Missing);
    Missing = Missing.ctypes.data_as(ctypes.POINTER(ctypes.c_int));
    B = np.asfortranarray(B);
    B = B.ctypes.data_as(ctypes.POINTER(ctypes.c_double));  # row based. need to use column based.
    f = np.asfortranarray(f);
    f = f.ctypes.data_as(ctypes.POINTER(ctypes.c_double));
    stat = np.asfortranarray(stat);
    stat = stat.ctypes.data_as(ctypes.POINTER(ctypes.c_double));

    alpha_factors = np.asfortranarray(alpha_factors);
    alpha_factors = alpha_factors.ctypes.data_as(ctypes.POINTER(ctypes.c_double));
    _nAlpha = ctypes.c_int(nAlpha);
    lambda_factors = np.asfortranarray(lambda_factors);
    lambda_factors = lambda_factors.ctypes.data_as(ctypes.POINTER(ctypes.c_double));
    _nLambda = ctypes.c_int(nLambda);

    mseStd = np.asfortranarray(mseStd);
    mseStd = mseStd.ctypes.data_as(ctypes.POINTER(ctypes.c_double));

    Bout = np.asfortranarray(Bout);
    Bout = Bout.ctypes.data_as(ctypes.POINTER(ctypes.c_double));
    _kFold = ctypes.c_int(kFold);
    m = ctypes.c_int(M);
    n = ctypes.c_int(N);
    v = ctypes.c_int(verbose);

    semlib.mainSML_adaENstabilitySelection(Y, X, M, N, Missing, B, f, stat, alpha_factors, nAlpha,
                                           lambda_factors, nLambda, mseStd, verbose, Bout, kFold)
    tEnd = time.process_time()
    runTime = tEnd - tStart

    if verbose >= 1:
        print("\tcomputation time:", runTime, "sec\n")

    Bout = np.ctypeslib.as_array(Bout, shape=(MM*nAlpha,1));
    temp = (Bout != 0).astype(float) #convert to binary 0/1
    Bout = temp.reshape(MM, nAlpha, order='F')

    return Bout


def enSEM_stability_selection(Y, X, Missing=None, B=None, alpha_factors=np.arange(1, 0.01, -0.1),
                              lambda_factors=10 ** np.arange(-0.2, -3, -0.2), kFold=5, nBootstrap=100, verbose=0):
    '''
    #' Stability Selection
    #' Ref1: Meinshausen N. and Buhlmann P, J. R. Statist. Soc. B (2010) 72
    #' Ref2: Shah R. and Samworth R,J. R. Statist. Soc. B (2013) 75
    #'
    #' STS runs bootstrap for nBootstrap (default=100) times, each time only half of the datapoints used.
    #' This results low power when N is too small, and performance may not be as strong as CV.
    #' When N is relatively large, STS may have strong performance than CV.
    #'
    #'
    #'output:
    #'for a list of candidate threshold (denoted as pi in the reference paper):
    #'FDRset:
    #'col1: pi (per reference paper, recommended to choose pi = 0.9)
    #'col2: pre-comparison error rate --- E(v)/p
    #'col3: E(v) the expected number of falsely selected variables -- Eq 9.
    #'col4: E(v)_ShahR E(v) in Ref2
    #'col5: nSTS - the number of stable selected edges
    #'col6: FDR: False discovery rate = E(v)/nSelected
    #'col7: FDR_ShahR FDR in Ref2 = E(v_shahR)/nSelected
    #'
    #'The optimal threshold from simulation is the one with the smallest FDR and largest true nSTS.
    #'
    Parameters
    ----------
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
    alpha_factors
                The set of candidate alpha values.  Default is seq(start = 0.95, to = 0.05, step = -0.05)

    lambda_factors
                The set of candidate lambda values. Default is 10^seq(start =1, to = 0.001, step = -0.2)
    kFold
                Kcv folds of cross validation.  Default is 5-fold CV (k =5).

    nBootstrap
                number of bootstrapping, default = 100
    verbose
                log output, to minimize the output, set to -1.
    Returns
        STS
                The stable effects are those effects selected by STS, i.e., the non-zero values in matrix B.
        statistics
                the final STS scores with components of:
                1. threshold: denoted as pi in  Meinshausen N. and Buhlmann P (2010)
                2. pre-comparison error rate
                3. E(v)
                4. E(v)_ShahR
                5. nSTS: final number of stable effects with pi that leads to minimum FDR
                6. FDR
                7. FDR_ShahR

        STS data
                Bootstrapping details.

        call
                call to the function.
    -------

    '''
    M = Y.shape[0]
    N = Y.shape[1]

    if Missing is None:
        Missing = np.zeros((M, N))
    if B is None:
        B = np.zeros((M, M))

    if X.shape[0] != M:
        if verbose >= 0:
            print("error: sparseSEM currently supports only the same dimension of X and Y.")
        return None

    this_call = locals()

    if verbose >= 0:
        print("\tstability selection elastic net SEM;", M, "Nodes,", N, "samples; verbose:", verbose)
        if nBootstrap != 0:
            print("\tbootstrapping:", nBootstrap, "\n")

    f = np.ones((M, 1))
    stat = np.zeros(6)
    tStart = time.process_time()

    # R_package parameter
    nAlpha = len(alpha_factors)
    nLambda = len(lambda_factors)
    mse = np.zeros(nLambda * nAlpha)
    mseSte = np.zeros(nLambda * nAlpha)
    mseStd = np.zeros(nLambda * 2)

    parameters = np.zeros((0, 2))
    for i in alpha_factors:
        col1 = np.full(nLambda, i)
        col2 = lambda_factors
        para = np.column_stack((col1, col2))
        parameters = np.vstack((parameters, para))
    nPara = parameters.shape[0]
    Bselection = [None] * nBootstrap

    # Selection Path
    MM = M * M
    NtopEff = MM - M
    qEffect = np.zeros(nBootstrap)
    piThreshold = np.arange(0.6, 1, 0.01)

    qsEffect = np.zeros((MM, nBootstrap))

    for i in range(nBootstrap):
        if verbose >= -1:
            print("bootstrapping:", i)

        currentSTS = elasticnetSEM_STS(i, Y.copy(), X.copy(), Missing, B.copy(), STS_para=parameters, kFold=kFold,
                               verbose=verbose)  # MM by nPara matrix with 0, or 1 values
        Bselection[i] = currentSTS

    # Compute results
    nStep = nPara
    nTopEff = MM - M
    qsEffect = np.zeros((MM, nBootstrap))
    qABave = np.zeros(nStep)  # ave of (alpha,lambda) selected effects over nBootstrap
    qAB = np.zeros((MM, nStep))  # each effect: how many times selected with this ab
    for i_ab in range(nStep):
        for j_repeat in range(nBootstrap):
            qsEffect[:, j_repeat] = Bselection[j_repeat][:, i_ab]
        qABave[i_ab] = np.mean(np.sum(qsEffect, axis=0))  # average number of effect for this a, b
        qAB[:, i_ab] = np.sum(qsEffect, axis=1)  # each effect: how many times selected with this ab

    # Re-grid the set of shrinkage parameters
    piThreshold = np.arange(0.6, 1, 0.01)
    nPi = len(piThreshold)
    stablySelected = [None] * nPi
    preFDR = np.zeros(nPi)
    preFDR2 = np.zeros(nPi)
    pCER = np.zeros(nPi)
    nSTS = np.zeros(nPi)

    nGrid = nStep
    qStable = np.zeros(MM)
    FDRsetS = [None] * nGrid
    for i_grid in range(nGrid):
        nStep = i_grid + 1
        qABave_igrid = qABave[:nStep]  # average number of effect for this a, b
        qAB_igrid = qAB[:, :nStep]  # each effect: how many times selected with this ab
        for i in range(MM):
            qStable[i] = np.max(qAB_igrid[i, :])  # each effect: how many times selected with this ab-set
        for i in range(nPi):
            piThresholdCut = piThreshold[i] * nBootstrap
            stablySelected[i] = np.where(qStable >= piThresholdCut)[0]
            nSTS[i] = len(stablySelected[i])
            pCER[i] = (np.mean(qABave_igrid) ** 2) / ((2 * piThreshold[i] - 1) * (nTopEff ** 2))
            preFDR[i] = (np.mean(qABave_igrid) ** 2) / ((2 * piThreshold[i] - 1) * nTopEff)
            if piThreshold[i] <= 0.75:
                preFDR2[i] = (np.mean(qABave_igrid) ** 2) / (
                            2 * (2 * piThreshold[i] - 1 - 1 / (2 * nBootstrap)) * nTopEff)
            else:
                preFDR2[i] = (4 * (1 - piThreshold[i] + 1 / (2 * nBootstrap)) * (np.mean(qABave_igrid) ** 2)) / (
                            (1 + 1 / (2 * nBootstrap)) * nTopEff)
        FDR = preFDR
        FDR2 = preFDR2
        FDRset = np.column_stack((piThreshold, pCER, preFDR, preFDR2, nSTS, FDR, FDR2))
        for i in range(nPi):
            piThresholdCut = piThreshold[i] * nBootstrap
            FDRset[i, 5] = FDRset[i, 5] / len(np.where(qStable >= piThresholdCut)[0])
            FDRset[i, 6] = FDRset[i, 6] / len(np.where(qStable >= piThresholdCut)[0])
        FDRsetS[i_grid] = FDRset

    # Optimal i_grid; theoretically, i_grid = nStep when setting alpha. lambda to be with large penalty
    OUT = elasticNetSEMcv(Y, X, Missing, B, alpha_factors, lambda_factors, kFold=5, verbose=-1)
    mse = OUT['cv'][:, 2]
    index = np.argmin(mse)
    j = index
    while j > 0:
        if OUT['cv'][j, 2] > (OUT['cv'][index, 2] + OUT['cv'][index, 3]):
            break
        j -= 1
    index = j + 1
    alpha = OUT['cv'][index, 0]
    lambda_ = OUT['cv'][index, 1]
    ind1 = np.where(parameters[:, 0] == alpha)[0]
    ind2 = np.where(parameters[:, 1] == lambda_)[0]
    index = np.intersect1d(ind1, ind2)[0]

    FDRset = FDRsetS[index]

    FDRset_columns = ['threshold', 'pre-comparison error rate', 'E(v)', 'E(v)_ShahR', 'nSTS', 'FDR', 'FDR_ShahR']
    FDRset = pd.DataFrame(FDRset, columns=FDRset_columns)
    tEnd = time.process_time()
    simTime = tEnd - tStart

    j = np.argmin(FDRset['FDR_ShahR'])
    stableEffcts_index = stablySelected[j]
    data = np.zeros(MM)
    data[stableEffcts_index] = 1
    stableEffects = data.reshape(M, M, order='F')
    stat = FDRset.iloc[j, :].values
    computeData = {'simTime': simTime, 'piThreshold': piThreshold, 'stablySelected': stablySelected,
                   'FDRset': FDRset, 'nSTS': nSTS, 'nBootstrap': nBootstrap, 'bootstrap_B': qsEffect,
                   'bootstrap_count': qEffect, 'NtopEff': NtopEff, 'FDRsetS': FDRsetS, 'selectionB': Bselection,
                   'parameters': parameters, 'qABave': qABave, 'qAB': qAB}
    output = {'STS': stableEffects, 'statistics': stat, 'STS data': computeData}
    output['call'] = this_call

    return output

