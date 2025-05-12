import numpy as np
from sklearn.preprocessing import minmax_scale
import cupy as cp
import pandas as pd
from cuml.neighbors import NearestNeighbors
from scipy.special import digamma
from sklearn.preprocessing import minmax_scale
import time
import logging



#Initialize logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

#############################CMINN#######################################################
# === Feature Selection ===
def MIfeatureselection0(xMC, xMF, k, NoF):
    logger.info("Starting Mutual Information feature selection (CPU)...")
    start_time = time.time()
    nf = xMF.shape[1]
    fVi = np.arange(1, nf + 1)
    fM = np.empty((xMF.shape[0], 0))
    fVs = np.full((NoF,), np.nan)
    fVMI = np.full((NoF,), np.nan)

    for i in range(NoF):
        mi = np.full((fVi.size,), np.nan)
        for j in range(fVi.size):
            idx = fVi[j] - 1
            if np.isnan(np.sum(xMF[:, idx])):
                continue
            noisy_feature = xMF[:, idx] + np.random.normal(0, 1e-5, xMF.shape[0])
            mi_local, _, _ = cmikra1n(xMC, noisy_feature, fM, k)
            mi[j] = mi_local

        max_idx = np.nanargmax(mi)
        max_val = mi[max_idx]
        if np.isnan(max_val):
            logger.warning("All MI values are NaN. Stopping.")
            break

        best_feature = fVi[max_idx]
        fM = np.hstack((fM, xMF[:, best_feature - 1].reshape(-1, 1)))
        fVs[i] = best_feature
        fVMI[i] = max_val
        fVi = np.delete(fVi, max_idx)

    logger.info(f"Completed in {time.time() - start_time:.2f} sec.")
    return np.vstack((fVs, fVMI))

# === CMI Support ===
def cmikra1n(xM, yM, zM, k):
    if zM.shape[1] == 0:
        mi = mkraskov1(xM, yM, k)
        return mi[0], [np.nan, np.nan], 0
    xMb = np.concatenate((xM, yM.reshape(-1, 1), zM), axis=1)
    maxdistV = cpu_kd_tree_knn(xMb, k)
    nz = npoinmultranges(zM, maxdistV, k)
    nyz = npoinmultranges(np.concatenate((yM.reshape(-1, 1), zM), axis=1), maxdistV, k)
    nxz = npoinmultranges(np.concatenate((xM, zM), axis=1), maxdistV, k)
    mi = digamma(k) - np.mean(digamma(nxz) + digamma(nyz) - digamma(nz))
    return mi, [digamma(nz), digamma(nyz)], np.corrcoef(xM.T, yM.T)[0, 1]

def cpu_kd_tree_knn(data, k):
    model = NearestNeighbors(n_neighbors=k + 1, algorithm='auto')
    model.fit(data)
    distances, _ = model.kneighbors(data)
    return distances[:, -1]

def npoinmultranges(xM, rV, k):
    n = xM.shape[0]
    model = NearestNeighbors(n_neighbors=n, algorithm='auto')
    model.fit(xM)
    distances, _ = model.kneighbors(xM)
    return np.sum(distances <= rV[:, None], axis=1).astype(np.float64)

def mkraskov1(xM1, xM2, k):
    xMb = np.concatenate((xM1, xM2.reshape(-1, 1)), axis=1)
    maxdistV = cpu_kd_tree_knn(xMb, k)
    topsi1 = npoinmultranges(xM1, maxdistV, k)
    topsi2 = npoinmultranges(xM2.reshape(-1, 1), maxdistV, k)
    mi = digamma(k) + digamma(xM1.shape[0]) - np.mean(digamma(topsi1) + digamma(topsi2))
    return mi, topsi1, topsi2

#####################Configuration of parameters to enable proper execution of PCMINN ###########################################################

#initidx is the class variable
#xM refers to the features
#if len(initidx.shape) == 1:         #check the dimensions, to avoid problems with np.concatenate
 #   xMC = initidx.reshape(-1, 1)
#if len(xM.shape) == 1:
 #   xM = xM.reshape(-1, 1)
#nf = xM.shape[1]
#xMC = minmax_scale(xMC)   ############# Scaling the variables to a common range to avoid bias
#xM = minmax_scale(xM, axis=0)     ####### 


