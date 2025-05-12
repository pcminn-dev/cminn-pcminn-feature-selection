import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0 explicitly

import cupy as cp
import pandas as pd
from cuml.neighbors import NearestNeighbors
from scipy.special import digamma
from sklearn.preprocessing import minmax_scale
import time
import logging
import warnings

warnings.filterwarnings("ignore", message="All-NaN axis encountered")

# Initialize logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_dataset(filepath):
    start_time = time.time()
    logger.info(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)
    X = df.iloc[:, :-1].values  # Features
    y = df.iloc[:, -1].values  # Target variable
    logger.info(f"Dataset loaded in {time.time() - start_time:.4f} seconds.")
    return cp.asarray(X), cp.asarray(y).reshape(-1, 1)


def MIfeatureselection0(xMC, xMF, k, NoF):
    logger.info("Starting Mutual Information feature selection...")
    start_time = time.time()

    nf = xMF.shape[1]
    # fVi holds the *1-based* indices of the remaining features
    fVi = cp.arange(1, nf + 1)

    # Prepare containers for the selected features
    fM = cp.empty((xMF.shape[0], 0))
    fVs = cp.full((NoF,), cp.nan)
    fVMI = cp.full((NoF,), cp.nan)

    for i in range(NoF):
        # Allocate MI array only for the features that remain in fVi
        current_size = fVi.size
        mi = cp.full((current_size,), cp.nan)

        # Compute MI for each remaining feature
        for j in range(current_size):
            # Remember fVi is 1-based; we convert to 0-based by subtracting 1
            feature_index_0_based = fVi[j] - 1

            # Ensure we skip NaN or invalid columns
            if cp.isnan(cp.sum(xMF[:, feature_index_0_based])):
                continue

            # Add a small noise to break ties, as your original code suggests
            noisy_xMF_j = xMF[:, feature_index_0_based] + cp.random.normal(0, 10**-5, xMF.shape[0])

            # Compute MI with cmikra1n
            mi_local, _, _ = cmikra1n(xMC, noisy_xMF_j, fM, k)
            mi[j] = mi_local

        # Identify the feature with maximum MI
        MaxMI = cp.nanargmax(mi)
        max_val = mi[MaxMI]

        if cp.isnan(max_val):
            logger.warning(f"All MI values are NaN at iteration {i}. Stopping early.")
            break

        # Get the actual feature id from fVi
        best_feature = fVi[MaxMI]
        logger.info(f"Iteration {i}: best_feature={best_feature}, max_MI={max_val}")

        # Append this selected feature to fM
        fM = cp.hstack((fM, cp.expand_dims(xMF[:, best_feature - 1], axis=1)))
        fVs[i] = best_feature
        fVMI[i] = max_val

        # Remove the chosen feature from the pool
        fVi = cp.delete(fVi, MaxMI)  # remove by index, not by value

    logger.info(f"Feature selection completed in {time.time() - start_time:.4f} seconds.")
    # Return the selected features and their MI values
    return cp.vstack((fVs, fVMI))


def cmikra1n(xM, yM, zM, k):
    if zM.shape[1] == 0:
        mi = mkraskov1(xM, yM, k)
        return mi[0], [cp.nan, cp.nan], 0

    xM = xM.reshape(-1, 1) if xM.ndim == 1 else xM
    yM = yM.reshape(-1, 1) if yM.ndim == 1 else yM

    xMb = cp.concatenate((xM, yM, zM), axis=1)
    maxdistV = gpu_kd_tree_knn(xMb, k)

    nz = npoinmultranges(zM, maxdistV, k)
    nyz = npoinmultranges(cp.concatenate((yM, zM), axis=1), maxdistV, k)
    nxz = npoinmultranges(cp.concatenate((xM, zM), axis=1), maxdistV, k)

    mi = digamma(k) - cp.mean(digamma(nxz) + digamma(nyz) - digamma(nz))

    return mi, [digamma(nz), digamma(nyz)], cp.corrcoef(xM.T, yM.T)[0, 1]


def gpu_kd_tree_knn(data, k):
    model = NearestNeighbors(n_neighbors=k + 1, algorithm='brute')
    model.fit(data)
    distances, _ = model.kneighbors(data)
    return distances[:, -1]


def npoinmultranges(xM, rV, k):
    n_samples = xM.shape[0]
    model = NearestNeighbors(n_neighbors=n_samples, algorithm='brute')
    model.fit(xM)
    distances, _ = model.kneighbors(xM)
    within_radius = (distances <= rV[:, None])
    return cp.asarray(cp.sum(within_radius, axis=1), dtype=cp.float64)


def mkraskov1(xM1, xM2, k):
    n1 = xM1.shape[0]

    if xM2.ndim == 1:
        xM2 = xM2[:, cp.newaxis]

    xMb = cp.concatenate((xM1, xM2), axis=1)
    maxdistV = gpu_kd_tree_knn(xMb, k)

    topsi1 = npoinmultranges(xM1, maxdistV, k)
    topsi2 = npoinmultranges(xM2, maxdistV, k)

    mi = digamma(k) + digamma(n1) - cp.mean(digamma(topsi1) + digamma(topsi2))
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

#For PCMINN GPU.
#xMC, xM = cp.asarray(xMC), cp.asarray(xM)