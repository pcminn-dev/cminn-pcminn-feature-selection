import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.special import digamma
from sklearn.preprocessing import minmax_scale
import time
import logging
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore", message="All-NaN axis encountered")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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

# === Dataset Generator ===
def binarize_y(yV, bins):
    xYY = np.linspace(min(yV), max(yV), bins + 1)
    y2V = np.zeros(len(yV), dtype=int)
    for i in range(bins):
        y2V[(yV >= xYY[i]) & (yV <= xYY[i+1])] = i + 1
    if bins == 2 and np.max(y2V) == 2:
        y2V[y2V != 1] = 2
    return y2V

def generate_dataset(seed, bins, dataset_type):
    rng = np.random.default_rng(seed)
    if dataset_type == 'A':
        n, m_total, rho, coef = 10000, 22, 0.5, 0.5
        SigM = rho * np.ones((m_total, m_total)) + (1 - rho) * np.eye(m_total)
        xM = rng.multivariate_normal(np.zeros(m_total), SigM, n)
        y1 = -3*xM[:,0] + 2*xM[:,1] + rng.normal(size=n)
        y2 = 3*xM[:,2] + 2*xM[:,3] - 4*xM[:,4] + rng.normal(size=n)
        yV = coef * y1 + (1 - coef) * y2
    elif dataset_type == 'B':
        n, m_total = 10000, 22
        xM = rng.normal(0, 1, (n, m_total))
        f1, f2 = xM[:,0], xM[:,1]
        f3 = 0.2*f1 + 0.3*f2 + 2.0*xM[:,2]
        f4 = 0.1*f1**2 + 0.1*f2**2
        xM[:,2], xM[:,3] = f3, f4
        yV = f1 + f2 + 0.2*f3 + 0.3*f4 + rng.normal(size=n)
    else:  # C
        n, m_total = 1000, 30
        fM = np.full((n, m_total), np.nan)
        x1M = rng.normal(0, 1, (n, 5))
        x5 = rng.normal(0, 1, n) * rng.normal(0, 1, n)
        x1, x2, x3, x4 = x1M[:,0], x1M[:,1], x1M[:,2], x1M[:,3]
        fM[:,0:6] = np.column_stack([x1, x2, x1*x2, x3, x4**2, x1*x5])
        fM[:,6:12] = 0.8 * fM[:,0:6] + np.sqrt(1 - 0.8**2) * rng.normal(0, 1, (n, 6))
        fM[:,12:18] = 0.4 * fM[:,0:6] + np.sqrt(1 - 0.4**2) * rng.normal(0, 1, (n, 6))
        fM[:,18:] = rng.normal(0, 1, (n, 12))
        std_f = np.std(fM[:, 0:6], axis=0)
        bi = 1 / std_f
        yV = np.sum(bi * fM[:, 0:6], axis=1) + rng.normal(size=n)
        xM = fM

    y_bin = binarize_y(yV, bins).reshape(-1, 1)
    xM = minmax_scale(xM)
    y_bin = minmax_scale(y_bin)
    return xM, y_bin

# === MAIN RUNNER ===
dataset_configs = [("A", 5), ("B", 4), ("C", 6)]
bins_list = [2, 10]
base_path = "/home/pptower/mlops/data/cminn"
pcminn_root = os.path.join(base_path, "CMINN")
os.makedirs(pcminn_root, exist_ok=True)

for name, NoF in dataset_configs:
    for bins in bins_list:
        print(f"\n🧠 DATASET: {name} | BINS: {bins} | FEATURES: {NoF}")
        output_dir = os.path.join(pcminn_root, f"CMINN_{name}_{bins}bin")
        os.makedirs(output_dir, exist_ok=True)

        execution_times = []
        outputs = []

        for i in tqdm(range(10), desc=f"{name}_{bins}bin"):
            try:
                seed = 5 + i
                X, y = generate_dataset(seed, bins, name)
                start_time = time.time()
                selected = MIfeatureselection0(y, X, k=20, NoF=NoF)
                duration = time.time() - start_time
                execution_times.append(duration)
                outputs.append(selected)
            except Exception as e:
                logger.error(f"Run {i+1} failed: {str(e)}")
                execution_times.append(None)
                outputs.append([None] * NoF * 2)

        df_times = pd.DataFrame({'Run': list(range(1, 11)), 'ExecutionTime': execution_times})
        df_outputs = pd.DataFrame([out.flatten() if out is not None else [None]*NoF*2 for out in outputs])
        df_times.to_csv(os.path.join(output_dir, "execution_times.csv"), index=False)
        df_outputs.to_csv(os.path.join(output_dir, "selected_features_runs.csv"), index=False)
        logger.info(f"✅ Saved results for {name} with {bins} bins\n")

