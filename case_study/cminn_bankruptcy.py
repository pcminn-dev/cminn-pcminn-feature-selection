
#!/usr/bin/env python
# CMINN Bankruptcy Dataset - CPU Evaluation Script

import time
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.filterwarnings("ignore", message="Not patching Numba")

df = pd.read_csv("data.csv")
y_raw = df["Bankrupt?"].values.reshape(-1, 1)
X_raw = df.drop(columns=["Bankrupt?"]).values
X_scaled = MinMaxScaler().fit_transform(X_raw)
y_scaled = MinMaxScaler().fit_transform(y_raw)

k = 20
nof_list = [5, 10, 20, 30]
repeats = 10
results = []

def cpu_kd_tree_knn(data, k):
    model = NearestNeighbors(n_neighbors=k + 1, algorithm='brute')
    model.fit(data)
    distances, _ = model.kneighbors(data)
    return distances[:, -1]

def npoinmultranges_cpu(xM, rV, k):
    model = NearestNeighbors(n_neighbors=xM.shape[0], algorithm='brute')
    model.fit(xM)
    distances, _ = model.kneighbors(xM)
    return np.sum(distances <= rV[:, None], axis=1).astype(np.float64)

def mkraskov1_cpu(xM1, xM2, k):
    xMb = np.concatenate((xM1, xM2.reshape(-1,1)), axis=1)
    maxdistV = cpu_kd_tree_knn(xMb, k)
    topsi1 = npoinmultranges_cpu(xM1, maxdistV, k)
    topsi2 = npoinmultranges_cpu(xM2.reshape(-1,1), maxdistV, k)
    return digamma(k) + digamma(xM1.shape[0]) - np.mean(digamma(topsi1) + digamma(topsi2))

def cmikra1n_cpu(xM, yM, zM, k):
    if zM.shape[1] == 0:
        return mkraskov1_cpu(xM, yM, k), None, 0
    xMb = np.concatenate((xM, yM.reshape(-1,1), zM), axis=1)
    maxdistV = cpu_kd_tree_knn(xMb, k)
    nz = npoinmultranges_cpu(zM, maxdistV, k)
    nyz = npoinmultranges_cpu(np.concatenate((yM.reshape(-1,1), zM), axis=1), maxdistV, k)
    nxz = npoinmultranges_cpu(np.concatenate((xM, zM), axis=1), maxdistV, k)
    return digamma(k) - np.mean(digamma(nxz) + digamma(nyz) - digamma(nz)), None, 0

def MIfeatureselection0_cpu(xMC, xMF, k, NoF):
    nf = xMF.shape[1]
    fVi = np.arange(1, nf + 1)
    fM = np.empty((xMF.shape[0], 0))
    fVs = np.full((NoF,), np.nan)

    for i in range(NoF):
        mi = np.full((fVi.size,), np.nan)
        for j in range(fVi.size):
            idx = fVi[j] - 1
            noisy_feature = xMF[:, idx] + np.random.normal(0, 1e-5, xMF.shape[0])
            mi_local, _, _ = cmikra1n_cpu(xMC, noisy_feature, fM, k)
            mi[j] = mi_local

        max_idx = np.nanargmax(mi)
        best_feature = fVi[max_idx]
        fM = np.hstack((fM, xMF[:, best_feature - 1].reshape(-1, 1)))
        fVs[i] = best_feature
        fVi = np.delete(fVi, max_idx)

    return fVs.astype(int) - 1

model_dict = {
    "LogReg": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(),
    "NaiveBayes": GaussianNB(),
    "RandomForest": RandomForestClassifier(n_estimators=100)
}

for repeat in range(repeats):
    for NoF in nof_list:
        print(f"🔁 Run {repeat+1}/{repeats} - Top {NoF} Features")
        start_cpu = time.time()
        selected_cpu = MIfeatureselection0_cpu(y_scaled, X_scaled, k, NoF)
        cpu_time = time.time() - start_cpu
        X_selected = X_scaled[:, selected_cpu]
        y_cpu = y_raw.ravel()
        pd.DataFrame({"CPU_Selected_Features": selected_cpu}).to_csv(f"selected_features_cpu_{NoF}_run{repeat+1}.csv", index=False)
        for model_name, model in model_dict.items():
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42 + repeat)
            acc_scores = cross_val_score(model, X_selected, y_cpu, cv=cv, scoring='accuracy')
            acc_mean = np.mean(acc_scores)
            acc_std = np.std(acc_scores)
            results.append([model_name, "CPU", NoF, repeat + 1, cpu_time, acc_mean, acc_std])

df_outputs = pd.DataFrame(results, columns=["Model", "Mode", "NoF", "Run", "Time (s)", "Accuracy", "Std_Accuracy"])
df_outputs.to_csv("accuracy_comparison_cpu.csv", index=False)
print("✅ CPU Evaluation Complete. Results saved.")
