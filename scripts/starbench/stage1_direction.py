import json
import os

import numpy as np

HOME = os.path.expanduser(os.environ["STARBENCH_HOME"])
with open(os.path.join(HOME, "results", "nominal-intra_blue-green-gt.json")) as f:
    records = json.load(f)


def apply(H, pts):
    return pts @ H[:2, :2].T + H[:2, 2]


def rmse(a, b):
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


import scipy.io as sio

res = {"T(lm)~lf": [], "inv(T)(lm)~lf": [], "T(lf)~lm": [], "inv(T)(lf)~lm": []}
for r in records[:12]:
    mat = os.path.join(HOME, "data", "nominal", "Nominal", "Intra_blue-green", r["set"])
    m = sio.loadmat(mat)
    lf = np.asarray(m["Landmarks"][0][0][0], dtype=np.float64).reshape(-1, 2)
    lm = np.asarray(m["Landmarks"][0][0][1], dtype=np.float64).reshape(-1, 2)
    T = np.asarray(r["T"], dtype=np.float64)
    Ti = np.linalg.inv(T)
    res["T(lm)~lf"].append(rmse(apply(T, lm), lf))
    res["inv(T)(lm)~lf"].append(rmse(apply(Ti, lm), lf))
    res["T(lf)~lm"].append(rmse(apply(T, lf), lm))
    res["inv(T)(lf)~lm"].append(rmse(apply(Ti, lf), lm))
for k, v in res.items():
    print(k, "median", round(float(np.median(v)), 3), "max", round(float(np.max(v)), 3))
print("lf range", lf.min(axis=0), lf.max(axis=0), "lm range", lm.min(axis=0), lm.max(axis=0))
