import json
import os

import numpy as np
import scipy.io as sio

HOME = os.path.expanduser(os.environ["STARBENCH_HOME"])
with open(os.path.join(HOME, "results", "nominal-intra_blue-green-gt.json")) as f:
    records = json.load(f)


def apply(H, pts):
    return pts @ H[:2, :2].T + H[:2, 2]


def rmse(a, b):
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


rows = []
for r in records[:12]:
    m = sio.loadmat(os.path.join(HOME, "data", "nominal", "Nominal", "Intra_blue-green", r["set"]))
    lf = np.asarray(m["Landmarks"][0][0][0], dtype=np.float64).reshape(-1, 2)
    lm = np.asarray(m["Landmarks"][0][0][1], dtype=np.float64).reshape(-1, 2)
    A = np.hstack([lm, np.ones((len(lm), 1))])
    X, *_ = np.linalg.lstsq(A, lf, rcond=None)
    F = X.T  # 2x3 affine, moving -> fixed
    T = np.asarray(r["T"], dtype=np.float64)
    ang = np.degrees(np.arctan2(F[1, 0], F[0, 0]))
    rows.append((r["set"], rmse(apply(F, lm), lf), rmse(apply(F, lm), apply(T, lm)), ang, F[:, 2]))
for name, fit_res, fit_vs_T, ang, t in rows:
    print(name, "fit residual", round(fit_res, 2), "fit vs T_reg_gt", round(fit_vs_T, 2), "angle", round(ang, 2), "t", np.round(t, 1))
