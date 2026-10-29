import hashlib
import json
import os
import sys

import numpy as np

HOME = os.path.expanduser(os.environ["STARBENCH_HOME"])
sys.path.insert(0, os.path.join(HOME, "vendor"))
sys.path.insert(0, HOME)
from starbench_detectors import XFeatDetector

MODALITIES = ["Intra_blue-red", "Intra_green-red", "OPT-OPT", "SAR_VH-RGB", "SAR_VH-VH", "SAR_VH-VV", "SAR_VV-RGB", "SAR_VV-VV"]


def tre(est, gt, lm):
    return float(np.sqrt(np.mean(np.sum((lm @ est[:, :2].T + est[:, 2] - (lm @ gt[:, :2].T + gt[:, 2])) ** 2, axis=1))))


def stats(rs):
    ok = np.array([r["tre_px"] for r in rs if r["tre_px"] is not None])
    inv = np.array([r["tre_inverse_px"] for r in rs if r["tre_inverse_px"] is not None])
    ran = [r for r in rs if r["seconds"] is not None]
    out = {"sets": len(rs), "failures": len(rs) - len(ok), "errors": sum(r["error"] is not None for r in rs)}
    if ok.size:
        out.update(
            tre_px_median=round(float(np.median(ok)), 3),
            tre_px_p90=round(float(np.percentile(ok, 90)), 3),
            tre_px_mean=round(float(ok.mean()), 3),
            fraction_tre_le_5px=round(float(np.mean(ok <= 5)), 3),
            diagnostic_inverse_tre_px_median=round(float(np.median(inv)), 3),
        )
    if ran:
        out.update(
            inliers_median=int(np.median([r["inliers"] for r in ran])),
            seconds_median=round(float(np.median([r["seconds"] for r in ran])), 4),
            seconds_total=round(float(sum(r["seconds"] for r in ran)), 2),
        )
    return out


with open(os.path.join(HOME, "results", "nominal-8mod-gt.json")) as f:
    data = json.load(f)
records = data["records"]

detector = XFeatDetector("cuda", ransac_thresh=5.0)
for r in records:
    r.update(inliers=0, seconds=None, tre_px=None, tre_inverse_px=None, error=None)
    try:
        result = detector.register(r["fixed"], r["moving"], ransac_thresh=5.0)
    except Exception as e:
        r["error"] = repr(e)[:200]
        continue
    r["inliers"] = int(result["matches_count"])
    r["seconds"] = float(result["time"])
    if result["transform"] is not None:
        M = np.asarray(result["transform"], dtype=np.float64)[:2, :]
        T3 = np.asarray(r["T"], dtype=np.float64)
        lm = np.asarray(r["landmarks_mov"], dtype=np.float64)
        r["tre_px"] = tre(M, T3[:2, :], lm)
        r["tre_inverse_px"] = tre(M, np.linalg.inv(T3)[:2, :], lm)

summary = {mod: stats([r for r in records if r["modality"] == mod]) for mod in MODALITIES}
summary["all_8"] = stats(records)

with open(os.path.join(HOME, "results", "nominal-8mod-xfeat.json"), "w") as f:
    json.dump({"records": records, "errors": data["errors"]}, f)

with open(os.path.join(HOME, "accelerated_features", "weights", "xfeat.pt"), "rb") as f:
    weights_sha256 = hashlib.sha256(f.read()).hexdigest()
prov = {
    "dataset": {
        "source": "Drive IRG-AEROSPACE@SIGMETRICS2026, Dataset_Processed/Nominal (export Nominal-20261027T104620Z-1-001.zip)",
        "zip_sha256": "fffd7846dc0ce740907a0a51b3dd111efeb54e73bf9ff7c8e60d59e110aaa08b",
        "modalities": MODALITIES,
        "sets": len(records),
        "gt_load_errors": len(data["errors"]),
        "pair": "moving = SO<n>a.png, fixed = SO<n>b.png; ground truth = T_reg_gt and the landmarks in SO<n>.mat",
    },
    "code": {
        "starbench": "necst/STAR-Bench@0e3c9bf05506f6e398723d0de9dfacf9cf89cf53",
        "accelerated_features": "verlab/accelerated_features@e92685f57f8318b18725c5c8c0bd28c7fe188d9a",
        "xfeat_weights_sha256": weights_sha256,
    },
    "runtime": {
        "board": "Jetson Xavier NX, L4T R35.6.5 / JetPack 5.1.7",
        "python_env": "xfeat-jp517-py38 (PyTorch 2.1.0a0+41361538.nv23.06)",
        "device": "cuda",
        "ransac_thresh_px": 5.0,
        "timing": "XFeatDetector.register() time per pair, as returned by the detector",
    },
    "metric": "TRE: RMSE in pixels over the landmarks, estimated vs T_reg_gt, as in starbench_registration._calculate_rmse",
    "summary": summary,
}
with open(os.path.join(HOME, "results", "nominal-8mod-provenance.json"), "w") as f:
    json.dump(prov, f, indent=1)

for mod in MODALITIES + ["all_8"]:
    s = summary[mod]
    print(mod, "sets", s["sets"], "fail", s["failures"], "err", s["errors"], "TRE med", s.get("tre_px_median"),
          "p90", s.get("tre_px_p90"), "<=5px", s.get("fraction_tre_le_5px"), "inliers", s.get("inliers_median"),
          "sec", s.get("seconds_median"), "inv", s.get("diagnostic_inverse_tre_px_median"))
