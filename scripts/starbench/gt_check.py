import json
import os

import cv2
import numpy as np

HOME = os.path.expanduser(os.environ["STARBENCH_HOME"])
OUT = os.path.join(HOME, "results", "pixelcheck")
with open(os.path.join(HOME, "results", "nominal-intra_blue-green-gt.json")) as f:
    intra = json.load(f)
for r in intra:
    r["modality"] = "Intra_blue-green"
with open(os.path.join(HOME, "results", "nominal-8mod-gt.json")) as f:
    recs = intra + json.load(f)["records"]
with open(os.path.join(OUT, "pixelcheck.json")) as f:
    chk = json.load(f)
assert len(chk) == len(recs)


def pick(mod, cond):
    return next(i for i, c in enumerate(chk) if c["modality"] == mod and cond(c))


idx = [
    pick("Intra_blue-green", lambda c: True),
    pick("SAR_VH-RGB", lambda c: c["dense_px"] is not None and c["dense_px"] <= 5),
    pick("SAR_VH-RGB", lambda c: c["dense_px"] is not None and c["dense_px"] > 20),
]
rows = []
for i in idx:
    r = recs[i]
    b = cv2.imread(r["fixed"], cv2.IMREAD_GRAYSCALE)
    g = cv2.imread(r["moving"][:-5] + "g.png", cv2.IMREAD_GRAYSCALE)
    row = np.hstack([np.dstack([b, b, b]), np.dstack([g, g, g]), np.dstack([g, g, b])])
    label = f"{r['modality']} {r['set']} golden vs fixed"
    cv2.putText(row, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    rows.append(row)
cv2.imwrite(os.path.join(OUT, "gt_overlays.png"), np.vstack(rows))
print("saved", os.path.join(OUT, "gt_overlays.png"), flush=True)
