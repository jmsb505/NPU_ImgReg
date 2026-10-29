import json
import os

import cv2
import numpy as np

HOME = os.path.expanduser(os.environ["STARBENCH_HOME"])
with open(os.path.join(HOME, "results", "nominal-8mod-xfeat.json")) as f:
    records = json.load(f)["records"]

for r in records:
    if r["tre_px"] is not None:
        continue
    print(r["modality"], r["set"], "inliers", r["inliers"], "seconds", r["seconds"])
    for key in ["moving", "fixed"]:
        im = cv2.imread(r[key], cv2.IMREAD_GRAYSCALE)
        print("   ", key, os.path.basename(r[key]), im.shape, "min", int(im.min()), "max", int(im.max()), "unique", int(np.unique(im).size))
