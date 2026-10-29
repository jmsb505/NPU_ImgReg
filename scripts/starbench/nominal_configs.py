import json
import os

HOME = os.path.expanduser(os.environ["STARBENCH_HOME"])
ROOT = os.path.join(HOME, "results", "preproc")
with open(os.path.join(ROOT, "summary.json")) as f:
    prev = json.load(f)
with open(os.path.join(ROOT, "summary_rgb.json")) as f:
    rgb = json.load(f)
out = {
    "baseline": prev["raw"],
    "sar_median5": prev["median5"],
    "sar_median5_clahe": prev["median5_clahe"],
    "sar_median5_gradient": prev["median5_gradient"],
    "both_median5_clahe": rgb["sar_mc_rgb_mc"],
    "sar_clahe_rgb_clahe": rgb["sar_mc_rgb_c"],
}
with open(os.path.join(HOME, "results", "nominal_configs.json"), "w") as f:
    json.dump(out, f)
print("ok", {k: len(v) for k, v in out.items()})
