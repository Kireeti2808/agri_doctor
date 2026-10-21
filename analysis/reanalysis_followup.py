"""Independent re-analysis of the follow-up records (rice field adaptation, quadrant rules).
Written separately from the notebook's exp_lib/followup_lib; numpy + scipy only.
Usage: python3 -I reanalysis_followup.py RESULTS_DIR OUT_JSON
"""
import json
import sys
from math import comb, sqrt
from pathlib import Path

import numpy as np

RES, OUT = Path(sys.argv[1]), Path(sys.argv[2])
C = ["Rice_Bacterial_Leaf_Blight", "Rice_Brown_Spot", "Rice_Healthy", "Rice_Leaf_Blast",
     "Rice_Leaf_scald", "Rice_Sheath_Blight", "Weed_Broadleaf", "Weed_Grass"]
FIELD = ["Rice_Bacterial_Leaf_Blight", "Rice_Brown_Spot", "Rice_Leaf_Blast"]
SEEDS = [42, 7, 2024]
REPS = 2000


def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def mcnemar(a, b):
    x, y = int(np.sum(a & ~b)), int(np.sum(~a & b)); n = x + y
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n) if n else 1.0
    return {"a_only": x, "b_only": y, "p": p}


def aurc(s, c):
    o = np.argsort(-s, kind="stable"); e = (~c[o]).astype(float)
    return float(np.mean(np.cumsum(e) / np.arange(1, len(e) + 1)))


def boot(fn, n, seed=0):
    rng = np.random.default_rng(seed)
    v = [fn(rng.integers(0, n, n)) for _ in range(REPS)]
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


def load(k):
    return json.loads((RES / f"rows_field_test__{k}.json").read_text())


def arrays(rows):
    P = np.array([r["probs"] for r in rows]); y = np.array([C.index(r["true"]) for r in rows])
    pred = P.argmax(1); c = pred == y; msp = P.max(1)
    agree = np.array([r["agreement"] for r in rows])
    contra = np.array([sum(1 for l in r["quad_labels"] if l != C[k] and "Healthy" not in l)
                       for r, k in zip(rows, pred)])
    healthy_q = np.array([sum("Healthy" in l for l in r["quad_labels"]) for r in rows])
    return dict(P=P, y=y, pred=pred, c=c, msp=msp, quad=agree + msp, hn=(4 - contra) + msp,
                agree=agree, contra=contra, healthy_q=healthy_q)


models = ["deployed"] + [f"{k}_seed{s}" for s in SEEDS for k in ("lab", "adapted")]
R = {k: load(k) for k in models}
files = [r["file"] for r in R["deployed"]]
assert all([r["file"] for r in R[k]] == files for k in models), "models were not tested on the same photographs"
A = {k: arrays(R[k]) for k in models}
out = {"n": len(files), "class_counts": {c: int(sum(r["true"] == c for r in R["deployed"])) for c in FIELD}}

# accuracy and per-class recall
out["accuracy"] = {k: {"k": int(a["c"].sum()), "pct": round(100 * a["c"].mean(), 2), "ci95": wilson(int(a["c"].sum()), len(a["c"]))}
                   for k, a in A.items()}
out["recall"] = {k: {c: int(((a["y"] == C.index(c)) & a["c"]).sum()) for c in FIELD} for k, a in A.items()}
for kind in ("lab", "adapted"):
    v = [out["accuracy"][f"{kind}_seed{s}"]["pct"] for s in SEEDS]
    out[f"{kind}_mean_sd"] = [round(float(np.mean(v)), 2), round(float(np.std(v, ddof=1)), 2)]
out["mcnemar_lab_vs_adapted"] = {s: mcnemar(A[f"lab_seed{s}"]["c"], A[f"adapted_seed{s}"]["c"]) for s in SEEDS}
out["mcnemar_deployed_vs_lab42"] = mcnemar(A["deployed"]["c"], A["lab_seed42"]["c"])
out["top_errors_lab42"] = {}
from collections import Counter
for k in ("lab_seed42", "adapted_seed42", "deployed"):
    a = A[k]
    out[f"top_errors_{k}"] = Counter(f"{C[t]} -> {C[p]}" for t, p, ok in zip(a["y"], a["pred"], a["c"]) if not ok).most_common(6)

# confident errors
out["errors_msp_ge_0.9"] = {k: [int(((a["msp"] >= 0.9) & ~a["c"]).sum()), int((~a["c"]).sum())] for k, a in A.items()}

# AURC: MSP, quadrant agreement, healthy-neutral; paired differences
out["aurc"] = {}
for k, a in A.items():
    c = a["c"]; n = len(c)
    d = {name: round(aurc(a[name], c), 4) for name in ("msp", "quad", "hn")}
    d["quad_minus_msp_ci"] = boot(lambda i: aurc(a["quad"][i], c[i]) - aurc(a["msp"][i], c[i]), n)
    d["hn_minus_msp_ci"] = boot(lambda i: aurc(a["hn"][i], c[i]) - aurc(a["msp"][i], c[i]), n)
    d["hn_minus_quad_ci"] = boot(lambda i: aurc(a["hn"][i], c[i]) - aurc(a["quad"][i], c[i]), n)
    d["errors"] = int((~c).sum())
    out["aurc"][k] = d

# mechanism: healthy-looking quadrants, agreement on correct verdicts
out["mechanism"] = {}
for k in ("deployed", "lab_seed42", "adapted_seed42"):
    a = A[k]; bs = a["y"] == C.index("Rice_Brown_Spot")
    out["mechanism"][k] = {
        "healthy_quadrants_mean_all": round(float(a["healthy_q"].mean()), 2),
        "brown_spot_correct_n": int((bs & a["c"]).sum()),
        "brown_spot_correct_healthy_quadrants_mean": round(float(a["healthy_q"][bs & a["c"]].mean()), 2) if (bs & a["c"]).any() else None,
        "correct_with_agreement_below_2": [int(((a["agree"] < 2) & a["c"]).sum()), int(a["c"].sum())],
        "wrong_with_agreement_4": [int(((a["agree"] == 4) & ~a["c"]).sum()), int((~a["c"]).sum())],
        "correct_with_no_contradiction": [int(((a["contra"] == 0) & a["c"]).sum()), int(a["c"].sum())],
        "wrong_with_no_contradiction": [int(((a["contra"] == 0) & ~a["c"]).sum()), int((~a["c"]).sum())],
    }

# laboratory test check, few-shot, split report (read through)
out["lab_test_check"] = json.loads((RES / "lab_test_check.json").read_text())
out["few_shot"] = json.loads((RES / "few_shot_rice.json").read_text())["curve"]
out["split"] = json.loads((RES / "field_split.json").read_text())["report"]
for s in SEEDS:
    m = json.loads((RES / f"adapted_seed{s}" / "metrics.json").read_text())
    out.setdefault("training", {})[s] = {"train_images": m["train_images"], "val_images": m["val_images"],
                                         "minutes": m["minutes"], "epochs": [len(m["history"]["head"]["loss"]),
                                                                            len((m["history"]["finetune"] or {}).get("loss", []))]}
OUT.write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: out[k] for k in ("accuracy", "lab_mean_sd", "adapted_mean_sd", "mcnemar_lab_vs_adapted")}, indent=1))
print(json.dumps(out["aurc"], indent=1)); print(json.dumps(out["mechanism"], indent=1))
print(out["errors_msp_ge_0.9"]); print(out["recall"]); print(out["training"])
print(out["top_errors_lab_seed42"]); print(out["top_errors_deployed"]); print(out["mcnemar_deployed_vs_lab42"])
