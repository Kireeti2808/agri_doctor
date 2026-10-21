"""Independent re-analysis of the Agri-Doctor Pro per-image records.

Reads only rows_*.json from the results folder and recomputes the paper's
headline statistics without using the notebook's own summaries.
Usage: python3 -I reanalysis.py RESULTS_DIR OUT_JSON
"""
import json
import sys
from math import sqrt
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

RES = Path(sys.argv[1])
OUT = Path(sys.argv[2])
RNG = np.random.default_rng(0)
B = 2000
TAU, AMIN = 50.0, 2


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * (c - h), 1), round(100 * (c + h), 1))


def load(name):
    return json.loads((RES / f"rows_{name}.json").read_text())


def aurc(score, correct):
    order = np.argsort(-score, kind="stable")
    err = 1 - correct[order]
    return float(np.mean(np.cumsum(err) / np.arange(1, len(err) + 1)))


def scores(rows):
    p = np.array([r["probs"] for r in rows])
    s = -np.sort(-p, axis=1)
    pred = p.argmax(1)
    msp = s[:, 0]
    tta = np.array([r["tta_probs"] for r in rows])  # n x 2 x k
    tta_mean = (p[np.arange(len(p)), pred] + tta[np.arange(len(p)), 0, pred] + tta[np.arange(len(p)), 1, pred]) / 3
    tta_agree = (tta.argmax(2) == pred[:, None]).sum(1)
    agree = np.array([r["agreement"] for r in rows])
    gate_pass = np.array([(not r["needs_review"]) for r in rows]).astype(float)
    ent = np.sum(np.where(p > 0, p * np.log(np.clip(p, 1e-12, 1)), 0), 1)
    eps = 1e-3  # lexicographic: discrete level first, softmax breaks ties
    return {
        "max_softmax": msp,
        "margin": s[:, 0] - s[:, 1],
        "neg_entropy": ent,
        "tta_mean_conf": tta_mean,
        "tta_agree_plus_conf": tta_agree + eps * msp,
        "quadrant_agree_plus_conf": agree + eps * msp,
        "gate_pass_plus_conf": gate_pass + eps * msp,
    }


def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return e


def analyse(rows):
    n = len(rows)
    correct = np.array([r["pred"] == r["true"] for r in rows]).astype(float)
    old_correct = np.array([r["pred_old"] == r["true"] for r in rows]).astype(float)
    k = int(correct.sum())
    out = {"n": n, "correct": k, "acc": round(100 * k / n, 2), "ci": wilson(k, n)}
    out["old_rule_correct"] = int(old_correct.sum())
    a = int(((correct == 1) & (old_correct == 0)).sum())
    b = int(((correct == 0) & (old_correct == 1)).sum())
    out["mcnemar_new_only_old_only"] = [a, b]
    out["mcnemar_p"] = binomtest(min(a, b), a + b, 0.5).pvalue if a + b else 1.0

    flagged = np.array([r["needs_review"] for r in rows])
    out["flagged"] = int(flagged.sum())
    out["errors_caught"] = int((flagged & (correct == 0)).sum())
    out["correct_flagged"] = int((flagged & (correct == 1)).sum())
    keep = ~flagged
    out["acc_unflagged"] = round(100 * correct[keep].mean(), 2)
    sc = scores(rows)
    m = int(keep.sum())
    top = np.argsort(-sc["max_softmax"], kind="stable")[:m]
    out["msp_acc_same_coverage"] = round(100 * correct[top].mean(), 2)
    diffs = []
    for _ in range(B):
        idx = RNG.integers(0, n, n)
        c, f, s = correct[idx], flagged[idx], sc["max_softmax"][idx]
        mm = int((~f).sum())
        if mm == 0:
            continue
        t = np.argsort(-s, kind="stable")[:mm]
        diffs.append(c[~f].mean() - c[t].mean())
    out["gate_minus_msp_ci_points"] = [round(100 * np.percentile(diffs, 2.5), 2), round(100 * np.percentile(diffs, 97.5), 2)]

    out["aurc"] = {kk: round(aurc(v, correct), 4) for kk, v in sc.items()}
    base = sc["max_softmax"]
    q = sc["quadrant_agree_plus_conf"]
    d = []
    for _ in range(B):
        idx = RNG.integers(0, n, n)
        d.append(aurc(q[idx], correct[idx]) - aurc(base[idx], correct[idx]))
    d = np.array(d)
    out["aurc_quadrant_minus_msp"] = round(aurc(q, correct) - aurc(base, correct), 4)
    out["aurc_quadrant_minus_msp_ci"] = [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]
    out["aurc_quadrant_minus_msp_one_sided_tail"] = round(float((d >= 0).mean()), 4)

    agree = np.array([r["agreement"] for r in rows])
    out["agree_hist_correct"] = [int(((agree == i) & (correct == 1)).sum()) for i in range(5)]
    out["agree_hist_wrong"] = [int(((agree == i) & (correct == 0)).sum()) for i in range(5)]
    conf = sc["max_softmax"]
    wrong = correct == 0
    out["wrong_conf_ge_0.9"] = int((conf[wrong] >= 0.9).sum())
    out["wrong_conf_median"] = round(float(np.median(conf[wrong])), 3) if wrong.any() else None
    out["ece_raw_15bin"] = round(ece(conf, correct), 3)

    leaves = [r for r in rows if r["true_crop"] in ("Maize", "Rice", "Cotton") and not r["true"].startswith("Weed")]
    rc = {}
    for r in leaves:
        rc[r["router_crop"]] = rc.get(r["router_crop"], 0) + 1
    out["router_leaf_images"] = len(leaves)
    out["router_calls"] = rc
    mis = [r for r in leaves if r["router_crop"] != r["true_crop"]]
    out["router_misrouted"] = len(mis)
    out["router_misrouted_ge75"] = sum(r["router_conf"] >= 75 for r in mis)
    out["router_misrouted_median_conf"] = round(float(np.median([r["router_conf"] for r in mis])), 1) if mis else None
    out["wrong_crop_auto_accepted"] = sum((not r["router_review"]) for r in mis)
    return out


def origin(rows):
    res = {}
    for r in rows:
        name = Path(r["file"]).name
        key = "IMG_" if name.startswith("IMG_") else ("aug_" if name.startswith("aug_") else "other")
        a = res.setdefault(key, [0, 0])
        a[0] += 1
        a[1] += r["pred"] == r["true"]
    return res


report = {
    "maize_field": analyse(load("maize_field__maize")),
    "rice_new": analyse(load("rice_id_test__rice_new")),
    "rice_old": analyse(load("rice_id_test__rice_old")),
}
new, old = load("rice_id_test__rice_new"), load("rice_id_test__rice_old")
report["origin_new"] = origin(new)
report["origin_old"] = origin(old)
cn = np.array([r["pred"] == r["true"] for r in new])
co = np.array([r["pred"] == r["true"] for r in old])
a, b = int((co & ~cn).sum()), int((cn & ~co).sum())
report["old_vs_new_discordant_old_only_new_only"] = [a, b]
report["old_vs_new_p"] = binomtest(min(a, b), a + b, 0.5).pvalue
img = np.array([Path(r["file"]).name.startswith("IMG_") for r in new])
a2, b2 = int((co & ~cn & img).sum()), int((cn & ~co & img).sum())
report["camera_originals_old_only_new_only"] = [a2, b2]
report["camera_originals_p"] = binomtest(min(a2, b2), a2 + b2, 0.5).pvalue
OUT.write_text(json.dumps(report, indent=1, default=float))
print(json.dumps(report, indent=1, default=float))
