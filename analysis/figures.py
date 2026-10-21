"""Generate all data figures for the manuscript from the per-image records.
Usage: python3 -I figures.py RESULTS_DIR OUT_DIR
"""
import glob
import json
import sys
from math import sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
COL = 3.45  # IEEE single-column width (in)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True, "lines.linewidth": 1.4,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})


def rows(name):
    return json.loads((RES / f"rows_{name}.json").read_text())


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


maize, rice = rows("maize_field__maize"), rows("rice_id_test__rice_new")


def arrays(rs):
    p = np.array([r["probs"] for r in rs])
    corr = np.array([r["pred"] == r["true"] for r in rs]).astype(float)
    msp = p.max(1)
    tta = np.array([r["tta_probs"] for r in rs])
    tta_mean = ((p + tta[:, 0] + tta[:, 1]) / 3).max(1)
    agree = np.array([r["agreement"] for r in rs])
    flag = np.array([r["needs_review"] for r in rs])
    return dict(p=p, corr=corr, msp=msp, tta=tta_mean, agree=agree, flag=flag)


A = {"maize": arrays(maize), "rice": arrays(rice)}


def rc_curve(score, corr):
    o = np.argsort(-score, kind="stable")
    e = 1 - corr[o]
    n = len(e)
    return np.arange(1, n + 1) / n, np.cumsum(e) / np.arange(1, n + 1)


# Fig: risk-coverage
fig, axes = plt.subplots(1, 2, figsize=(2 * COL, 2.0))
for ax, key, title in [(axes[0], "maize", "(a) Maize field photographs (n = 83)"),
                       (axes[1], "rice", "(b) Rice held-out test (n = 1,391)")]:
    a = A[key]
    for score, lab, col, ls in [(a["msp"], "Max softmax", BLUE, "-"),
                                (a["tta"], "Flip-TTA mean confidence", ORANGE, "--"),
                                (a["agree"] + 1e-3 * a["msp"], "Quadrant agreement + confidence", AQUA, "-.")]:
        x, y = rc_curve(score, a["corr"])
        ax.plot(x, 100 * y, ls, color=col, label=lab)
    keep = ~a["flag"]
    ax.plot([keep.mean()], [100 * (1 - a["corr"][keep].mean())], marker="*", ms=9, color=INK,
            ls="none", label=r"Deployed gate ($\tau$ = 0.5, $a_{\min}$ = 2)")
    ax.set_xlabel("Coverage (share of images answered)")
    ax.set_ylabel("Error on answered images (%)")
    ax.set_title(title)
    ax.set_xlim(0, 1.01)
    ax.set_ylim(bottom=0)
axes[1].legend(loc="upper left")
fig.tight_layout()
save(fig, "risk_coverage")

# Fig: quadrant agreement distribution
fig, axes = plt.subplots(1, 2, figsize=(2 * COL, 1.8))
for ax, key, title in [(axes[0], "maize", "(a) Maize field photographs"), (axes[1], "rice", "(b) Rice held-out test")]:
    a = A[key]
    c, w = a["corr"] == 1, a["corr"] == 0
    hc = np.array([np.sum(a["agree"][c] == i) for i in range(5)]) / c.sum() * 100
    hw = np.array([np.sum(a["agree"][w] == i) for i in range(5)]) / w.sum() * 100
    x = np.arange(5)
    ax.bar(x - 0.19, hc, 0.36, color=BLUE, label=f"Correct (n = {int(c.sum())})")
    ax.bar(x + 0.19, hw, 0.36, color=ORANGE, hatch="////", edgecolor="white", linewidth=0,
           label=f"Wrong (n = {int(w.sum())})")
    ax.axvline(1.5, color=INK2, ls=":", lw=0.8)
    ax.set_xticks(x)
    ax.set_xlabel("Quadrants agreeing with the whole-leaf label, $a$")
    ax.set_ylabel("Share of verdicts (%)")
    ax.set_title(title)
    ax.legend(loc="upper left")
fig.tight_layout()
save(fig, "agreement")

# Fig: gate sweep
fig, axes = plt.subplots(1, 2, figsize=(2 * COL, 2.0))
taus = np.linspace(0.30, 0.99, 70)
styles = [(0, INK2, ":"), (1, BLUE, "-"), (2, ORANGE, "--"), (3, AQUA, "-."), (4, YELLOW, "-")]
for ax, key, title in [(axes[0], "maize", "(a) Maize field photographs"), (axes[1], "rice", "(b) Rice held-out test")]:
    a = A[key]
    for amin, col, ls in styles:
        cov, acc = [], []
        for t in taus:
            keep = (a["msp"] >= t) & (a["agree"] >= amin)
            if keep.sum() >= 5:
                cov.append(keep.mean())
                acc.append(100 * a["corr"][keep].mean())
        lab = r"$a_{\min}$ = 0 (softmax only)" if amin == 0 else rf"$a_{{\min}}$ = {amin}"
        ax.plot(cov, acc, ls, color=col, label=lab)
    keep = ~a["flag"]
    ax.plot([keep.mean()], [100 * a["corr"][keep].mean()], "*", ms=9, color=INK, label="Deployed setting")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Accuracy on answered images (%)")
    ax.set_title(title)
axes[1].legend(loc="lower left", ncol=2)
fig.tight_layout()
save(fig, "gate_sweep")

# Fig: reliability (raw)
fig, axes = plt.subplots(1, 2, figsize=(2 * COL, 1.9))
for ax, key, title in [(axes[0], "maize", "(a) Maize field photographs"), (axes[1], "rice", "(b) Rice held-out test")]:
    a = A[key]
    edges = np.linspace(0, 1, 11)
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (a["msp"] > lo) & (a["msp"] <= hi)
        if m.sum() >= 3:
            xs.append(a["msp"][m].mean())
            ys.append(a["corr"][m].mean())
            ns.append(int(m.sum()))
    ax.plot([0, 1], [0, 1], ":", color=INK2, lw=0.8)
    ax.plot(xs, ys, "-o", color=BLUE, ms=3.5)
    for x, y, n in zip(xs, ys, ns):
        ax.annotate(str(n), (x, y), textcoords="offset points", xytext=(3, -9), fontsize=6, color=INK2)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Mean confidence in bin")
    ax.set_ylabel("Accuracy in bin")
    ax.set_title(title)
fig.tight_layout()
save(fig, "reliability")

# Fig: router confidence on rice leaves
leaves = [r for r in rice if not r["true"].startswith("Weed")]
conf_r = np.array([r["router_conf"] for r in leaves if r["router_crop"] == "Rice"])
conf_m = np.array([r["router_conf"] for r in leaves if r["router_crop"] == "Maize"])
fig, ax = plt.subplots(figsize=(COL, 1.9))
bins = np.arange(30, 102.5, 5)
ax.hist(conf_r, bins=bins - 1.1, rwidth=0.42, color=BLUE, label=f"Routed to rice (n = {len(conf_r)})")
ax.hist(conf_m, bins=bins + 1.1, rwidth=0.42, color=ORANGE, hatch="////", edgecolor="white", linewidth=0,
        label=f"Routed to maize (n = {len(conf_m)})")
ax.axvline(75, color=INK, ls="--", lw=0.9)
ax.text(74, ax.get_ylim()[1] * 0.55, r"$\theta$ = 75%", ha="right", fontsize=7, color=INK)
ax.set_xlabel("Router confidence (%)")
ax.set_ylabel("Rice leaf images")
ax.legend(loc="upper left")
fig.tight_layout()
save(fig, "router")

# Fig: confusion matrices
an = json.loads((RES / "analysis.json").read_text())
short = {"Rice_Bacterial_Leaf_Blight": "Bact. leaf blight", "Rice_Brown_Spot": "Brown spot", "Rice_Healthy": "Healthy",
         "Rice_Leaf_Blast": "Leaf blast", "Rice_Leaf_scald": "Leaf scald", "Rice_Sheath_Blight": "Sheath blight",
         "Weed_Broadleaf": "Broadleaf weed", "Weed_Grass": "Grass weed", "Maize_Blight": "Blight (NLB)",
         "Maize_Common_Rust": "Common rust", "Maize_Gray_Leaf_Spot": "Gray leaf spot", "Maize_Healthy": "Healthy"}


def cm_plot(ax, key, rows_keep=None, title=""):
    c = an[key]["confusion"]
    m = np.array(c["matrix"], dtype=float)
    labs = [short[l] for l in c["labels"]]
    rl = list(range(len(labs))) if rows_keep is None else rows_keep
    m2 = m[rl]
    norm = m2 / np.maximum(m2.sum(1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(m2.shape[0]):
        for j in range(m2.shape[1]):
            if m2[i, j] > 0:
                ax.text(j, i, int(m2[i, j]), ha="center", va="center", fontsize=6.5,
                        color="white" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs, rotation=40, ha="right")
    ax.set_yticks(range(len(rl)))
    ax.set_yticklabels([labs[i] for i in rl])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    ax.grid(False)


fig, axes = plt.subplots(1, 2, figsize=(2 * COL, 2.6), gridspec_kw={"width_ratios": [1.25, 1]})
cm_plot(axes[0], "rice_id_test__rice_new", title="(a) Rice, held-out test (n = 1,391)")
cm_plot(axes[1], "maize_field__maize", rows_keep=[0, 1, 2], title="(b) Maize, PlantDoc field (n = 83)")
fig.tight_layout()
save(fig, "confusion")

# Fig: lab vs field recall (maize) -- lab counts from the validation run (Table: maize lab)
lab = {"Blight (NLB)": (212, 229), "Common rust": (260, 261), "Gray leaf spot": (91, 114)}
fld = {"Blight (NLB)": (39, 51), "Common rust": (18, 20), "Gray leaf spot": (4, 12)}
fig, ax = plt.subplots(figsize=(COL, 1.9))
x = np.arange(3)
for off, d, col, lab_, hatch in [(-0.19, lab, BLUE, "Laboratory validation", None),
                                 (0.19, fld, ORANGE, "PlantDoc field", "////")]:
    k = np.array([v[0] for v in d.values()])
    n = np.array([v[1] for v in d.values()])
    lo, hi = zip(*[wilson(a, b) for a, b in zip(k, n)])
    y = 100 * k / n
    ax.bar(x + off, y, 0.36, color=col, hatch=hatch, edgecolor="white", linewidth=0, label=lab_)
    ax.errorbar(x + off, y, yerr=[y - 100 * np.array(lo), 100 * np.array(hi) - y], fmt="none", ecolor=INK, lw=0.8, capsize=2)
ax.set_xticks(x)
ax.set_xticklabels(list(lab))
ax.set_ylabel("Recall (%)")
ax.set_ylim(0, 122)
ax.set_yticks([0, 25, 50, 75, 100])
ax.legend(loc="upper center", ncol=2)
fig.tight_layout()
save(fig, "lab_field")

# Fig: backbones
fig, ax = plt.subplots(figsize=(COL, 2.0))
for name, lab_, col, mk in [("MobileNetV3Large", "MobileNetV3-Large (3.0 M)", BLUE, "o"),
                            ("EfficientNetV2B0", "EfficientNetV2-B0 (5.9 M)", ORANGE, "s"),
                            ("ConvNeXtTiny", "ConvNeXt-Tiny (27.8 M)", AQUA, "^")]:
    acc, lat = [], []
    for f in sorted(glob.glob(str(RES / f"backbone_{name}_seed*.json"))):
        d = json.loads(Path(f).read_text())
        acc.append(100 * d["test"]["accuracy"])
        lat.append(d["cpu_latency_ms_median_mixed_precision"])
    ax.scatter(lat, acc, s=14, color=col, marker=mk, alpha=0.55, linewidths=0)
    ax.errorbar([np.mean(lat)], [np.mean(acc)], yerr=[np.std(acc, ddof=1)], fmt=mk, color=col, ms=6,
                capsize=3, lw=1, label=lab_)
ax.set_xscale("log")
ax.set_xticks([300, 500, 1000, 1650])
ax.set_xticklabels(["300", "500", "1000", "1650"])
ax.minorticks_off()
ax.set_xlabel("CPU latency per image, ms (log scale)")
ax.set_ylabel("Test accuracy (%)")
ax.legend(loc="lower right")
fig.tight_layout()
save(fig, "backbones")

# Fig: few-shot
fs = json.loads((RES / "few_shot.json").read_text())["maize_field"]["curve"]
ks = sorted(fs, key=int)
fig, ax = plt.subplots(figsize=(COL, 1.8))
ax.errorbar([int(k) for k in ks], [100 * fs[k]["accuracy_mean"] for k in ks],
            yerr=[100 * fs[k]["accuracy_sd"] for k in ks], fmt="-o", color=BLUE, ms=4, capsize=3,
            label="Linear head on frozen EfficientNetV2-B0")
ax.axhline(73.49, color=ORANGE, ls="--", lw=1.1, label="Deployed fine-tuned model (73.5%)")
ax.set_xticks([0, 5, 10, 20])
ax.set_xlabel("Field photographs added per class")
ax.set_ylabel("Field accuracy (%)")
ax.set_ylim(55, 80)
ax.legend(loc="lower right")
fig.tight_layout()
save(fig, "few_shot")
print("figures written to", OUT)
