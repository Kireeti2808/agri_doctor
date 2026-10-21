"""Figures for the follow-up experiments (rice field adaptation, rice few-shot).
Usage: python3 -I figures_followup.py FOLLOWUP_DIR FIRST_RUN_DIR OUT_DIR
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FU, FIRST, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
COL, DCOL = 3.45, 7.16
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True, "lines.linewidth": 1.4,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})
C = ["Rice_Bacterial_Leaf_Blight", "Rice_Brown_Spot", "Rice_Healthy", "Rice_Leaf_Blast",
     "Rice_Leaf_scald", "Rice_Sheath_Blight", "Weed_Broadleaf", "Weed_Grass"]
SHORT = {"Rice_Bacterial_Leaf_Blight": "Bacterial\nblight", "Rice_Brown_Spot": "Brown\nspot", "Rice_Healthy": "Healthy",
         "Rice_Leaf_Blast": "Blast", "Rice_Leaf_scald": "Leaf\nscald", "Rice_Sheath_Blight": "Sheath\nblight"}
SEEDS = [42, 7, 2024]


def rows(k):
    return json.loads((FU / f"rows_field_test__{k}.json").read_text())


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf"); plt.close(fig)


# ---------------------------------------------------------------- adaptation
FIELD = ["Rice_Bacterial_Leaf_Blight", "Rice_Brown_Spot", "Rice_Leaf_Blast"]
LABC = ["Rice_Bacterial_Leaf_Blight", "Rice_Brown_Spot", "Rice_Healthy", "Rice_Leaf_Blast", "Rice_Leaf_scald", "Rice_Sheath_Blight"]
lab_chk = json.loads((FU / "lab_test_check.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(DCOL, 2.1), gridspec_kw={"width_ratios": [3, 6]})


def bars(ax, classes, getter, title):
    x = np.arange(len(classes)); w = 0.36
    for j, (kind, col, lab) in enumerate([("lab", ORANGE, "Laboratory only"), ("adapted", BLUE, "Laboratory + field pool")]):
        vals = np.array([[getter(kind, s, c) for c in classes] for s in SEEDS])
        ax.bar(x + (j - 0.5) * w, vals.mean(0), w, color=col, alpha=0.85, label=lab)
        for s in range(len(SEEDS)):
            ax.scatter(x + (j - 0.5) * w, vals[s], s=6, color=INK, zorder=3, linewidths=0)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[c] for c in classes]); ax.set_ylim(0, 105)
    ax.set_ylabel("Recall (%)"); ax.set_title(title, loc="left")


def field_recall(kind, s, c):
    r = rows(f"{kind}_seed{s}"); m = [x for x in r if x["true"] == c]
    return 100 * sum(x["pred"] == c for x in m) / len(m)


bars(axes[0], FIELD, field_recall, "(a) Held-out field photographs ($n = 900$)")
bars(axes[1], LABC, lambda kind, s, c: 100 * lab_chk[str(s)][kind]["per_class_recall"][c], "(b) Laboratory test split of each seed")
axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.28), ncol=2)
fig.tight_layout(); save(fig, "rice_adapt")

# ---------------------------------------------------------------- few-shot (maize + rice)
fsm = json.loads((FIRST / "few_shot.json").read_text())["maize_field"]["curve"]
fsr = json.loads((FU / "few_shot_rice.json").read_text())["curve"]
fig, axes = plt.subplots(1, 2, figsize=(DCOL, 1.9))
ks = sorted(fsm, key=int)
axes[0].errorbar([int(k) for k in ks], [100 * fsm[k]["accuracy_mean"] for k in ks], yerr=[100 * fsm[k]["accuracy_sd"] for k in ks],
                 fmt="-o", color=BLUE, ms=4, capsize=3, label="Linear head, frozen EfficientNetV2-B0")
axes[0].axhline(73.49, color=ORANGE, ls="--", lw=1.1, label="Deployed fine-tuned model (73.5%)")
axes[0].set_xticks([0, 5, 10, 20]); axes[0].set_ylim(55, 80)
axes[0].set_xlabel("Field photographs added per class"); axes[0].set_ylabel("Field accuracy (%)")
axes[0].set_title("(a) Maize, PlantDoc ($n = 83$)", loc="left"); axes[0].legend(loc="lower right")
kr = ["0", "5", "10", "20", "50", "100"]
xr = np.arange(len(kr) + 1)
axes[1].errorbar(xr[:-1], [100 * fsr[k]["accuracy_mean"] for k in kr], yerr=[100 * fsr[k]["accuracy_sd"] for k in kr],
                 fmt="-o", color=BLUE, ms=4, capsize=3, label="Linear head (3 classes)")
axes[1].plot([xr[-1]], [100 * fsr["all_pool"]["accuracy_mean"]], "o", color=BLUE, ms=4)
axes[1].axhline(97.15, color=AQUA, ls="--", lw=1.1, label="Fine-tuned on lab + field pool (97.2%)")
axes[1].axhline(23.19, color=ORANGE, ls="--", lw=1.1, label="Fine-tuned on lab only (23.2%)")
axes[1].set_xticks(xr); axes[1].set_xticklabels(kr + ["all\n(~994)"]); axes[1].set_ylim(0, 105)
axes[1].set_xlabel("Field photographs added per class"); axes[1].set_ylabel("Field accuracy (%)")
axes[1].set_title("(b) Rice, Sethy et al. held-out test ($n = 900$)", loc="left"); axes[1].text(0.1, 91, "fine-tuned, lab + field pool (97.2%)", color=AQUA, fontsize=7, va="top")
axes[1].text(0.1, 26, "fine-tuned, lab only (23.2%)", color=ORANGE, fontsize=7, va="bottom")
axes[1].text(2.1, 52, "linear head on frozen features (3 classes)", color=BLUE, fontsize=7, va="top")
fig.tight_layout(); save(fig, "few_shot")


print("figures written to", OUT)
