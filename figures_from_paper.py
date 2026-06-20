#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figures_from_paper.py
Builds the three ContamPerc figures from the values reported in the paper.

  fig1.pdf  Canary percolation, one panel per model (FC curve + chain/star
            points at tau=10K).
  fig2.pdf  CGD leaderboard bar chart (one bar per model).
  fig3.pdf  Size control: GPT-OSS 120B vs 20B, canary and Tier-1.

Data provenance (matches the resubmission):
  - Canary point estimates (all five models in fig1, and the 120B panel of
    fig3) are the E1 phase-transition values carried over from the original
    submission. E1 was NOT re-run for these models, so these are the old
    reported canary numbers. 95% Wilson CIs use n=100.
  - Tier-1 levels and CGD come from the new symmetric re-measurement run.
  - The 20B panel of fig3 is the only re-run E1 (size control); its canary
    is read from E1_phase_transition.json and its Tier-1 from
    E2_tier1_semantic_gpt_oss_20b.json.

Note: the canary numbers here are the synthetic E1 sweep, which is separate
from the MIMIC-IV slice (canary_results.json / tier1_results.json); the two
are different experiments and are expected to differ.

Run: python figures_from_paper.py  ->  fig1.pdf, fig2.pdf, fig3.pdf
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OI = {"orange":"#E69F00","skyblue":"#56B4E9","green":"#009E73",
      "blue":"#0072B2","verm":"#D55E00","black":"#000000"}
C_CAN, C_T1 = OI["blue"], OI["verm"]
C_CHAIN, C_STAR = OI["orange"], OI["green"]

MODELS = ["GPT-OSS 120B","Claude Sonnet 4.6","Llama 4 Maverick",
          "Gemini 2.5 Flash","GPT-4o-mini"]
TAU = [500, 1000, 5000, 10000]
CGD   = {"GPT-OSS 120B":"+51","Claude Sonnet 4.6":"-13","Llama 4 Maverick":"-20",
         "Gemini 2.5 Flash":"-62","GPT-4o-mini":"-64"}
CLASS = {"GPT-OSS 120B":"CB","Claude Sonnet 4.6":"NZ","Llama 4 Maverick":"IF",
         "Gemini 2.5 Flash":"IF","GPT-4o-mini":"IF"}

# FC canary: (point, lo, hi) per tau, straight from the E7 ablation table
CANARY_FC = {
 "GPT-OSS 120B":      [(.04,.016,.098),(.09,.048,.162),(.03,.010,.085),(.01,.002,.054)],
 "Claude Sonnet 4.6": [(.95,.888,.978),(1.00,.963,1.0),(.99,.946,.998),(.96,.902,.984)],
 "Llama 4 Maverick":  [(1.00,.96,1.00)]*4,
 "Gemini 2.5 Flash":  [(1.00,.95,1.00)]*4,
 "GPT-4o-mini":       [(1.00,.963,1.0),(1.00,.963,1.0),(.98,.93,.994),(.76,.668,.833)],
}
# canary at other topologies, tau=10K (Finding-1 text)
TOPO_10K = {"GPT-4o-mini": {"chain":.30, "star":.11}}
ALL_TOPO_NOTE = {"Llama 4 Maverick":"chain, star: 100%",
                 "Gemini 2.5 Flash":"chain, star: 100%",
                 "GPT-OSS 120B":"blocked in all topologies"}

# Tier-1 in FC: ("flat", v) or ("range", lo, hi), from leaderboard / cross-domain
TIER1_FC = {
 "GPT-OSS 120B":      ("flat", .60),
 "Claude Sonnet 4.6": ("range", .90, .96),
 "Llama 4 Maverick":  ("flat", .80),
 "Gemini 2.5 Flash":  ("range", .32, .48),
 "GPT-4o-mini":       ("range", .07, .74),
}
T1_LABEL = {"GPT-OSS 120B":"T1 60%","Claude Sonnet 4.6":"T1 90-96%",
            "Llama 4 Maverick":"T1 80%","Gemini 2.5 Flash":"T1 32-48%",
            "GPT-4o-mini":"T1 7-74%"}

XLO, XHI = 400, 13000

def canary_xyerr(model):
    xs, ys, lo, hi = [], [], [], []
    for t, (p, l, u) in zip(TAU, CANARY_FC[model]):
        xs.append(t); ys.append(p); lo.append(max(0., p-l)); hi.append(max(0., u-p))
    return np.array(xs), np.array(ys), np.clip(np.array([lo, hi]), 0, None)

def style(ax, title):
    ax.set_title(title, fontsize=10)
    ax.set_xscale("log"); ax.set_xlim(XLO, XHI); ax.set_ylim(-0.04, 1.06)
    ax.set_xticks(TAU); ax.set_xticklabels(["500","1K","5K","10K"], fontsize=8)
    ax.set_xlabel(r"$\tau$ (payload size)", fontsize=9)
    ax.grid(True, which="major", linewidth=0.3, alpha=0.4)
    ax.tick_params(labelsize=8)

def make_fig1():
    plt.rcParams["font.family"] = "serif"
    fig, axes = plt.subplots(2, 3, figsize=(12, 7)); axes = axes.ravel()
    for ax, m in zip(axes, MODELS):
        x, y, err = canary_xyerr(m)
        ax.errorbar(x, y, yerr=err, color=C_CAN, marker="o", markersize=4,
                    capsize=2.5, linewidth=1.5, elinewidth=0.8, label="FC")
        if m in TOPO_10K:
            ax.scatter([10000], [TOPO_10K[m]["chain"]], color=C_CHAIN, marker="^",
                       s=42, zorder=5, label="chain (10K)")
            ax.scatter([10000], [TOPO_10K[m]["star"]], color=C_STAR, marker="s",
                       s=36, zorder=5, label="star (10K)")
        if m in ALL_TOPO_NOTE:
            ax.text(0.04, 0.06, ALL_TOPO_NOTE[m], transform=ax.transAxes,
                    fontsize=8, style="italic", color="0.35")
        style(ax, f"{m}  ({CLASS[m]}, CGD {CGD[m]})")
        ax.set_ylabel("Canary percolation", fontsize=9)
    leg = axes[5]; leg.axis("off")
    handles = [Line2D([0],[0], color=C_CAN, marker="o", lw=1.6, label="FC (curve, 95% Wilson CI)"),
               Line2D([0],[0], color=C_CHAIN, marker="^", lw=0, label="chain (tau=10K)"),
               Line2D([0],[0], color=C_STAR, marker="s", lw=0, label="star (tau=10K)")]
    leg.legend(handles=handles, loc="center", fontsize=10, frameon=False, title="Topology")
    fig.tight_layout(); fig.savefig("fig1.pdf", bbox_inches="tight")
    plt.close(fig); print("wrote fig1.pdf")

def make_fig2():
    """Figure 2: CGD leaderboard, horizontal bar chart with +/-15 band."""
    plt.rcParams["font.family"] = "serif"
    C_CB  = OI["green"]    # content-blind  (teal)
    C_NZ  = "#E0B040"      # near-zero      (gold)
    C_IF  = OI["verm"]     # instruction-following (orange)
    cls_color = {"CB": C_CB, "NZ": C_NZ, "IF": C_IF}

    vals = {m: int(CGD[m]) for m in MODELS}          # +51,-13,-20,-62,-64
    order = list(MODELS)                              # top-to-bottom as listed
    ypos = list(range(len(order)))[::-1]              # first model at top

    fig, ax = plt.subplots(figsize=(8.2, 4.6))

    # +/-15 classification band + zero line
    ax.axvspan(-15, 15, color="0.85", alpha=0.55, zorder=0,
               label="$\\pm$15 classification band")
    ax.axvline(0, color="0.35", linestyle="--", linewidth=1.0, zorder=1)

    # CI half-widths for the two boundary models (whiskers), from the gap CIs
    err = {"Claude Sonnet 4.6": 2.2}                  # ~[-15.5,-11.1] around -13

    for m, y in zip(order, ypos):
        c = cls_color[CLASS[m]]
        v = vals[m]
        ax.barh(y, v, height=0.62, color=c, edgecolor="0.2", linewidth=0.8, zorder=2)
        if m in err:
            ax.errorbar(v, y, xerr=err[m], color="black", capsize=3.5,
                        elinewidth=1.2, zorder=4)
        # bold value label, outside the bar end
        if v >= 0:
            ax.text(v + 2.0, y, f"+{v}", va="center", ha="left",
                    fontsize=12, fontweight="bold", zorder=5)
        else:
            ax.text(v - 2.0, y, f"{v}", va="center", ha="right",
                    fontsize=12, fontweight="bold", zorder=5)

    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{m}\n({CLASS[m]})" for m in order], fontsize=10)
    ax.set_xlim(-72, 64)
    ax.set_xticks([-60,-45,-30,-15,0,15,30,45,60])
    ax.set_xlabel("Contamination Gap Diagnostic (percentage points)", fontsize=11)
    ax.tick_params(axis="x", labelsize=9)
    for s in ["top","right"]:
        ax.spines[s].set_visible(False)

    # direction arrows above the plot
    ax.annotate("$\\leftarrow$ instruction-following", xy=(-15,1.02),
                xycoords=("data","axes fraction"), ha="right", va="bottom",
                fontsize=10.5, color=C_IF)
    ax.annotate("content-blind $\\rightarrow$", xy=(15,1.02),
                xycoords=("data","axes fraction"), ha="left", va="bottom",
                fontsize=10.5, color=C_CB)

    # legend
    handles = [Patch(facecolor=C_CB, edgecolor="0.2", label="Content-blind (CGD $> +15$)"),
               Patch(facecolor=C_NZ, edgecolor="0.2", label="Near-zero ($|$CGD$| \\leq 15$)"),
               Patch(facecolor=C_IF, edgecolor="0.2", label="Instruction-following (CGD $< -15$)"),
               Patch(facecolor="0.85", edgecolor="0.7", label="$\\pm$15 classification band")]
    ax.legend(handles=handles, loc="lower right", fontsize=9, frameon=True,
              framealpha=0.95)

    fig.tight_layout()
    fig.savefig("fig2.pdf", bbox_inches="tight")
    plt.close(fig); print("wrote fig2.pdf (CGD bar chart)")


def make_fig3():
    """Figure 3: GPT-OSS size control. 120B (left) vs 20B (right).
    120B canary uses the E1 sweep (max 9%), matching the headline CGD +51."""
    import numpy as _np
    plt.rcParams["font.family"] = "serif"
    C_CAN3, C_T13 = OI["blue"], OI["orange"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    def band(ax, lo, hi):
        ax.axhspan(lo, hi, color="0.80", alpha=0.55, zorder=0)
    def setup(ax, title):
        ax.set_title(title, fontsize=11)
        ax.set_xscale("log"); ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel(r"$\tau$ (payload size)", fontsize=10)
        ax.grid(True, which="major", linewidth=0.3, alpha=0.4)
        for s in ["top","right"]: ax.spines[s].set_visible(False)

    # ---- LEFT: GPT-OSS 120B (E1 canary, max 9%) ----
    tauL = [500, 1000, 5000, 10000]
    canL = [0.04, 0.09, 0.03, 0.01]
    canL_lo = [0.016, 0.048, 0.010, 0.002]; canL_hi = [0.098, 0.162, 0.085, 0.054]
    errL = _np.array([[c-l for c,l in zip(canL,canL_lo)], [u-c for c,u in zip(canL,canL_hi)]])
    band(axL, max(canL), 0.60)
    axL.axhline(0.60, color=C_T13, linewidth=2.2, label="Tier-1 (60%)", zorder=3)
    axL.errorbar(tauL, canL, yerr=errL, color=C_CAN3, marker="s", markersize=5,
                 linestyle="--", capsize=3, linewidth=1.5, elinewidth=0.9,
                 label="Canary", zorder=4)
    setup(axL, "GPT-OSS 120B (paper)")
    axL.set_xticks(tauL); axL.set_xticklabels(["500","1K","5K","10K"], fontsize=9)
    axL.set_ylabel("Contamination rate (FC)", fontsize=10)
    axL.text(0.40, 0.42, "CGD = +51\ncontent-blind", transform=axL.transAxes,
             fontsize=12, fontweight="bold", ha="center",
             bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.95))
    axL.legend(loc="center right", fontsize=9.5, frameon=True)

    # ---- RIGHT: GPT-OSS 20B (real E1 canary + E2 Tier-1) ----
    tauC = [500,1000,2000,3000,4000,5000,6000,7000,8000]
    canR = [0.12,0.16,0.10,0.14,0.07,0.13,0.11,0.07,0.12]
    canR_lo=[0.070,0.101,0.055,0.085,0.034,0.078,0.063,0.034,0.070]
    canR_hi=[0.198,0.244,0.174,0.221,0.137,0.210,0.186,0.137,0.198]
    errCan = _np.array([[c-l for c,l in zip(canR,canR_lo)], [u-c for c,u in zip(canR,canR_hi)]])
    tauT = [500,1000,2000,3000,5000,8000,10000,15000]
    t1R = [0.67,0.71,0.69,0.67,0.68,0.68,0.68,0.64]
    t1_lo=[0.573,0.615,0.594,0.573,0.583,0.583,0.583,0.542]
    t1_hi=[0.754,0.790,0.772,0.754,0.763,0.763,0.763,0.727]
    errT1 = _np.array([[c-l for c,l in zip(t1R,t1_lo)], [u-c for c,u in zip(t1R,t1_hi)]])
    band(axR, max(canR), sum(t1R)/len(t1R))
    axR.errorbar(tauT, t1R, yerr=errT1, color=C_T13, marker="o", markersize=5,
                 linewidth=2.0, capsize=3, elinewidth=0.9, label="Tier-1", zorder=4)
    axR.errorbar(tauC, canR, yerr=errCan, color=C_CAN3, marker="s", markersize=5,
                 linestyle="--", capsize=3, linewidth=1.5, elinewidth=0.9,
                 label="Canary", zorder=4)
    setup(axR, "GPT-OSS 20B (new run, E1+E2)")
    axR.set_xticks([500,1000,2000,5000,10000]); axR.set_xticklabels(["500","1K","2K","5K","10K"], fontsize=9)
    axR.text(0.38, 0.40, "CGD = +52\ncontent-blind\n(preliminary)", transform=axR.transAxes,
             fontsize=12, fontweight="bold", ha="center",
             bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.95))
    axR.legend(loc="center right", fontsize=9.5, frameon=True)

    fig.suptitle("Content-blind classification is size-independent within GPT-OSS: "
                 "120B (+51) vs 20B (+52)", fontsize=12, y=1.00)
    fig.tight_layout(rect=[0,0,1,0.97]); fig.savefig("fig3.pdf", bbox_inches="tight")
    plt.close(fig); print("wrote fig3.pdf (size control, 120B canary max 9%)")


if __name__ == "__main__":
    make_fig1()
    make_fig2()
    make_fig3()
