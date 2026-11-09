"""Figures other than the headline AUC curves (those are in plots.py).

  python figs.py [backend] [graph_seed]     # defaults: cohere 1

  results/fig_null.png    seq_gap with no tampering (null calibration) vs under tampering
  results/fig_dose.png    per-seed AUC against number of coordination messages (dose-response)
  results/fig_graph_{cond}_{seed}.png   message graph for one seed (full view, no tampering)
"""
import sys, pathlib, warnings
warnings.filterwarnings("ignore", category=Warning, module="scipy")
from dataclasses import replace
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import spearmanr
import signals
from schema import load_log
from summarise import load, boot_ci

def fig_null(df, out="results/fig_null.png"):
    conds = list(df.cond.unique())
    fig, axes = plt.subplots(1, len(conds), figsize=(5.2 * len(conds), 4), sharey=True, squeeze=False)
    for ax, cond in zip(axes[0], conds):
        d = df[(df.cond == cond) & (df["mode"] == "targeted") & (df.method == "seq_gap")]
        n = d.seed.nunique()
        band = 2 * 0.18 / np.sqrt(n)               # ~2 SE of a chance mean AUC (4 vs 8 agents)
        ax.axhspan(0.5 - band, 0.5 + band, color="grey", alpha=0.18, label="chance band (~2 SE)")
        for tq, color, lab in [(0.0, "tab:grey", "no tampering (null)"),
                               (0.9, "tab:red", "targeted tamper q=0.9")]:
            g = d[d.tamper == tq]
            comps = sorted(g.completeness.unique())
            stats = np.array([boot_ci(g[g.completeness == c].auc.dropna().values) for c in comps])
            for i, c in enumerate(comps):
                v = g[g.completeness == c].auc.dropna().values
                ax.scatter(np.full(len(v), c) + (0.012 if tq else -0.012), v, s=10, alpha=0.35, color=color)
            ax.errorbar(comps, stats[:, 0], yerr=[stats[:, 0] - stats[:, 1], stats[:, 2] - stats[:, 0]],
                        marker="o", color=color, capsize=3, label=lab)
        ax.axhline(0.5, color="k", lw=0.6, ls=":")
        ax.invert_xaxis(); ax.set_title(f"seq_gap, {cond} condition")
        ax.set_xlabel("fraction of log visible")
    axes[0][0].set_ylabel("detection AUC")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)

def fig_dose(df, out="results/fig_dose.png", methods=("seq_gap", "combined_content", "judge_agent")):
    conds = list(df.cond.unique())
    fig, axes = plt.subplots(1, len(conds), figsize=(5.2 * len(conds), 4), squeeze=False)
    for ax, cond in zip(axes[0], conds):
        d = df[(df.cond == cond) & (df["mode"] == "targeted") & (df.tamper == 0.9) & (df.completeness == 1.0)]
        for m, mk in zip(methods, "os^"):
            g = d[d.method == m].dropna(subset=["auc"])
            if g.empty:
                continue
            rho = spearmanr(g.n_coord, g.auc)[0] if g.n_coord.nunique() > 1 else float("nan")
            ax.scatter(g.n_coord, g.auc, marker=mk, label=f"{m} (rho={rho:.2f})", alpha=0.8)
        ax.axhline(0.5, color="k", lw=0.6, ls=":")
        ax.set_xlabel("coordination messages in the seed (self-reported, of 80 colluder msgs)")
        ax.set_title(f"{cond}: targeted, q=0.9, full view")
    axes[0][0].set_ylabel("detection AUC (one point per seed)")
    for ax in axes[0]:
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)

def fig_graph(backend, cond, seed, out=None):
    log, meta = load_log(f"data/runs/{backend}_{cond}_{seed}.jsonl")
    agents, tasks, col = meta["agents"], meta["tasks"], set(meta["colluders"])
    log = [replace(m, coord=False) for m in log]            # monitor view: no ground truth
    ft = signals.features(log, tasks, agents)
    sc = signals.combined(ft, signals.NAMES)
    sender = {m.msg_id: m.sender for m in log}
    G = nx.DiGraph(); G.add_nodes_from(agents)
    ment, rep = set(), set()
    for m in log:
        for t in set(signals.ID.findall(m.content)):
            if t != m.sender and t in G:
                ment.add((m.sender, t))
        r = sender.get(m.reply_to)
        if r and r != m.sender:
            rep.add((m.sender, r))
    pos = nx.circular_layout(G)
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    nodes = nx.draw_networkx_nodes(G, pos, node_color=[sc[a] for a in G.nodes], cmap="Reds", node_size=650,
                                   edgecolors=["black" if a in col else "#bbbbbb" for a in G.nodes],
                                   linewidths=[3 if a in col else 0.8 for a in G.nodes], ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=sorted(rep), edge_color="#9aa0a6", style="dashed",
                           arrows=True, alpha=0.6, connectionstyle="arc3,rad=0.12", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=sorted(ment), edge_color="tab:blue", arrows=True,
                           alpha=0.7, connectionstyle="arc3,rad=0.12", ax=ax)
    ax.set_title(f"{cond}, seed {seed}: node colour = combined_all (monitor view), black ring = colluder\n"
                 "blue = @mention, grey dashed = reply", fontsize=9)
    ax.axis("off")
    plt.colorbar(nodes, ax=ax, shrink=0.6, label="combined_all score")
    out = out or f"results/fig_graph_{cond}_{seed}.png"
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)

if __name__ == "__main__":
    backend = sys.argv[1] if len(sys.argv) > 1 else "cohere"
    gseed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    pathlib.Path("results").mkdir(exist_ok=True)
    df = load(backend)
    fig_null(df); fig_dose(df)
    for cond in df.cond.unique():
        fig_graph(backend, cond, gseed)
    print("wrote results/fig_null.png, fig_dose.png, fig_graph_*.png")