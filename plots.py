import pandas as pd, matplotlib.pyplot as plt
import networkx as nx
import signals

METHODS = ["baseline", "combined_content", "combined_all"]

def auc_curve(path="results/results.csv", methods=METHODS):
    df = pd.read_csv(path)
    modes = sorted(df["mode"].unique())
    tqs = [t for t in sorted(df.tamper.unique()) if t > 0]
    fig, axes = plt.subplots(len(modes), len(tqs), figsize=(5 * len(tqs), 3.8 * len(modes)), 
                             sharey=True, squeeze=False)
    for i, mode in enumerate(modes):
        for j, tq in enumerate(tqs):
            ax = axes[i][j]
            for m in methods:
                g = df[(df.method == m) & (df.tamper == tq) & (df["mode"] == mode)] \
                    .groupby("completeness").auc
                mu, sd = g.mean(), g.std()
                ax.errorbar(mu.index, mu.values, yerr=sd.values, marker="o", label=m, capsize=3)
            ax.set_title(f"{mode}, tamper = {tq}")
            ax.invert_xaxis()
            if i == len(modes) - 1: ax.set_xlabel("fraction of log visible")
            if j == 0: ax.set_ylabel("detection AUC")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout; fig.savefig("results/auc_vs_completeness.png", dpi=150)

def mention_graph(log, meta, scores, path="results/graph.png"):
    G = nx.DiGraph(); G.add_nodes_from(meta["agents"])
    for m in log:
        for t in set(signals.ID.findall(m.content)):
            if t != m.sender and t in G: G.add_edge(m.sender, t)
    plt.figure(figsize=(6,6))
    nx.draw_networkx(G, node_color=[scores[a] for a in G.nodes], cmap="Reds", 
                     node_size=500, font_size=8)
    plt.savefig(path, dpi=150)

if __name__ == "__main__":
    auc_curve()