import argparse, pathlib
import pandas as pd, matplotlib.pyplot as plt
import networkx as nx
import signals
from schema import load_log

METHODS = ["baseline", "judge_mean", "combined_content", "combined_all"]

def auc_curve(path, methods=METHODS, output_path=None):
    path = pathlib.Path(path)
    if path.suffix.lower() == ".png" and output_path is None:
        output_path = path
        path = path.with_suffix(".csv")

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
    fig.tight_layout()
    output_path = pathlib.Path(output_path) if output_path is not None else path.with_suffix(".png")
    fig.savefig(output_path, dpi=150)

def mention_graph(log, meta, scores, path="results/graph.png"):
    G = nx.DiGraph(); G.add_nodes_from(meta["agents"])
    for m in log:
        for t in set(signals.ID.findall(m.content)):
            if t != m.sender and t in G: G.add_edge(m.sender, t)
    plt.figure(figsize=(6,6))
    nx.draw_networkx(G, node_color=[scores[a] for a in G.nodes], cmap="Reds",
                     node_size=500, font_size=8)
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an AUC plot or mention graph.")
    parser.add_argument("input", help="CSV/PNG AUC data or a JSONL run log")
    parser.add_argument("output", nargs="?", help="Optional output image path")
    args = parser.parse_args()

    input_path = pathlib.Path(args.input)
    if input_path.suffix.lower() == ".jsonl":
        log, meta = load_log(input_path)
        feats = signals.features(log, meta["tasks"], meta["agents"])
        scores = {agent: feats[agent]["mention_in"] for agent in meta["agents"]}
        output_path = args.output or pathlib.Path("results/mention_graphs") / f"{input_path.stem}_mention_graph.png"
        mention_graph(log, meta, scores, output_path)
    elif input_path.suffix.lower() in {".csv", ".png"}:
        auc_curve(input_path, output_path=args.output)
    else:
        parser.error("input must be a .jsonl run log, .csv data file, or .png AUC output path")
