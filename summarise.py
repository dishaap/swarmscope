"""Summary statistics for the results CSVs.

  python summarise.py [backend]        # default: cohere

Reads results/results_{backend}_{id,noid}.csv plus the cached logs (for per-seed coordination
counts) and writes, under results/:
  seed_table.csv        per seed: cond, n_coord, compliant
  summary_all.csv       mean AUC + 95% seed-bootstrap CI per cell and method, ALL seeds (primary)
  summary_compliant.csv same, only seeds with n_coord >= config.COMPLIANT_MIN (secondary)
  paired_all.csv / paired_compliant.csv   paired differences between methods, per cell
  headline_all.md / headline_compliant.md  compact tables for the report

Seeds are the unit of replication: bootstrap resamples seeds. p-values (Wilcoxon, needs >= 6
seeds) are descriptive: there are many cells, so rely on the CI excluding 0, not on p."""
import sys, pathlib
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import config
from schema import load_log

B = 5000
RNG = np.random.default_rng(0)
MIN = getattr(config, "COMPLIANT_MIN", 5)
KEYS = ["cond", "mode", "tamper", "completeness", "method"]
CELL = ["cond", "mode", "tamper", "completeness"]
PAIRS = [("seq_gap", "judge_agent"), ("seq_gap", "combined_content"),
         ("combined_all", "combined_content"), ("combined_all", "judge_agent"),
         ("judge_agent", "judge_mean")]
HEADLINE_METHODS = ["baseline", "judge_mean", "judge_agent", "cross_topic", "combined_content",
                    "seq_gap", "combined_all"]

def load(backend):
    frames = []
    for cond in ["id", "noid"]:
        p = pathlib.Path(f"results/results_{backend}_{cond}.csv")
        if not p.exists():
            print("missing", p, "(skipped)")
            continue
        df = pd.read_csv(p)
        df["cond"] = cond
        nc = {}
        for s in sorted(df.seed.unique()):
            log, _ = load_log(f"data/runs/{backend}_{cond}_{s}.jsonl")
            nc[s] = int(sum(m.coord for m in log))
        df["n_coord"] = df.seed.map(nc)
        df["compliant"] = df.n_coord >= MIN
        frames.append(df)
    if not frames:
        raise SystemExit("no results files found")
    return pd.concat(frames, ignore_index=True)

def boot_ci(x):
    x = np.asarray(x, float)
    idx = RNG.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(axis=1)
    return x.mean(), *np.percentile(m, [2.5, 97.5])

def summarise(df):
    rows = []
    for k, g in df.groupby(KEYS):
        x = g.auc.dropna().values
        if len(x) == 0:
            continue
        mu, lo, hi = boot_ci(x)
        rows.append(dict(zip(KEYS, k), mean=mu, lo=lo, hi=hi, n_seeds=len(x)))
    return pd.DataFrame(rows)

def paired(df):
    rows = []
    for k, g in df.groupby(CELL):
        w = g.pivot_table(index="seed", columns="method", values="auc")
        for a, b in PAIRS:
            if a not in w or b not in w:
                continue
            d = (w[a] - w[b]).dropna().values
            if len(d) < 2:
                continue
            mu, lo, hi = boot_ci(d)
            p = np.nan
            if len(d) >= 6 and np.any(d != 0):
                p = wilcoxon(d).pvalue
            rows.append(dict(zip(CELL, k), a=a, b=b, mean_diff=mu, lo=lo, hi=hi,
                             n_seeds=len(d), ci_excludes_0=bool(lo > 0 or hi < 0), p_wilcoxon=p))
    return pd.DataFrame(rows)

def headline_md(summ, completenesses=(1.0, 0.4)):
    out = []
    for cond in summ.cond.unique():
        for comp in completenesses:
            s = summ[(summ.cond == cond) & (summ.completeness == comp)]
            cols = [(m, t) for m in sorted(s["mode"].unique()) for t in sorted(s.tamper.unique())]
            out.append(f"\n**{cond}, completeness {comp}** (mean AUC [95% CI over seeds]; "
                       f"n seeds = {int(s.n_seeds.max())})\n")
            out.append("| method | " + " | ".join(f"{m} q={t}" for m, t in cols) + " |")
            out.append("|---|" + "---|" * len(cols))
            for meth in HEADLINE_METHODS:
                cells = []
                for m, t in cols:
                    r = s[(s.method == meth) & (s["mode"] == m) & (s.tamper == t)]
                    cells.append(f"{r['mean'].iloc[0]:.2f} [{r.lo.iloc[0]:.2f}, {r.hi.iloc[0]:.2f}]"
                                 if len(r) else "n/a")
                out.append(f"| {meth} | " + " | ".join(cells) + " |")
    return "\n".join(out)

if __name__ == "__main__":
    backend = sys.argv[1] if len(sys.argv) > 1 else "cohere"
    pathlib.Path("results").mkdir(exist_ok=True)
    df = load(backend)

    seeds = df.drop_duplicates(["cond", "seed"])[["cond", "seed", "n_coord", "compliant"]]
    seeds.to_csv("results/seed_table.csv", index=False)
    print(f"compliance rule: n_coord >= {MIN}")
    print(seeds.to_string(index=False))
    for cond, g in seeds.groupby("cond"):   # is the split sensitive to the threshold?
        sets = {t: frozenset(g[g.n_coord >= t].seed) for t in range(1, 11)}
        same = [t for t, s in sets.items() if s == sets[MIN]]
        print(f"[{cond}] same compliant set for thresholds {min(same)}..{max(same)}")

    for name, d in [("all", df), ("compliant", df[df.compliant])]:
        if d.empty:
            continue
        summ = summarise(d)
        summ.to_csv(f"results/summary_{name}.csv", index=False)
        paired(d).to_csv(f"results/paired_{name}.csv", index=False)
        pathlib.Path(f"results/headline_{name}.md").write_text(headline_md(summ))
        print(f"wrote summary/paired/headline for {name} ({d.groupby('cond').seed.nunique().to_dict()} seeds)")
    print(open("results/headline_all.md").read())