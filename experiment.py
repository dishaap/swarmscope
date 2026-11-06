import argparse, pathlib
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import sim, degrade, signals, config
from schema import save_log, load_log

# area under curve for colluders vs all agents
def auc(scores, colluders):
    agents = list(scores)
    y = [a in colluders for a in agents]
    s = np.nan_to_num([scores[a] for a in agents])
    return roc_auc_score(y, s) if 0 < sum(y) < len(y) else np.nan

def main(backend, n_seeds):
    cond = "id" if config.ADDRESS_BY_ID else "noid"
    rows = []
    for seed in range(n_seeds):
        p = pathlib.Path(f"data/runs/{backend}_{cond}_{seed}.jsonl")
        if p.exists():
            log, meta = load_log(p)                      # cached: no re-running agents
        else:
            log, meta = sim.run(f"{backend}-{cond}-{seed}", seed, backend)
            save_log(p, log, meta)
        agents, tasks, col = meta["agents"], meta["tasks"], set(meta["colluders"])
        n_c = sum(m.sender in col for m in log)
        n_side = sum(m.coord for m in log)
        print(f"seed {seed} [{cond}]: {len(log)} msgs | colluder msgs {n_c} | "
              f"self-reported SIDE msgs {n_side}")
        if n_side == 0:
            print(" Warning: no SIDE messages -> tampering will do nothing for this seed")
        for mode in config.MODES:
            for tq in config.TAMPER:    
                for comp in config.COMPLETENESS:
                    dl = degrade.degrade(log, comp, tq, mode, seed)
                    ft = signals.features(dl, tasks, agents)
                    scores = {n: {a: ft[a][n] for a in agents} for n in signals.NAMES}
                    scores["combined_content"] = signals.combined(ft, signals.CONTENT)
                    scores["combined_all"] = signals.combined(ft, signals.NAMES)
                    scores["baseline"] = signals.baseline_offtask(dl, tasks, agents)
                    for name, sc in scores.items():
                        rows.append(dict(seed=seed, mode=mode, tamper=tq, completeness=comp,
                                        method=name, auc=auc(sc, col), n_msgs=len(dl)))
    df = pd.DataFrame(rows)
    out = f"results/results_{backend}_{cond}.csv"
    df.to_csv(out, index=False)
    print("saved", out)
    pd.set_option("display.width", 200)
    print(df.groupby(["mode", "tamper", "method", "completeness"]).auc.mean()
          .unstack("completeness").round(2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mock")
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    main(a.backend, a.seeds)