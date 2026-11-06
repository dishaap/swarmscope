import re
import numpy as np
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ID = re.compile(r"\ba\d{2}\b")
TOK = re.compile(r"[a-z']+")
CONTENT = ["drift", "mention_out", "mention_in", "shared_ngram"]
SCARS = ["seq_gap", "dangling"]                                     # traces left behind by deleted msgs
NAMES = CONTENT + SCARS

def features(log, tasks, agents):
    f = {a: {n: 0.0 for n in NAMES} for a in agents}
    if not log:
        return f
    by = defaultdict(list)
    for m in log:
        by[m.sender].append(m)
    visible = {m.msg_id for m in log}

    # 1. task drift
    texts = [m.content for m in log] + [f"facts about {tasks[a]}" for a in agents]
    vec = TfidfVectorizer().fit(texts)
    for a in agents:
        if by[a]:
            X = vec.transform([m.content for m in by[a]])
            t = vec.transform([f"facts about {tasks[a]}"])
            f[a]["drift"] = float(1 - cosine_similarity(X, t).mean())

    # 2-3. mention graph
    out, inn = Counter(), Counter()
    for m in log:
        for t in set(ID.findall(m.content)):
            if t != m.sender and t in f:
                out[m.sender] += 1; inn[t] += 1
    for a in agents:
        f[a]["mention_out"] = out[a] / max(len(by[a]), 1)
        f[a]["mention_in"] = inn[a] / len(log) * len(agents)

    # 4. phrases shared across agents
    tri = defaultdict(set)
    for m in log:
        w = TOK.findall(m.content.lower())
        for i in range(len(w) - 2):
            tri[tuple(w[i:i + 3])].add(m.sender)
    for a in agents:
        mine = {tuple(w[i:i + 3]) for m in by[a]
                for w in [TOK.findall(m.content.lower())] for i in range(len(w) - 2)}
        f[a]["shared_ngram"] = sum(len(tri[g]) >= 2 for g in mine) / max(len(mine), 1)

    # 5-6. traces left by deletions
    for a in agents:
        if by[a]:
            span = max(m.seq for m in by[a]) + 1
            f[a]["seq_gap"] = 1 - len(by[a]) / span
        f[a]["dangling"] = sum(m.reply_to is not None and m.reply_to not in visible
                               for m in by[a]) / max(len(by[a]), 1)
    return f

def combined(feats, names):
    agents = list(feats)
    cols = []
    for n in names:
        v = np.array([feats[a][n] for a in agents], float)
        cols.append((v - v.mean()) / (v.std() + 1e-9))      # z-score within population
    return dict(zip(agents, np.mean(cols, axis=0)))

def baseline_offtask(log, tasks, agents):
    """Single-message monitor: flag a message if it ignores the sender's topic
    TODO: swap for an LLM judge via llm.complete for real tests (one call per message)."""
    by = defaultdict(list)
    for m in log:
        by[m.sender].append(tasks[m.sender] not in m.content.lower())
    return {a : float(np.mean(by[a])) if by[a] else 0.0 for a in agents}