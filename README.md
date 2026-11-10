# SwarmScope: detecting swarm coordination from partial and tampered logs

*Deletion leaves scars:* SwarmScope simulates a 12-agent LLM swarm in which four agents secretly coordinate, lets those agents delete their own incriminating messages, hides a further random share of the log from the monitor, and measures how well different monitors still pick out the colluders.

Submission to the **Swarm Alignment Hackathon 2026** (AI Alignment Foundation / Apart Research): Tracks 2 (emergent swarm behaviour) and 4 (detecting coordination and oversight from partial or tampered logs).

**Paper:** TBD after hackathon results are released.

## Main finding

Content-based monitors (keyword baseline, text signals, LLM judges) lose their signal once colluders can delete evidence. A structural signal, gaps in each sender's platform-stamped sequence numbers, survives. Mean AUC over 10 seeds, full log visibility, targeted tampering at q = 0.9:

| monitor | id condition | noid condition |
|---|---|---|
| `seq_gap` (deletion scar) | 0.86 | 0.99 |
| `combined_all` | 0.77 | 0.94 |
| `combined_content` | 0.69 | 0.79 |
| `judge_agent` (LLM) | 0.63 | 0.77 |
| keyword baseline | 0.54 | 0.57 |

Under cascade tampering at q = 0.9, `combined_content` falls to chance in both conditions (0.50 and 0.51), while `seq_gap` stays at 0.85 (id) and 0.79 (noid). Confidence intervals, paired differences, the compliant-seed analysis and the cases where `seq_gap` fails (low visibility, heavy cascade deletion) are in the paper and in `results/`.

## Method in brief

- **Swarm:** 12 agents x 20 rounds x 1 message board where each agent has a benign task: short factual notes on one of eight topics. Agents run on Cohere Command A (`command-a-03-2025`). The platform stamps every message with an id and a per-sender sequence number.
- **Collusion proxy.** Four random agents are privately told to pool work across their topics and keep it secret. This is a deliberately benign stand-in for covert coordination; no harmful content is generated. Two conditions: `id` (colluders address partners as `@id`) and `noid` (they may not write ids and use reply tags instead). A code-word condition exists in the code but was **not evaluated**.
- **Ground truth.** Colluders append `SIDE: yes/no` to each message, saying whether it served the private objective. This is self-reported, stripped from the log, and never shown to a monitor.
- **Degradation.** (1) Tampering: coordination messages are deleted with probability q in {0, 0.5, 0.9}; `targeted` deletes only those, `cascade` also deletes the colluders' own replies to them. (2) Partial visibility: each remaining message is kept with probability in {1.0, 0.7, 0.4, 0.2}. 24 cells per seed.
- **Monitors.** Content signals (`drift`, `mention_out`, `mention_in`, `shared_ngram`, `cross_topic`), scar signals (`seq_gap`, `dangling`), two z-score combinations, a keyword baseline, and two LLM judges (message-level and agent-level, on `cohere-r-plus-08-2024`).
- **Metric.** Per-seed AUC for separating the 4 colluders from the 8 honest agents; mean over seeds with seed-bootstrap 95% CIs. Under chance the 10-seed mean is within about 0.5 ± 0.12.

## Repository layout

| file | purpose |
|---|---|
| `config.py` | all settings: model, agents, rounds, tamper and visibility grids, `ADDRESS_BY_ID`, `CODEWORD`, `COMPLIANT_MIN`, `JUDGE_BACKEND` |
| `schema.py` | message schema; log save/load |
| `llm.py` | provider wrapper (Cohere and Groq via OpenAI-compatible endpoints, plus an offline `mock` backend) |
| `sim.py` | the swarm simulation and prompts |
| `degrade.py` | tampering and partial-visibility models |
| `signals.py` | monitor signals and combinations |
| `judge.py` | message-level and agent-level LLM judges, with caching |
| `experiment.py` | runs seeds and the degradation grid, writes `results/results_{backend}_{cond}.csv` |
| `summarise.py` | bootstrap CIs, paired differences, compliant-seed analysis |
| `plots.py`, `figs.py` | AUC curves, null calibration, dose-response, message graphs |
| `diagnose.py` | inspects a single run's labels |
| `results/` | final CSVs, summaries and figures |

Generated per-run logs go to `data/runs/`, judge scores to `data/judge/`.

## Reproducing

Python 3.10+ with `numpy pandas scikit-learn scipy matplotlib networkx openai`. Windows PowerShell shown; use `export` on macOS/Linux.

```powershell
pip install numpy pandas scikit-learn scipy matplotlib networkx openai
$env:COHERE_API_KEY = "..."

# offline smoke test, no API calls
python experiment.py --backend mock --seeds 2 --judge

# real runs: set ADDRESS_BY_ID in config.py (True = id, False = noid), then
python experiment.py --backend cohere --seeds 10 --judge

# analysis and figures
python summarise.py cohere
python plots.py results\results_cohere_id.csv
python plots.py results\results_cohere_noid.csv
python figs.py cohere 1
```

Logs are cached per seed and condition. The run refuses to reuse a cached log made with different `ADDRESS_BY_ID`/`CODEWORD` settings. Delete `data/runs/*` to regenerate. Judge scores are cached under the judge tag in the file name.

## Pre-hackathon work (declared)

Per the sprint rules, the following was written **before the sprint began on 6 November 2026**: the simulator and colluder instruction (both addressing conditions, `SIDE` self-reporting), the log schema, targeted and cascade tampering with random drops, the signals `drift`, `mention_in`, `mention_out`, `shared_ngram`, `seq_gap` and `dangling`, the z-scored combinations, the keyword baseline, the experiment harness and the AUC plots. Exploratory runs with these components (4 seeds in id, 4–5 in noid) are not used in the paper.

Developed **during the sprint**: handling of missing `SIDE` lines as unlabelled, raw outputs stored outside the message schema, condition-aware cache names with a mismatch guard, stem-aware topic matching, the `cross_topic` signal, the coordination quota instruction, both LLM judges, the 10-seed runs in both conditions, the statistical analysis and the figures. All results reported come from logs regenerated during the sprint.

## Limitations

- Collusion is a benign synthetic proxy, and labels are the colluders' self-report.
- In `noid`, the quota makes coordination near-saturated (61–79 of 80 colluder messages), and `cross_topic` was designed after inspecting a preliminary noid log, so treat it as a positive control rather than an independent result.
- The tamper model is an oracle that only deletes. It does not forge, renumber or rewrite messages.
- `seq_gap` assumes sequence numbers that agents cannot forge. Spoofing was not tested.
- One agent model family, one judge model, 12 agents, 20 rounds, fixed topics.
- Two of ten id seeds had almost no coordination (0 and 2 messages); the paper reports all seeds as primary and a compliant-seed analysis as secondary.
- `seq_gap` falls below chance in one regime (noid, cascade, q = 0.9, 40% visibility). A suggested explanation, that it only counts gaps below the highest visible sequence number, was not tested.

## Dual-use note

The cascade strategy shows an adversary how to hide dangling references, and the `seq_gap` weakness shows how a defender's signal can be blunted. The task is a benign proxy, and the practical takeaway for defenders is to use tamper-evident logs: append-only, platform-assigned counters or hash chains. See the paper's dual-use appendix.

## Future work

Forging and renumbering attacks, the code-word condition, prompt-infection scenarios, real incident traces, a transcript-level judge, other model families.

## License and citation

This repository is distributed under the MIT License. 