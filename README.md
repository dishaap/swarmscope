# SwarmScope

Objective: measure and detect emergent coordination in LLM agent swarms when the monitor sees a partial or tampered log.

Submitted to the [Swarm Alignment Hackathon 2026 (AI Alignment Foundation / Apart Research)](https://apartresearch.com/sprints/swarm-alignment-2026), Tracks 2 and 4.

## Status

**Pre-hackathon work (preexisting, exploratory).** Everything in this repository up to the 5-seeds no-id results was written before the sprint started on 6 November 2026 (upto 13:30 PT/ 21:30 UTC). Per the sprint rules, this code is declared preexisting. The final report's results must be regenerated during the sprint, so the numbers here are for orientation only.

Work done during the sprint is listed in the Hackathon section below and tracked in the commit history from the sprint start.

## What is in place (pre-hackathon)

- **Simulator** (`sim.py`, `llm.py`): 12 agents x 20 rounds x 8 topics x 4 colluders per run, posting to a shared board through a Cohere model via its OpenAI-compatible endpoint.
- **Colluder instruction**: a benign, synthetic proxy for covert coordination (pooling work across tasks off-topic). Two addressing conditions: `ADDRESS_BY_ID = True` (colluders write `@id`) and `False` (colluders told never to write ids). A `CODEWORD` flag exists but has not been run.
- **Ground truth**: colluders self-report each message with a `SIDE: yes/no` line. The line is stripped before any monitor sees the log.
- **Log schema** (`schema.py`): message IDs, per-sender sequence numbers, reply references.
- **Degradation** (`degrade.py`): `targeted` and `cascade` tamper modes (colluders delete coordination messages, and in cascade also their replies to deleted messages), and random drops for partial visibility.
- **Signals** (`signals.py`): drift, mention in/out, shared n-grams, `seq_gap`, `dangling`; `combined_content` and `combined_all` (z-scored averages); a keyword baseline.
- **Experiment harness** (`experiment.py`): caches each seed's log, computes AUC per seed, writes a results CSV.
- **Plots** (`plots.py`): AUC against visible fraction, with error bars across seeds.

## Results (pre-hackathon)

- **Address-by-id condition, 5 seeds:** content signals and the combined score are above chance under targeted tampering (`combined_all` ≈ 0.94 at full visibility). Under cascade tampering the combined score keeps a gap over content-only (≈0.72 vs 0.55 at 20% visibility, tamper 0.5). `seq_gap` is the most robust signal (≈0.92 at full visibility with 90% tampering). `dangling` separates the two tamper modes, but inverts under cascade.
- **No-id condition, 5 seeds:** not interpretable yet. Self-reported coordination is very rare (0-25 messages per seed, one seed with none), and the keyword baseline sits near chance. The labels are not validated, so these results are not used in the pre-sprint report.

## Known issues (open)

1. **Parser conflates missing and negative labels.** `parse()` returns `side = False` when the `SIDE:` line is missing. Missing lines should be `None`.
2. **Raw model outputs are not saved**, so label failures cannot be diagnosed.
3. **No-id labels are unvalidated.** Colluders may not be going off-topic without ids, or may be skipping the label line.
4. **Cache filenames ignore `CODEWORD`**, so codeword runs would overwrite no-codeword runs.
5. **`drift` is close to the keyword baseline**, since both mostly track topic-word presence.
6. **The baseline is a keyword check**, not an LLM judge.
7. **`seq_gap` assumes unforgeable sequence numbers.** Not tested.
8. **Four seeds** give about ±0.2-0.3 error bars. Comparisons need paired tests and more seeds.
9. **Topics are fixed** across all runs, so results may depend on this topic list.

## Setup
### Bash
```bash
python -m venv .venv
source .venv/bin/activate       
pip install openai numpy pandas scikit-learn networkx matplotlib
```
### Powershell
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install openai numpy pandas scikit-learn networkx matplotlib
```

Set the Cohere key:
### Bash
```bash
export COHERE_API_KEY=...
```
### Powershell
```powershell
$env: COHERE_API_KEY="..."
```
> If you would like to use any other LLM, please refer to `llm.py` for the corresponding API key name and add it as an env variable

## Running

### Bash
```bash
python experiment.py --backend mock --seeds 5        # offline pipeline check
python experiment.py --backend cohere --seeds 5      # real agents; cached per seed
python plots.py results/results_cohere_id.csv        # AUC plot
python graph.py data/runs/cohere_id_1.jsonl          # message graph for one seed
```

### Powershell
```powershell
python experiment.py --backend mock --seeds 5    
python experiment.py --backend cohere --seeds 5      
python plots.py results\results_cohere_id.csv     
python graph.py data\runs\cohere_id_1.jsonl 
```

Delete `data/runs/*` after any change to the log schema or simulator, since cached logs will not match the new format.

The exact Cohere model ID is set in `llm.py` (currently `[confirm model ID]`).

> Replace `--backend cohere` arg with preferred LLM by referring to `llm.py` for correct LLM reference values.

## Files

| File | Purpose |
|---|---|
| `config.py` | Agent count, rounds, conditions, sweep values |
| `sim.py` | Simulator and colluder instructions |
| `llm.py` | Backends (mock, Cohere, Groq) |
| `schema.py` | Message dataclass and log I/O |
| `degrade.py` | Tamper modes and partial visibility |
| `signals.py` | Detection signals and baseline |
| `experiment.py` | Runs the grid, caches logs, writes results |
| `plots.py` | AUC figures |
| `graph.py` | Message graph per seed |
| `REPORT_OUTLINE.md` | Report structure and status |

## Hackathon plan (sprint work, to be done)

- Fix known issues 1-4 and save raw outputs.
- Validate no-id labels by hand and diagnose the cause.
- Run the 10-seed grid on both addressing conditions, with paired comparisons and bootstrap intervals.
- Build an LLM-judge baseline on cached logs.
- Add `seq_gap` spoofing test.
- Topic-robustness run.
- Dual-use and limitations appendix.

## Ethics and dual use

All coordination is a synthetic, benign proxy. Findings about deletion and cascade tampering could help an adversary hide coordination, so the report includes a dual-use appendix stating this explicitly.

## AI Tools Use - Declaration
Claude & Copilot were used to plan the structure of the python files, and for debugging/code-generation. Claude was also used to consolidate my work from written logs into this README file.