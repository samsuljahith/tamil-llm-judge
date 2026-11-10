# Reproducing the results from a fresh clone

Every number, table and figure can be re-created offline from this repository, except the
LLM-judge calls themselves (step 03), which need paid API keys. Their cleaned logs are
tracked in `results/judgements_clean/`, so step 03 is not needed to reproduce the results.

## Commands (macOS / Linux)

```bash
git clone https://github.com/samsuljahith/tamil-llm-judge.git
cd tamil-llm-judge
python3 -m venv .venv                 # Python 3.11 or newer (checked with 3.12.4)
.venv/bin/python -m pip install -r requirements.txt

.venv/bin/python scripts/01_download_data.py    # PARIKSHA sources at pinned commits, SHA-256 checked
.venv/bin/python scripts/02_merge_pariksha.py   # data/processed/ + results/human_baseline.json
# step 03 skipped: it calls paid APIs (see below)
.venv/bin/python scripts/04_agreement_bias.py   # results/agreement/
.venv/bin/python scripts/05_embed_muril.py      # data/embeddings/ (downloads MuRIL, ~0.9 GB, once)
.venv/bin/python scripts/06_litejudge.py        # results/litejudge/ (+ exploratory/)
.venv/bin/python scripts/07_ensembles_cost.py   # results/ensembles/
.venv/bin/python scripts/08_figures.py          # results/figures/
```

On Windows use `.venv\Scripts\python` instead of `.venv/bin/python`.

## Expected runtime

Measured on an Apple M2 laptop (8 GB RAM, CPU only) on 10 Nov 2026:

| Step | Time |
|---|---|
| install requirements | 42 s (depends on network) |
| 01 download data (~190 MB) | 4 s (depends on network) |
| 02 merge | 1 s |
| 04 agreement and bias | 30 s |
| 05 MuRIL embeddings (incl. ~0.9 GB download) | 108 s |
| 06 LiteJudge + exploratory check | 123 s |
| 07 ensembles and cost | 21 s |
| 08 figures | 31 s |
| **total, after install** | **about 5 min** |

Disk: about 2 GB (virtual environment ~1 GB, MuRIL in the HuggingFace cache ~0.9 GB,
raw data ~190 MB).

## API keys

Only step 03 (`scripts/03_run_llm_judges.py`) needs keys: `OPENROUTER_API_KEY` (frontier,
low_cost and base_model judges) and `SEA_LION_API_KEY` (SEA-LION judge), put in a `.env` file
copied from `.env.example`. The original run cost about US$57. Re-running it would give new
verdicts (the frontier and low-cost models do not accept temperature 0), so the tracked logs in
`results/judgements_clean/` are the record the analysis uses. No other step reads a key or
makes a network call, apart from the downloads in steps 01 and 05.

## What to expect when you compare outputs

A fresh-clone run on 10 Nov 2026 (commit `384412f`) reproduced all 85 tracked files in
`data/processed/` and `results/` byte-for-byte, including every figure, except four files
whose only differences are run timings and timestamps:

- `results/litejudge/run_info.json`: creation time, fit/predict CPU seconds, embedding seconds
- `results/litejudge/summary.md`: the CPU-time sentence on the last line
- `results/ensembles/cost_agreement.csv` and `results/ensembles/summary.md`: LiteJudge
  compute seconds (e.g. 28.9 vs 28.5 s)

All κ values, confidence intervals, contrasts, costs and figures were identical. The MuRIL
embeddings in `data/embeddings/` (git-ignored) were also byte-identical after a fresh model
download (same weights SHA-256).
