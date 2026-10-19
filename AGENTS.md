# Notes for AI coding assistants (Cursor and others)

This is an MSc dissertation pipeline that tests how reliable LLM judges are for Tamil (see README.md).
Follow these rules when adding or changing code:

- Keep the numbered pipeline: `scripts/NN_name.py`, each runnable on its own. Read from `data/`,
  write to `data/processed/` or `results/`.
- Never edit `data/raw/`; re-create it with `scripts/01_download_data.py`.
- Never commit secrets. Load API keys from `.env` (python-dotenv); `.env` is git-ignored.
- Never write annotator IDs or rater justification texts to any output (GDPR data minimisation).
- Reproducibility: seed 42, pinned versions in `requirements.txt`, judge temperature 0, and a JSONL
  log of every LLM call (prompt, raw output, model version, date).
- Judge every pair in both answer orders (A/B and B/A) and map the swapped verdict back.
- Keep `tie` as a label; never drop ties (32% of Tamil human verdicts are ties).
- Only 20 prompts per language: group cross-validation folds and bootstrap intervals by `prompt_id`.
- All code is GPL-3.0.
- Comment new code clearly. The author must explain every line in a viva.
