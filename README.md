# Tamil LLM-as-a-Judge Reliability

MSc Data Science Project (7FTC2001), University of Hertfordshire at PSB Academy, Singapore.
Sheik Abdullah Samsul Jahith · Lecturer: Dr Rama Abirami K

**Research question.** How reliably do LLM judges evaluate Tamil responses compared with human raters,
and can a lightweight, self-developed neural judge (LiteJudge) achieve comparable agreement at a
fraction of the cost?

## Status

25,136 judgements from four LLM judges are in `results/judgements_clean/`. On Tamil unique
pairs, current judges excluding `gpt4_32k`: Cohen's κ vs the human majority label,
swap-aggregated, is 0.44–0.51; mean pairwise Cohen's κ of each judge against the three
individual raters is 0.39–0.45 (human–human ceiling 0.41). The agreement and bias analysis
(step 04) is complete. LiteJudge (steps 05–06) is in progress.

| Step | Script | What it does | Status |
|---|---|---|---|
| 01 | `scripts/01_download_data.py` | Download the PARIKSHA sources at pinned commits; verify SHA-256 checksums | Done |
| 02 | `scripts/02_merge_pariksha.py` | Merge the sources; measure the human baseline (the "human ceiling") | Done |
| 03 | `scripts/03_run_llm_judges.py` | Run four LLM judges on every pair, in both answer orders | Done |
| 04 | `scripts/04_agreement_bias.py` | Agreement with humans (Cohen's kappa, Kendall's tau) and bias analysis | Done |
| 05 | `scripts/05_muril_embeddings.py` | Frozen MuRIL embeddings for prompts and responses | Planned |
| 06 | `scripts/06_train_litejudge.py` | Train and tune LiteJudge with prompt-grouped 5-fold cross-validation | Planned |
| 07 | `scripts/07_mitigation_cost.py` | Order-swapping, ensembles and agreement per dollar | Planned |
| 08 | `scripts/08_figures.py` | Every figure in the report | Planned |

Milestone tags: `v0.1-data`, `v0.2-judges`, `v0.3-litejudge`, `v1.0-submission`.

## Repository structure

```
config/sources.json      pinned source commits and SHA-256 checksums
scripts/                 numbered pipeline steps (run in order)
data/raw/                downloaded sources (git-ignored; re-create with step 01)
data/processed/          analysis-ready tables (columns in docs/data_dictionary.md)
results/                 statistics and judge outputs
docs/data_dictionary.md  column definitions for every processed table
docs/DATA_LICENSE.md     licences for redistributed PARIKSHA-derived data
NOTICE                   short pointer to the code vs data licences
```

## Set-up

Needs Python 3.11 or newer (tested with 3.13) and Git.

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
py -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

On Windows, use `.venv\Scripts\python` wherever the commands below say `python`, or select the
`.venv` interpreter in your editor.

## Reproduce

```bash
python scripts/01_download_data.py    # about 170 MB; verifies commits and checksums
python scripts/02_merge_pariksha.py   # writes data/processed/ and results/human_baseline.json
```

Step 02 should end with this summary:

```
Tamil  2,002 pairs (1,820 unique + 182 repeats), 14 models, 18 annotators
       majority A 663 | B 708 | tie 631 (31.5% ties)
       all three agree 45.4% | Fleiss' kappa 0.413 (unique pairs) | repeat consistency 83.0%
Hindi  4,180 pairs (3,800 unique + 380 repeats), 20 models, 13 annotators
       majority A 1,941 | B 1,970 | tie 269 (6.4% ties)
       all three agree 55.3% | Fleiss' kappa 0.443 (unique pairs) | repeat consistency 85.5%
Tamil direct: 280 responses (236 checked by justification, 44 by position only)
       Fleiss' kappa: language 0.405 | task quality 0.505 | hallucination 0.319
Hindi sample: 1,000 unique pairs (50 per prompt, seed 42)
```

## Data sources and licences

| Source | Pinned commit | Licence | Used for |
|---|---|---|---|
| Karya, `karya-inc/pariksha` | `9557f94` | DAIA Tech BY-NC-SA-FS 1.0 | Human pairwise and direct ratings |
| Microsoft Research India, `microsoft/PARIKSHA` | `03bac88` | MIT | Prompts, responses, pairings, GPT-4-32K verdicts |
| Google, `google/muril-base-cased` (from step 05) | pinned in step 05 | Apache-2.0 | Frozen text encoder for LiteJudge |

The tables in `data/processed/` and the cleaned judge logs in `results/judgements_clean/` are
derived from PARIKSHA and stay under the source licences (non-commercial use, attribution,
share-alike). All code is GPL-3.0, because the Karya licence requires GPL-3.0 for code that
feeds the data into AI systems.

**Data licence.** See [`docs/DATA_LICENSE.md`](docs/DATA_LICENSE.md) (and `NOTICE` at the repo root).

## Data handling and ethics

- Published secondary data only; no new data is collected, so UH ethics approval is not needed.
- No personal data in the processed tables. Annotator IDs and free-text justifications are read
  only to validate the merge, and are never written out.
- API keys live in a git-ignored `.env` file (copy `.env.example`). Never commit keys.

## Data decisions made in step 02

- **Ties.** PARIKSHA codes a tie as `C`; it is renamed `tie`. When all three raters disagree, the
  majority label is a tie. This is PARIKSHA's rule, and the script checks it on every pair.
- **Repeats.** 182 Tamil pairs (380 Hindi) are shown twice, the second time with A and B swapped,
  to test rater consistency. Both copies are kept, and `is_repeat` flags the second copy.
- **Hindi sample.** A reproducible sample of 1,000 unique pairs (50 per prompt, seed 42) keeps the
  API cost down while keeping every prompt equally represented.
- **Tamil direct scores.** Karya's file has three ratings for all 280 responses but no model
  column. The model is recovered from the row order, which follows Microsoft's outputs file. For
  236 responses this is confirmed by Microsoft's per-annotator file (`model_check`).

## Known limitations

- Only 20 prompts per language. Answers to the same prompt are related, so confidence intervals
  and cross-validation folds are grouped by prompt.
- PARIKSHA's cultural prompts are India-centred, so results may not transfer fully to Singapore Tamil.
- 44 Tamil direct-score responses rely on row order alone (the rule is validated on the other 236).

## Citation

This repository is an MSc Data Science dissertation (University of Hertfordshire, 2026).

If you use the PARIKSHA source data, cite:

Watts, I. et al. (2024) 'PARIKSHA: A large-scale investigation of human-LLM evaluator agreement on
multilingual and multi-cultural data', in *Proceedings of the 2024 Conference on Empirical Methods
in Natural Language Processing*. Association for Computational Linguistics.

## Licence

Code: GNU General Public License v3.0 (see `LICENSE`). Data: see [`docs/DATA_LICENSE.md`](docs/DATA_LICENSE.md).
