# Data licence

The **code** in this repository is licensed under the GNU General Public License v3.0
(see `LICENSE` at the repo root).

The **data** in `data/processed/` and `results/judgements_clean/` is derived from PARIKSHA.
It is **not** GPL-3.0. It stays under the source licences below (non-commercial use,
attribution, share-alike). Do not treat GitHub's "GPL-3.0" badge as applying to the tables
or the cleaned judge logs.

## Karya human ratings — DAIA Tech BY-NC-SA-FS 1.0

Source: [`karya-inc/pariksha`](https://github.com/karya-inc/pariksha) at pinned commit
[`9557f94`](https://github.com/karya-inc/pariksha/tree/9557f949cc487e88dc6add4d8e49f648c9639bbe).

Licence file at that commit:
https://github.com/karya-inc/pariksha/blob/9557f949cc487e88dc6add4d8e49f648c9639bbe/LICENSE

Terms in short:

- **BY** — attribution to Karya / the PARIKSHA authors
- **NC** — non-commercial use only
- **SA** — share adaptations under the same terms
- **FS** — software that incorporates the data into an AI system must be GPL-3.0 (this repo's code is)

Used here for the three human pairwise votes and the Tamil direct (0–5) scores.

## Microsoft PARIKSHA — MIT

Source: [`microsoft/PARIKSHA`](https://github.com/microsoft/PARIKSHA) at pinned commit
[`03bac88`](https://github.com/microsoft/PARIKSHA/tree/03bac88c3d899b008efd9671b11109b11676b7c4).

Licence file at that commit:
https://github.com/microsoft/PARIKSHA/blob/03bac88c3d899b008efd9671b11109b11676b7c4/LICENSE

Used here for prompts, model responses, pairings, and PARIKSHA's original GPT-4-32K verdicts.

## What this repo redistributes

| Path | Contents |
|---|---|
| `data/processed/prompts.csv` | PARIKSHA prompts (Tamil and Hindi, plus English glosses) |
| `data/processed/responses.csv` | PARIKSHA model answers |
| `data/processed/pairs_*.csv`, `direct_tamil.csv` | Pairings, human majority labels, GPT-4-32K verdicts (no annotator IDs, no rater justifications) |
| `results/judgements_clean/*.jsonl.gz` | LLM-judge calls; `request_messages` repeats the prompt and answers |

Raw downloads (`data/raw/`) and raw per-call logs (`results/judgements/`) are git-ignored
and are re-created with `scripts/01_download_data.py` and `scripts/03_run_llm_judges.py`.

Pinned commits and SHA-256 checksums are in `config/sources.json`.
