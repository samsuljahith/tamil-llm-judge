# Data dictionary

All tables are written by `scripts/02_merge_pariksha.py` as UTF-8 CSV files in `data/processed/`.
Pairwise labels: `A` = response A is better, `B` = response B is better, `tie` = equally good or bad.

## prompts.csv (40 rows: 20 Tamil, 20 Hindi)

| Column | Meaning |
|---|---|
| `language` | `tamil` or `hindi` |
| `prompt_id` | PARIKSHA prompt ID |
| `prompt_type` | Topic category from PARIKSHA (for example health, finance, culture) |
| `prompt` | The prompt in the target language |
| `prompt_english` | PARIKSHA's English version of the prompt |

## responses.csv (680 rows: 280 Tamil, 400 Hindi)

| Column | Meaning |
|---|---|
| `language`, `prompt_id`, `model` | Which model answered which prompt |
| `response` | The model's answer |
| `response_chars`, `response_words` | Length features, used later for the length-bias analysis |

## pairs_tamil.csv (2,002 rows) and pairs_hindi_sample.csv (1,000 rows)

| Column | Meaning |
|---|---|
| `pairing_id` | PARIKSHA pair ID (the merge key) |
| `prompt_id` | The prompt both responses answer |
| `model_a`, `model_b` | Models whose responses were shown as A and B (join to `responses.csv`) |
| `human_1`, `human_2`, `human_3` | The three human votes, in rater order (no rater IDs) |
| `human_majority` | Label chosen by at least two raters; `tie` if all three differ |
| `all_agree`, `all_disagree` | All three votes equal / all three different |
| `gpt4_32k` | PARIKSHA's original GPT-4-32K verdict (a free baseline judge) |
| `is_duplicated` | The pair is shown twice, with A and B swapped |
| `duplicate_id` | `pairing_id` of the swapped copy |
| `is_repeat` | `True` for the second copy; exclude these rows for unique-pair analyses |

The Hindi sample contains unique pairs only (`is_repeat` is always `False`).

## direct_tamil.csv (280 rows)

| Column | Meaning |
|---|---|
| `prompt_id`, `model` | The response being scored (join to `responses.csv`) |
| `model_check` | `position+justification`: model confirmed by Microsoft's per-annotator file; `position`: row order only |
| `human_k_LA` (k = 1–3) | Linguistic acceptability, 0–2, from rater k |
| `human_k_TQ` | Task quality, 0–2 |
| `human_k_H` | Hallucination: 1 = the response contains a hallucination, 0 = it does not |
| `human_LA`, `human_TQ` | Median of the three ratings |
| `human_H` | Majority vote of the three ratings |
| `human_score` | Direct score 0–5 = `human_LA + human_TQ + (1 - human_H)` |
| `gpt4_LA`, `gpt4_TQ`, `gpt4_H`, `gpt4_score` | The same scores from PARIKSHA's GPT-4-32K judge |

## results/human_baseline.json

Per language: pair counts, number of models and annotators (counts only), majority-label counts,
the share of pairs where all three raters agree or all disagree, Fleiss' kappa (all pairs and unique
pairs), and repeat consistency (the share of swapped repeats whose majority verdict mirrors the
original). For Tamil direct scores: coverage of the model check, Fleiss' kappa per dimension and
the distribution of 0–5 scores.
