"""Step 02 (objective O1): merge the PARIKSHA sources into analysis-ready tables and measure
how much the human raters agree with each other (the "human ceiling").

Reads the pinned raw data from step 01 and writes:
  data/processed/prompts.csv             20 Tamil and 20 Hindi prompts, with English versions
  data/processed/responses.csv           model responses: 280 Tamil and 400 Hindi
  data/processed/pairs_tamil.csv         all 2,002 Tamil pairs: three human votes, majority, GPT-4-32K verdict
  data/processed/pairs_hindi_sample.csv  1,000 unique Hindi pairs, 50 per prompt (seed 42)
  data/processed/direct_tamil.csv        280 Tamil direct scores (0-5) from humans and GPT-4-32K
  results/human_baseline.json            human agreement statistics and data-check results

Labels: "A" = response A is better, "B" = response B is better, "tie" = equally good or bad.
PARIKSHA codes a tie as "C"; it is renamed "tie" here. When all three raters disagree,
PARIKSHA's majority label is a tie, and this script checks that rule on every pair.

Data minimisation (GDPR): annotator IDs and free-text justifications are read only to
validate the merge. They are never written to the processed files.

Usage:  python scripts/02_merge_pariksha.py      (run scripts/01_download_data.py first)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KARYA = ROOT / "data" / "raw" / "karya"
MS = ROOT / "data" / "raw" / "microsoft"
OUT = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

LANGUAGES = ("tamil", "hindi")
HINDI_PER_PROMPT = 50  # 20 prompts x 50 = 1,000 Hindi pairs
SEED = 42
PAIR_LABELS = {"A": "A", "B": "B", "C": "tie"}
CATEGORIES = ("A", "B", "tie")
SWAP = {"A": "B", "B": "A", "tie": "tie"}  # what a verdict becomes when A and B swap places
RATERS = (1, 2, 3)
# Keys in Microsoft's outputs file that describe the prompt; every other key is a model name.
PROMPT_KEYS = {"prompt_id", "prompt_timestamp", "prompt_creator", "prompt_type",
               "language", "system_prompt", "prompt"}
# PARIKSHA's three direct-assessment dimensions and the GPT-4-32K folder for each.
DIRECT_DIMS = {
    "LA": "linguistic_acceptability",  # 0-2
    "TQ": "task_quality",              # 0-2
    "H": "hallucinations",             # 1 = the response contains a hallucination
}


# ---------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------
def check(condition: bool, message: str) -> None:
    """Stop the pipeline when a data check fails, so a bad merge can never go unnoticed."""
    if not condition:
        sys.exit(f"DATA CHECK FAILED: {message}")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_table(path: Path, sep: str = ",") -> pd.DataFrame:
    """Read a CSV/TSV with every column as text and empty cells kept as ''."""
    return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)


def write_csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT / name, index=False, encoding="utf-8", lineterminator="\n")
    print(f"  wrote data/processed/{name} ({len(df):,} rows)")


def pct(part: float, whole: float) -> float:
    return round(100 * float(part) / float(whole), 1)


def label_counts(df: pd.DataFrame, columns: list[str], categories) -> np.ndarray:
    """Items x categories matrix: how many raters chose each category for each item."""
    return np.stack([(df[columns] == c).sum(axis=1).to_numpy() for c in categories], axis=1)


def fleiss_kappa(counts: np.ndarray) -> float:
    """Fleiss' kappa for an items x categories matrix of rating counts.

    Every item must have the same number of ratings n. Kappa compares the observed agreement
    between raters with the agreement expected by chance:
        kappa = (P_observed - P_chance) / (1 - P_chance)
    """
    n = counts.sum(axis=1)
    check(bool((n == n[0]).all()), "Fleiss' kappa needs the same number of ratings for every item")
    n = int(n[0])
    p_observed = ((counts * (counts - 1)).sum(axis=1) / (n * (n - 1))).mean()
    p_category = counts.sum(axis=0) / counts.sum()
    p_chance = (p_category ** 2).sum()
    return float((p_observed - p_chance) / (1 - p_chance))


# ---------------------------------------------------------------------------------------------
# Prompts and responses (Microsoft repo)
# ---------------------------------------------------------------------------------------------
def load_outputs(lang: str, english: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Prompts, model responses and the per-prompt model order for one language."""
    prompts, responses, model_order = [], [], {}
    for item in read_json(MS / "outputs" / "round1" / "formatted" / f"output_{lang}.json"):
        models = [key for key in item if key not in PROMPT_KEYS]
        model_order[item["prompt_id"]] = models
        prompts.append({
            "language": lang, "prompt_id": item["prompt_id"], "prompt_type": item["prompt_type"],
            "prompt": item["prompt"], "prompt_english": english.get(item["prompt_id"], ""),
        })
        for model in models:
            text = item[model]["response"]
            responses.append({
                "language": lang, "prompt_id": item["prompt_id"], "model": model, "response": text,
                "response_chars": len(text), "response_words": len(text.split()),
            })
    return pd.DataFrame(prompts), pd.DataFrame(responses), model_order


# ---------------------------------------------------------------------------------------------
# Pairwise judgements
# ---------------------------------------------------------------------------------------------
def build_pairs(lang: str, karya: pd.DataFrame, responses: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Join Karya's three human votes with Microsoft's pairing and GPT-4-32K verdict."""
    k = karya[karya.language == lang]
    ms = pd.DataFrame(read_json(MS / "pairings" / "round1" / "llm_eval" / f"{lang}_pairings.json"))
    gpt4 = pd.DataFrame(read_json(
        MS / "gpt_eval" / "outputs" / "round1" / "pairwise" / f"battle_outcomes_{lang}.json"))
    check(k.pairing_id.is_unique and ms.pairing_id.is_unique and gpt4.pairing_id.is_unique,
          f"{lang}: a pairing_id appears twice")
    check(set(k.pairing_id) == set(ms.pairing_id) == set(gpt4.pairing_id),
          f"{lang}: the Karya, Microsoft and GPT-4 files do not cover the same pairs")

    df = k.merge(ms[["pairing_id", "prompt_id", "model1", "model2", "is_duplicated", "duplicate_id"]],
                 on="pairing_id", suffixes=("", "_ms"), validate="one_to_one")
    df = df.merge(gpt4[["pairing_id", "battle_outcome"]], on="pairing_id", validate="one_to_one")
    check(bool((df.prompt_id == df.prompt_id_ms).all()),
          f"{lang}: prompt_id differs between Karya and Microsoft")
    known = set(zip(responses.prompt_id, responses.model))
    check(all((p, m) in known for p, m in zip(df.prompt_id, df.model1))
          and all((p, m) in known for p, m in zip(df.prompt_id, df.model2)),
          f"{lang}: a paired response is missing from the outputs file")

    for col in ("human_1", "human_2", "human_3", "human_battle_outcome", "battle_outcome"):
        check(bool(df[col].isin(list(PAIR_LABELS)).all()), f"{lang}: unexpected label in {col}")
        df[col] = df[col].map(PAIR_LABELS)

    votes = [f"human_{r}" for r in RATERS]
    counts = pd.DataFrame(label_counts(df, votes, CATEGORIES), columns=list(CATEGORIES), index=df.index)
    top = counts.max(axis=1)
    # Majority rule: the label chosen by at least two of the three raters; a tie if all three differ.
    df["human_majority"] = counts.idxmax(axis=1).where(top >= 2, "tie")
    df["all_agree"] = top == 3
    df["all_disagree"] = top == 1
    check(bool((df.human_majority == df.human_battle_outcome).all()),
          f"{lang}: the majority rule does not reproduce PARIKSHA's labels")
    check(bool((df.all_agree == (df.all_human_agree == "TRUE")).all()), f"{lang}: all-agree flags differ")
    check(bool((df.all_disagree == (df.all_human_disagree == "TRUE")).all()),
          f"{lang}: all-disagree flags differ")

    # Cross-source check: Microsoft's per-annotator file (codes 0 = A, 1 = B, 2 = tie) must give
    # the same three votes as Karya's file for every pair that has exactly three ratings.
    tsv = read_table(MS / "karya_eval" / "round1" / "pairwise" / f"{lang}_pairwise_ratings.tsv", sep="\t")
    code = {"0": "A", "1": "B", "2": "tie"}
    check(bool(tsv.battle_outcome.isin(list(code)).all()), f"{lang}: unexpected code in per-annotator file")
    per_pair = (tsv.assign(label=tsv.battle_outcome.map(code))
                   .groupby("pairing_id")["label"].apply(lambda s: tuple(sorted(s))))
    three = per_pair[per_pair.map(len) == 3]
    karya_votes = df.set_index("pairing_id")[votes].apply(lambda r: tuple(sorted(r)), axis=1)
    check(bool((karya_votes.loc[three.index] == three).all()),
          f"{lang}: Microsoft's per-annotator votes differ from Karya's")

    # PARIKSHA shows some pairs twice, the second time with A and B swapped, to test whether the
    # raters are consistent. Both copies are kept; the copy with the larger pairing_id is flagged
    # as the repeat, so analyses can also use unique pairs only.
    df["is_duplicated"] = df.is_duplicated.astype(bool)
    dup = df[df.is_duplicated].set_index("pairing_id")
    check(bool(dup.duplicate_id.isin(dup.index).all()), f"{lang}: a repeated pair has no partner")
    partner = dup.loc[dup.duplicate_id.to_numpy()]
    check(bool((partner.model1.to_numpy() == dup.model2.to_numpy()).all()
               and (partner.model2.to_numpy() == dup.model1.to_numpy()).all()),
          f"{lang}: repeats should show the same two models with A and B swapped")
    df["is_repeat"] = df.is_duplicated & (df.pairing_id > df.duplicate_id.fillna(""))
    check(int(df.is_repeat.sum()) * 2 == int(df.is_duplicated.sum()), f"{lang}: repeats are not in pairs")

    # Consistent raters give the swapped copy the mirror-image verdict (A <-> B; a tie stays a tie).
    majority_by_id = df.set_index("pairing_id").human_majority
    repeats = df[df.is_repeat]
    consistent = repeats.human_majority.map(SWAP).to_numpy() == majority_by_id.loc[repeats.duplicate_id].to_numpy()

    unique = df[~df.is_repeat]
    stats = {
        "pairs": len(df),
        "unique_pairs": len(unique),
        "repeat_pairs": int(df.is_repeat.sum()),
        "prompts": int(df.prompt_id.nunique()),
        "models": len(set(df.model1) | set(df.model2)),
        "annotators": int(tsv.annotator_id.nunique()),  # a count only; IDs are not stored
        "pairs_cross_checked_with_per_annotator_file": len(three),
        "majority_counts": {c: int((df.human_majority == c).sum()) for c in CATEGORIES},
        "majority_pct": {c: pct((df.human_majority == c).sum(), len(df)) for c in CATEGORIES},
        "all_three_agree_pct": pct(df.all_agree.sum(), len(df)),
        "all_three_disagree_pct": pct(df.all_disagree.sum(), len(df)),
        "fleiss_kappa_all_pairs": round(fleiss_kappa(label_counts(df, votes, CATEGORIES)), 3),
        "fleiss_kappa_unique_pairs": round(fleiss_kappa(label_counts(unique, votes, CATEGORIES)), 3),
        "repeat_consistency_pct": pct(consistent.sum(), len(repeats)),
        "gpt4_32k_verdict_counts": {c: int((df.battle_outcome == c).sum()) for c in CATEGORIES},
    }
    out = (df.rename(columns={"model1": "model_a", "model2": "model_b", "battle_outcome": "gpt4_32k"})
             [["pairing_id", "prompt_id", "model_a", "model_b", *votes, "human_majority",
               "all_agree", "all_disagree", "gpt4_32k", "is_duplicated", "duplicate_id", "is_repeat"]]
             .sort_values(["prompt_id", "pairing_id"]).reset_index(drop=True))
    return out, stats


def sample_hindi(pairs: pd.DataFrame) -> pd.DataFrame:
    """1,000 unique Hindi pairs: 50 per prompt, drawn with a fixed seed so the sample is reproducible."""
    unique = pairs[~pairs.is_repeat].sort_values("pairing_id")
    check(bool((unique.groupby("prompt_id").size() >= HINDI_PER_PROMPT).all()),
          "a Hindi prompt has fewer unique pairs than the sample needs")
    sample = unique.groupby("prompt_id").sample(n=HINDI_PER_PROMPT, random_state=SEED)
    return sample.sort_values(["prompt_id", "pairing_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# Direct assessment (Tamil)
# ---------------------------------------------------------------------------------------------
def build_direct_tamil(model_order: dict) -> tuple[pd.DataFrame, dict]:
    """Tamil direct scores (0-5): three human ratings per response plus GPT-4-32K's scores.

    Karya's file has three ratings for all 280 Tamil responses but no model column. Within each
    prompt its rows follow the model order of Microsoft's outputs file, so the model is recovered
    from the row position. The recovery is then checked in two independent ways against
    Microsoft's per-annotator file, which covers 236 of the 280 responses:
      1. identical justification texts must point to the same model, and
      2. that file's ratings must be a subset of Karya's three ratings for the same response.
    """
    k = read_table(KARYA / "pariksha-individual.csv")
    k = k[k.language == "tamil"].copy()
    sizes = k.groupby("prompt_id").size()
    check(set(sizes.index) == set(model_order) and all(sizes[p] == len(model_order[p]) for p in sizes.index),
          "Tamil direct: rows per prompt do not match the models per prompt")
    k["position"] = k.groupby("prompt_id").cumcount()
    k["model"] = [model_order[p][i] for p, i in zip(k.prompt_id, k.position)]

    tsv = read_table(MS / "karya_eval" / "round1" / "individual" / "tamil_ratings_output_indi_just.tsv",
                     sep="\t")
    # Check 1: justification texts link a Karya row to a model.
    text_to_models: dict[tuple[str, str], set[str]] = {}
    for row in tsv.itertuples(index=False):
        for text in (row.ling_rating_reasoning, row.task_rating_reasoning, row.hallucination_reasoning):
            text = text.strip()
            if text and text.lower() != "none":
                text_to_models.setdefault((row.prompt_id, text), set()).add(row.model)
    linked = []
    for _, row in k.iterrows():
        votes = Counter()
        for rater in RATERS:
            for dim in DIRECT_DIMS:
                votes.update(text_to_models.get((row.prompt_id, row[f"human_{rater}_{dim}_just"].strip()), ()))
        if votes:
            best = max(votes.values())
            check(row.model in {m for m, v in votes.items() if v == best},
                  f"Tamil direct: row position and justification point to different models ({row.prompt_id})")
        linked.append(bool(votes))
    k["model_check"] = np.where(linked, "position+justification", "position")

    # Check 2: the per-annotator ratings must be contained in Karya's three ratings.
    karya_by_response = k.set_index(["prompt_id", "model"])
    tsv_columns = {"LA": "ling_rating", "TQ": "task_rating", "H": "hallucination_rating"}
    for (prompt_id, model), group in tsv.groupby(["prompt_id", "model"]):
        row = karya_by_response.loc[(prompt_id, model)]
        for dim, col in tsv_columns.items():
            karya_votes = Counter(str(row[f"human_{r}_{dim}"]).upper() for r in RATERS)
            tsv_votes = Counter(group[col].str.upper())
            check(all(karya_votes[v] >= n for v, n in tsv_votes.items()),
                  f"Tamil direct: {dim} ratings differ between sources for {model} ({prompt_id})")

    out = k[["prompt_id", "model", "model_check"]].copy()
    for dim in DIRECT_DIMS:
        for rater in RATERS:
            col = f"human_{rater}_{dim}"
            out[col] = (k[col].str.upper() == "TRUE").astype(int) if dim == "H" else k[col].astype(int)
    for col in ("human_LA", "human_TQ", "human_H", "human_score"):
        out[col] = k[col].astype(int)
    for dim in ("LA", "TQ"):
        median = out[[f"human_{r}_{dim}" for r in RATERS]].median(axis=1).astype(int)
        check(bool((median == out[f"human_{dim}"]).all()), f"Tamil direct: human_{dim} is not the median")
    majority_h = (out[[f"human_{r}_H" for r in RATERS]].sum(axis=1) >= 2).astype(int)
    check(bool((majority_h == out.human_H).all()), "Tamil direct: human_H is not the majority vote")
    # The 0-5 direct score = language (0-2) + task quality (0-2) + 1 point for no hallucination.
    check(bool((out.human_score == out.human_LA + out.human_TQ + 1 - out.human_H).all()),
          "Tamil direct: human_score is not LA + TQ + (1 - H)")

    for dim, folder in DIRECT_DIMS.items():
        scores = {}
        for item in read_json(MS / "gpt_eval" / "outputs" / "round1" / "individual" / folder / "individual_tamil.json"):
            for model, value in item.items():
                if isinstance(value, dict) and "score" in value:
                    scores[(item["prompt_id"], model)] = int(value["score"])
        keys = list(zip(out.prompt_id, out.model))
        check(all(key in scores for key in keys), f"Tamil direct: a GPT-4-32K {dim} score is missing")
        out[f"gpt4_{dim}"] = [scores[key] for key in keys]
    out["gpt4_score"] = out.gpt4_LA + out.gpt4_TQ + 1 - out.gpt4_H
    out = out.sort_values(["prompt_id", "model"]).reset_index(drop=True)

    stats = {
        "responses": len(out),
        "model_checked_by_justification": int((out.model_check != "position").sum()),
        "model_from_position_only": int((out.model_check == "position").sum()),
        "fleiss_kappa": {
            "LA": round(fleiss_kappa(label_counts(out, [f"human_{r}_LA" for r in RATERS], (0, 1, 2))), 3),
            "TQ": round(fleiss_kappa(label_counts(out, [f"human_{r}_TQ" for r in RATERS], (0, 1, 2))), 3),
            "H": round(fleiss_kappa(label_counts(out, [f"human_{r}_H" for r in RATERS], (0, 1))), 3),
        },
        "human_score_counts": {str(s): int((out.human_score == s).sum()) for s in range(6)},
    }
    return out, stats


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------
def print_summary(stats: dict) -> None:
    print("\nHuman baseline")
    for lang in LANGUAGES:
        s = stats[lang]
        m = s["majority_counts"]
        print(f"  {lang.title():<6} {s['pairs']:,} pairs ({s['unique_pairs']:,} unique + {s['repeat_pairs']} repeats),"
              f" {s['models']} models, {s['annotators']} annotators")
        print(f"         majority A {m['A']:,} | B {m['B']:,} | tie {m['tie']:,} ({s['majority_pct']['tie']}% ties)")
        print(f"         all three agree {s['all_three_agree_pct']}% | Fleiss' kappa {s['fleiss_kappa_unique_pairs']}"
              f" (unique pairs) | repeat consistency {s['repeat_consistency_pct']}%")
    d = stats["tamil_direct"]
    print(f"  Tamil direct: {d['responses']} responses ({d['model_checked_by_justification']} checked by justification,"
          f" {d['model_from_position_only']} by position only)")
    print(f"         Fleiss' kappa: language {d['fleiss_kappa']['LA']} | task quality {d['fleiss_kappa']['TQ']}"
          f" | hallucination {d['fleiss_kappa']['H']}")
    h = stats["hindi_sample"]
    print(f"  Hindi sample: {h['pairs']:,} unique pairs ({h['per_prompt']} per prompt, seed {h['seed']})")


def main() -> None:
    for folder in (KARYA, MS):
        check(folder.is_dir(), f"{folder} not found: run scripts/01_download_data.py first")
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    english_by_lang = read_json(MS / "prompts" / "prompts_round1.json")
    karya_pairs = read_table(KARYA / "pariksha-pair.csv")
    prompts, responses, pairs, stats, model_order = [], [], {}, {}, {}
    for lang in LANGUAGES:
        english = {p["prompt_id"]: p.get("english", "") for p in english_by_lang[lang]}
        lang_prompts, lang_responses, model_order[lang] = load_outputs(lang, english)
        prompts.append(lang_prompts)
        responses.append(lang_responses)
        pairs[lang], stats[lang] = build_pairs(lang, karya_pairs, lang_responses)
    hindi_sample = sample_hindi(pairs["hindi"])
    stats["hindi_sample"] = {
        "pairs": len(hindi_sample), "per_prompt": HINDI_PER_PROMPT, "seed": SEED,
        "majority_counts": {c: int((hindi_sample.human_majority == c).sum()) for c in CATEGORIES},
    }
    direct, stats["tamil_direct"] = build_direct_tamil(model_order["tamil"])

    print("All data checks passed. Writing processed tables:")
    write_csv(pd.concat(prompts, ignore_index=True), "prompts.csv")
    write_csv(pd.concat(responses, ignore_index=True), "responses.csv")
    write_csv(pairs["tamil"], "pairs_tamil.csv")
    write_csv(hindi_sample, "pairs_hindi_sample.csv")
    write_csv(direct, "direct_tamil.csv")

    sources = read_json(ROOT / "config" / "sources.json")
    baseline = {"source_commits": {name: s["commit"] for name, s in sources.items()}, **stats}
    (RESULTS / "human_baseline.json").write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    print("  wrote results/human_baseline.json")
    print_summary(stats)


if __name__ == "__main__":
    main()
