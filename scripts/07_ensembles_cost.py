"""Step 07 (objective O6): majority-vote ensembles of judges, and agreement per US$.

Two questions:
  1. Does a vote of several judges agree with humans better than one judge does?
  2. What does each judge (or ensemble) cost per 1,000 pairs, and how much agreement does a
     dollar buy? (This includes order-swapping: judging both orders doubles the cost.)

Ensemble rule (fixed before this script was run)
  * Voters are the four LLM judges' swap-aggregated verdicts and the primary LiteJudge's
    out-of-fold verdict (step 06). gpt4_32k is not a voter: it is PARIKSHA's verdict, its cost
    was never logged by us, and it exists in one order only.
  * Plurality vote over A / B / tie. A `tie` vote counts like any other label. If no label has
    more votes than every other label (e.g. 2-2), the ensemble says `tie`. A voter with no valid
    verdict on a pair (parse failure) is skipped for that pair.
  * Main ensembles (from the step 05-08 plan agreed on 10 Nov 2026; chosen after step 04 results
    were known, before any ensemble result was seen):
      llm4            frontier + low_cost + sea_lion + base_model
      cheap3          low_cost + sea_lion + base_model            (no frontier model)
      llm4_lite       llm4 + LiteJudge
      free_lite       sea_lion + base_model + LiteJudge           (near-zero API cost)
  * Pre-specified contrasts: kappa(each main ensemble) minus kappa(frontier), in each language,
    paired prompt-clustered bootstrap (2,000 replicates, seed 42), Holm-adjusted over all 8.
  * Every other subset of 2+ voters is EXPLORATORY and reported without tests.

Cost
  * Cost per judge = the sum of `cost_usd` in the cleaned logs over the unique pairs scored
    (OpenRouter's reported cost). SEA-LION's API is free (rate-limited), so its cost is $0.
  * LiteJudge has no API cost; its compute time is reported in seconds instead (step 05 MuRIL
    embedding, wall-clock, + step 06 fit and predict, CPU time). It is not converted to dollars
    (no price assumption is made).
  * Ensemble cost = sum of its members' costs. kappa per US$ is only defined when cost > 0.

Reads:  data/processed/, results/judgements_clean/ (via step 04), results/litejudge/oof_predictions.csv,
        results/litejudge/run_info.json, data/embeddings/MANIFEST.json, results/agreement/agreement.csv
Writes: results/ensembles/
Usage:  python scripts/07_ensembles_cost.py
No API calls.
"""
from __future__ import annotations

import gzip
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "ensembles"
LITE = ROOT / "results" / "litejudge"

# Step 04's functions, so every metric is computed exactly as for the single judges.
_spec = importlib.util.spec_from_file_location("step04", ROOT / "scripts" / "04_agreement_bias.py")
s04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s04)

SEED = 42
LANGUAGES = ("tamil", "hindi")
LABELS = s04.LABELS
LLM = s04.LLM_JUDGES                         # ("frontier", "low_cost", "sea_lion", "base_model")
VOTERS = {**{j: f"{j}_swap" for j in LLM}, "litejudge": "litejudge"}  # voter -> verdict column
MAIN_ENSEMBLES = {
    "llm4": ("frontier", "low_cost", "sea_lion", "base_model"),
    "cheap3": ("low_cost", "sea_lion", "base_model"),
    "llm4_lite": ("frontier", "low_cost", "sea_lion", "base_model", "litejudge"),
    "free_lite": ("sea_lion", "base_model", "litejudge"),
}
REFERENCE = "frontier"  # the contrasts ask: can an ensemble replace the most expensive judge?


# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------
def load_frame(language: str) -> pd.DataFrame:
    """Unique pairs with every LLM verdict (step 04 code) plus LiteJudge's out-of-fold verdict."""
    df = s04.attach_all_judges(s04.load_pairs(language), language)
    oof = pd.read_csv(LITE / "oof_predictions.csv")
    oof = oof.loc[oof.language == language, ["pairing_id", "litejudge"]]
    df = df.merge(oof, on="pairing_id", how="left")
    if df.litejudge.isna().any():
        raise SystemExit(f"{language}: LiteJudge verdict missing for some pairs; run step 06 first.")
    return df


def vote(df: pd.DataFrame, members: tuple[str, ...]) -> np.ndarray:
    """Plurality vote of the members' verdicts; no single winner -> tie; invalid votes skipped."""
    counts = np.zeros((len(df), 3), dtype=int)  # votes for A, B, tie on every pair
    for member in members:
        verdicts = df[VOTERS[member]].to_numpy()
        for k, label in enumerate(LABELS):
            counts[:, k] += verdicts == label     # None / NaN never equals a label, so it adds nothing
    top = counts.max(axis=1)
    n_top = (counts == top[:, None]).sum(axis=1)  # how many labels share the highest count
    result = np.array(LABELS, dtype=object)[counts.argmax(axis=1)]
    result[n_top > 1] = "tie"                     # no clear winner
    result[top == 0] = None                       # nobody voted (does not happen here; kept safe)
    return result


# ---------------------------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------------------------
def judge_costs(language: str, pairing_ids: set[str]) -> list[dict]:
    """Logged API cost of each LLM judge over the unique pairs, for A/B only and both orders."""
    rows = []
    for judge in LLM:
        path = s04.CLEAN / f"{s04.TASK[language]}__{judge}.jsonl.gz"
        cost = {"AB": 0.0, "BA": 0.0}
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                rec = json.loads(line)
                if rec["item_id"] in pairing_ids:
                    cost[rec["order"]] += float(rec.get("cost_usd") or 0.0)
        n = len(pairing_ids)
        rows.append({"language": language, "judge": judge, "n_pairs": n,
                     "cost_usd_ab_only": round(cost["AB"], 4),
                     "cost_usd_both_orders": round(cost["AB"] + cost["BA"], 4),
                     "usd_per_1000_pairs_ab_only": round(1000 * cost["AB"] / n, 4),
                     "usd_per_1000_pairs_both_orders": round(1000 * (cost["AB"] + cost["BA"]) / n, 4)})
    return rows


def litejudge_compute_seconds(language: str) -> float:
    """Seconds LiteJudge needs for one language: MuRIL embedding (step 05, wall-clock) + fit + predict (step 06, CPU)."""
    emb = json.loads((ROOT / "data" / "embeddings" / "MANIFEST.json").read_text())["languages"][language]
    run = json.loads((LITE / "run_info.json").read_text())["per_language"][language]
    return float(emb["seconds"] + run["fit_seconds"] + run["predict_seconds"])


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)              # bootstrap CIs
    rng_contrast = np.random.default_rng(SEED + 1)  # paired contrasts

    agreement, ceiling, dist, boot, length, tau_rows, subsets, contrasts, costs = ([] for _ in range(9))
    for language in LANGUAGES:
        df = load_frame(language)

        # Main ensembles: every step 04 metric.
        for name, members in MAIN_ENSEMBLES.items():
            df[name] = vote(df, members)
            agreement.append(s04.agreement_row(df, language, name, "ensemble", name))
            ceiling.append(s04.human_ceiling_row(df, language, name, name))
            dist.extend(s04.verdict_rows(df, language, name, name, view="ensemble"))
            length.append(s04.length_row(df, language, name, name))
            for metric, arr in s04.bootstrap_metrics(df, name, name, rng).items():
                if np.all(np.isnan(arr)):
                    continue  # no flip rate for an ensemble
                lo, hi = s04.ci95(arr)
                boot.append({"language": language, "judge": name, "metric": metric,
                             "estimate": round(float(np.nanmean(arr)), 4),
                             "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
            diffs = s04.bootstrap_delta_same_lang(df, name, f"{REFERENCE}_swap", rng_contrast)
            lo, hi = s04.ci95(diffs)
            contrasts.append({"contrast": f"kappa {language}: {name} − {REFERENCE}",
                              "estimate": round(float(np.nanmean(diffs)), 4), "ci_low": round(lo, 4),
                              "ci_high": round(hi, 4), "p_boot": round(s04.two_sided_p(diffs), 4)})
        _, tau = s04.leaderboard_block(df, language, {n: n for n in MAIN_ENSEMBLES})
        tau_rows.append(tau)

        # EXPLORATORY: every subset of two or more voters (no tests).
        for size in range(2, len(VOTERS) + 1):
            for members in itertools.combinations(VOTERS, size):
                verdict = vote(df, members)
                tmp = df.assign(_v=verdict)
                row = s04.agreement_row(tmp, language, "+".join(members), "ensemble", "_v")
                ceil = s04.human_ceiling_row(tmp, language, "+".join(members), "_v")
                ok = pd.Series(verdict).isin(LABELS)
                row.update({"size": size, "gap": ceil["gap"],
                            "pct_tie": round(100 * float(np.mean(verdict[ok.to_numpy()] == "tie")), 2),
                            "main": members in MAIN_ENSEMBLES.values()})
                subsets.append(row)

        costs.extend(judge_costs(language, set(df.pairing_id)))

    # Holm over the 8 pre-specified contrasts.
    contrast_df = pd.DataFrame(contrasts)
    contrast_df["p_holm"] = np.round(s04.holm(list(contrast_df.p_boot)), 4)

    # ---- Cost vs agreement table: single judges (both views), LiteJudge, ensembles ----------
    cost_df = pd.DataFrame(costs)
    a4 = pd.read_csv(ROOT / "results" / "agreement" / "agreement.csv")
    lite = pd.read_csv(LITE / "agreement.csv")
    ens = pd.DataFrame(agreement)
    per_judge = cost_df.set_index(["language", "judge"])
    rows = []
    for language in LANGUAGES:
        for judge in LLM:
            c = per_judge.loc[(language, judge)]
            for view, usd in (("ab", c.usd_per_1000_pairs_ab_only), ("swap_agg", c.usd_per_1000_pairs_both_orders)):
                k = a4[(a4.language == language) & (a4.judge == judge) & (a4.view == view)].kappa.iloc[0]
                rows.append({"language": language, "judge": judge, "kind": "single", "view": view,
                             "kappa": k, "usd_per_1000_pairs": usd, "compute_seconds": None})
        k = lite[(lite.language == language) & (lite.judge == "litejudge")].kappa.iloc[0]
        rows.append({"language": language, "judge": "litejudge", "kind": "single", "view": "oof",
                     "kappa": k, "usd_per_1000_pairs": 0.0,
                     "compute_seconds": round(litejudge_compute_seconds(language), 2)})
        for name, members in MAIN_ENSEMBLES.items():
            usd = sum(per_judge.loc[(language, m)].usd_per_1000_pairs_both_orders for m in members if m in LLM)
            k = ens[(ens.language == language) & (ens.judge == name)].kappa.iloc[0]
            secs = litejudge_compute_seconds(language) if "litejudge" in members else None
            rows.append({"language": language, "judge": name, "kind": "ensemble", "view": "ensemble",
                         "kappa": k, "usd_per_1000_pairs": round(usd, 4),
                         "compute_seconds": None if secs is None else round(secs, 2)})
    cost_agreement = pd.DataFrame(rows)
    cost_agreement["kappa_per_usd_per_1000_pairs"] = np.where(
        cost_agreement.usd_per_1000_pairs > 0,
        (cost_agreement.kappa / cost_agreement.usd_per_1000_pairs).round(4), np.nan)

    # Order-swapping as a mitigation: how much kappa the second order buys, and at what price.
    swap_rows = []
    for (language, judge), g in cost_agreement[cost_agreement.judge.isin(LLM)].groupby(["language", "judge"], sort=False):
        ab, sw = g[g.view == "ab"].iloc[0], g[g.view == "swap_agg"].iloc[0]
        extra = sw.usd_per_1000_pairs - ab.usd_per_1000_pairs
        swap_rows.append({"language": language, "judge": judge, "kappa_ab": ab.kappa, "kappa_swap_agg": sw.kappa,
                          "kappa_gain": round(sw.kappa - ab.kappa, 4), "extra_usd_per_1000_pairs": round(extra, 4)})
    swap_df = pd.DataFrame(swap_rows)

    # ---- Write --------------------------------------------------------------------------------
    pd.DataFrame(agreement).to_csv(OUT / "ensemble_agreement.csv", index=False)
    pd.DataFrame(ceiling).to_csv(OUT / "ensemble_human_ceiling.csv", index=False)
    pd.DataFrame(dist).to_csv(OUT / "ensemble_verdict_distribution.csv", index=False)
    pd.DataFrame(length).to_csv(OUT / "ensemble_length_bias.csv", index=False)
    pd.DataFrame(boot).to_csv(OUT / "ensemble_bootstrap_ci.csv", index=False)
    pd.concat(tau_rows, ignore_index=True).to_csv(OUT / "ensemble_leaderboard_tau.csv", index=False)
    contrast_df.to_csv(OUT / "ensemble_contrasts.csv", index=False)
    pd.DataFrame(subsets).to_csv(OUT / "ensemble_subsets_exploratory.csv", index=False)
    cost_df.to_csv(OUT / "cost_per_judge.csv", index=False)
    cost_agreement.to_csv(OUT / "cost_agreement.csv", index=False)
    swap_df.to_csv(OUT / "order_swap_cost.csv", index=False)
    write_summary(pd.DataFrame(agreement), pd.DataFrame(ceiling), pd.DataFrame(dist), pd.DataFrame(boot),
                  contrast_df, pd.DataFrame(subsets), cost_agreement, swap_df)
    print(f"Wrote {len(list(OUT.glob('*')))} files under {OUT.relative_to(ROOT).as_posix()}/")


def write_summary(agreement, ceiling, dist, boot, contrast_df, subsets, cost_agreement, swap_df) -> None:
    f = lambda v, nd=3: "" if pd.isna(v) else f"{v:.{nd}f}"  # noqa: E731 (short number formatter)
    lines = ["# Step 07 summary: ensembles and cost", "",
             "Plurality vote (no clear winner -> tie) of swap-aggregated LLM verdicts and LiteJudge's "
             "out-of-fold verdict; step 04 metrics on the same unique pairs.", "",
             "## Main ensembles", "",
             "| language | ensemble | κ vs majority | 95% CI | gap to ceiling | % tie |", "|---|---|---|---|---|---|"]
    kci = boot[boot.metric == "kappa"].set_index(["language", "judge"])
    ties = dist[dist.label == "tie"].set_index(["language", "judge"])["pct"]
    gaps = ceiling.set_index(["language", "judge"])["gap"]
    for r in agreement.itertuples():
        key = (r.language, r.judge)
        lines.append(f"| {r.language} | {r.judge} | {f(r.kappa)} | [{f(kci.loc[key].ci_low)}, {f(kci.loc[key].ci_high)}] | "
                     f"{f(gaps.loc[key])} | {f(ties.loc[key], 1)} |")
    lines += ["", "## Pre-specified contrasts vs frontier (Holm over 8)", "",
              "| contrast | estimate | 95% CI | p | p_Holm |", "|---|---|---|---|---|"]
    for r in contrast_df.itertuples():
        lines.append(f"| {r.contrast} | {f(r.estimate)} | [{f(r.ci_low)}, {f(r.ci_high)}] | {r.p_boot} | {r.p_holm} |")
    lines += ["", "## Cost vs agreement (US$ per 1,000 unique pairs)", "",
              "| language | judge | view | κ | US$ / 1,000 pairs | κ per US$ | compute s |", "|---|---|---|---|---|---|---|"]
    for r in cost_agreement.itertuples():
        lines.append(f"| {r.language} | {r.judge} | {r.view} | {f(r.kappa)} | {f(r.usd_per_1000_pairs, 2)} | "
                     f"{f(r.kappa_per_usd_per_1000_pairs, 4)} | {f(r.compute_seconds, 1)} |")
    lines += ["", "## Order-swapping: κ gain for the second order", "",
              "| language | judge | κ A/B | κ swap-agg | gain | extra US$ / 1,000 pairs |", "|---|---|---|---|---|---|"]
    for r in swap_df.itertuples():
        lines.append(f"| {r.language} | {r.judge} | {f(r.kappa_ab)} | {f(r.kappa_swap_agg)} | {f(r.kappa_gain)} | "
                     f"{f(r.extra_usd_per_1000_pairs, 2)} |")
    lines += ["", "## EXPLORATORY: best five voter subsets by κ (no tests)", ""]
    for language in LANGUAGES:
        top = subsets[subsets.language == language].sort_values("kappa", ascending=False).head(5)
        lines.append(f"- {language}: " + "; ".join(f"{r.judge} κ {f(r.kappa)} (tie {f(r.pct_tie, 1)}%)"
                                                  for r in top.itertuples()))
    lines += ["", "SEA-LION is a free, rate-limited API ($0). LiteJudge has no API cost; its compute time "
              "(embedding wall-clock + fit/predict CPU) is shown instead of a dollar figure.", ""]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
