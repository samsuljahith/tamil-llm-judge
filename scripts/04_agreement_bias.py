"""Step 04 (objectives O3 and O4): agreement with humans, and bias of the LLM judges.

Reads the unique-pair tables from step 02 and the cleaned judge logs from step 03b, and writes
tables plus a short summary under results/agreement/. No API calls.

Main analysis rules (AGENTS.md and the step-04 brief):
  * unique pairs only (`is_repeat == False`); ties are kept as a label, never dropped;
  * `parse_ok == false` is treated as missing (that pair/order is skipped for that judge);
  * PARIKSHA's original GPT-4-32K verdict (`gpt4_32k`) is a fifth judge, A/B order only;
  * every pairwise LLM judge is scored three ways: A/B only, B/A mapped back, and swap-aggregated
    (the two orders agree -> that verdict; they disagree -> tie);
  * bootstrap intervals and contrasts are clustered by `prompt_id` (20 prompts per language),
    2,000 replicates, seed 42; Holm-corrected p-values on the pre-specified contrasts.

Usage:  python scripts/04_agreement_bias.py
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
CLEAN = ROOT / "results" / "judgements_clean"
CONFIG = ROOT / "config" / "judges.json"
OUT = ROOT / "results" / "agreement"

SEED = 42
N_BOOT = 2000
LABELS = ("A", "B", "tie")
LLM_JUDGES = ("frontier", "low_cost", "sea_lion", "base_model")
# Swap-aggregated is the main pairwise number; gpt4_32k only exists in A/B order.
MAIN_VIEW = "swap_agg"
PAIR_FILES = {
    "tamil": "pairs_tamil.csv",
    "hindi": "pairs_hindi_sample.csv",
}
TASK = {"tamil": "tamil_pairwise", "hindi": "hindi_pairwise"}


# ---------------------------------------------------------------------------------------------
# Small statistical helpers
# ---------------------------------------------------------------------------------------------
def cohen_kappa(y1: pd.Series | np.ndarray, y2: pd.Series | np.ndarray) -> float:
    """Cohen's kappa on the three labels A / B / tie. Both inputs must already be aligned and non-null."""
    a = np.asarray(y1)
    b = np.asarray(y2)
    return float(cohen_kappa_score(a, b, labels=list(LABELS)))


def accuracy(y1: pd.Series, y2: pd.Series) -> float:
    return float((y1.to_numpy() == y2.to_numpy()).mean())


def _encode3(x: np.ndarray) -> np.ndarray:
    """Map A/B/tie to 0/1/2 for a fast confusion matrix."""
    out = np.empty(x.size, dtype=np.int8)
    out[x == "A"] = 0
    out[x == "B"] = 1
    out[x == "tie"] = 2
    return out


def kappa_np(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa for the bootstrap inner loop (no pandas, three labels)."""
    n = a.size
    if n == 0:
        return float("nan")
    mat = np.zeros((3, 3), dtype=float)
    np.add.at(mat, (_encode3(a), _encode3(b)), 1)
    po = mat.trace() / n
    pe = (mat.sum(axis=1) * mat.sum(axis=0)).sum() / (n * n)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return float((po - pe) / (1.0 - pe))


def holm(pvalues: list[float]) -> np.ndarray:
    """Holm-adjusted p-values, same order as the input list."""
    adj = np.full(len(pvalues), np.nan)
    ok = [i for i, p in enumerate(pvalues) if pd.notna(p)]
    if ok:
        _, adj_ok, _, _ = multipletests([pvalues[i] for i in ok], method="holm")
        for i, q in zip(ok, adj_ok):
            adj[i] = q
    return adj


def two_sided_p(diffs: np.ndarray) -> float:
    """Two-sided bootstrap p-value for H0: mean(diff) = 0, from the share of diffs on each side of 0."""
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return float("nan")
    p = 2.0 * min((diffs >= 0).mean(), (diffs <= 0).mean())
    return float(min(1.0, p))


def ci95(samples: np.ndarray) -> tuple[float, float]:
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return (float("nan"), float("nan"))
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------
def load_pairs(language: str) -> pd.DataFrame:
    """Unique pairs only, with response lengths joined on for the length-bias analysis."""
    pairs = pd.read_csv(PROCESSED / PAIR_FILES[language])
    pairs = pairs.loc[~pairs.is_repeat].copy()
    responses = pd.read_csv(PROCESSED / "responses.csv")
    responses = responses[responses.language == language][["prompt_id", "model", "response_chars"]]
    pairs = pairs.merge(responses.rename(columns={"model": "model_a", "response_chars": "chars_a"}),
                        on=["prompt_id", "model_a"], how="left")
    pairs = pairs.merge(responses.rename(columns={"model": "model_b", "response_chars": "chars_b"}),
                        on=["prompt_id", "model_b"], how="left")
    return pairs.reset_index(drop=True)


def load_pairwise_log(language: str, judge: str) -> pd.DataFrame:
    """One row per (pair, order). parse_ok=false becomes a missing verdict."""
    path = CLEAN / f"{TASK[language]}__{judge}.jsonl.gz"
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            ok = rec.get("parse_ok") is True
            rows.append({
                "pairing_id": rec["item_id"],
                "order": rec["order"],
                "canonical": rec.get("verdict_canonical") if ok else None,
                "presented": rec.get("verdict_presented") if ok else None,
            })
    return pd.DataFrame(rows)


def attach_judge(pairs: pd.DataFrame, log: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Add <prefix>_ab, <prefix>_ba, <prefix>_swap columns (canonical orientation) to the pair table."""
    ab = log[log.order == "AB"][["pairing_id", "canonical", "presented"]].rename(
        columns={"canonical": f"{prefix}_ab", "presented": f"{prefix}_ab_presented"})
    ba = log[log.order == "BA"][["pairing_id", "canonical", "presented"]].rename(
        columns={"canonical": f"{prefix}_ba", "presented": f"{prefix}_ba_presented"})
    out = pairs.merge(ab, on="pairing_id", how="left").merge(ba, on="pairing_id", how="left")
    ab_v, ba_v = out[f"{prefix}_ab"], out[f"{prefix}_ba"]
    both = ab_v.isin(LABELS) & ba_v.isin(LABELS)
    swap = pd.Series(np.nan, index=out.index, dtype="object")
    swap.loc[both & (ab_v == ba_v)] = ab_v.loc[both & (ab_v == ba_v)].to_numpy()
    swap.loc[both & (ab_v != ba_v)] = "tie"
    out[f"{prefix}_swap"] = swap
    return out


def attach_all_judges(pairs: pd.DataFrame, language: str) -> pd.DataFrame:
    out = pairs.copy()
    out["gpt4_32k_ab"] = out["gpt4_32k"]
    for judge in LLM_JUDGES:
        out = attach_judge(out, load_pairwise_log(language, judge), judge)
    return out


# ---------------------------------------------------------------------------------------------
# Pairwise analyses
# ---------------------------------------------------------------------------------------------
def pair_mask(df: pd.DataFrame, col_a: str, col_b: str) -> pd.Series:
    return df[col_a].isin(LABELS) & df[col_b].isin(LABELS)


def agreement_row(df: pd.DataFrame, language: str, judge: str, view: str, col: str) -> dict:
    """Accuracy and Cohen's kappa vs human_majority, plus accuracy on human non-tie pairs."""
    m = pair_mask(df, col, "human_majority")
    sub = df.loc[m]
    n = int(m.sum())
    if n == 0:
        return {"language": language, "judge": judge, "view": view, "n": 0}
    acc = accuracy(sub[col], sub["human_majority"])
    kap = cohen_kappa(sub[col], sub["human_majority"])
    non_tie = sub["human_majority"] != "tie"
    n_nt = int(non_tie.sum())
    acc_nt = accuracy(sub.loc[non_tie, col], sub.loc[non_tie, "human_majority"]) if n_nt else float("nan")
    return {
        "language": language, "judge": judge, "view": view, "n": n,
        "accuracy": round(acc, 4), "kappa": round(kap, 4),
        "n_human_nontie": n_nt,
        "accuracy_human_nontie": None if n_nt == 0 else round(acc_nt, 4),
    }


def human_ceiling_row(df: pd.DataFrame, language: str, judge: str, col: str) -> dict:
    """Mean pairwise human-human kappa vs mean kappa of this judge against each individual rater."""
    raters = ["human_1", "human_2", "human_3"]
    hh = []
    for i, r1 in enumerate(raters):
        for r2 in raters[i + 1:]:
            m = pair_mask(df, r1, r2)
            hh.append(cohen_kappa(df.loc[m, r1], df.loc[m, r2]))
    hj = []
    for r in raters:
        m = pair_mask(df, col, r)
        hj.append(cohen_kappa(df.loc[m, col], df.loc[m, r]))
    return {
        "language": language, "judge": judge, "n": int(pair_mask(df, col, "human_majority").sum()),
        "human_human_kappa_mean": round(float(np.mean(hh)), 4),
        "judge_human_kappa_mean": round(float(np.mean(hj)), 4),
        "gap": round(float(np.mean(hj) - np.mean(hh)), 4),
    }


def verdict_rows(df: pd.DataFrame, language: str, judge: str, col: str) -> list[dict]:
    m = df[col].isin(LABELS)
    n = int(m.sum())
    counts = df.loc[m, col].value_counts()
    return [{"language": language, "judge": judge, "n": n,
             "label": lab, "count": int(counts.get(lab, 0)),
             "pct": round(100 * counts.get(lab, 0) / n, 2) if n else None}
            for lab in LABELS]


def position_row(df: pd.DataFrame, language: str, judge: str) -> dict:
    """Flip rate (canonical AB vs BA) and first-position preference from presented (non-tie) verdicts."""
    both = pair_mask(df, f"{judge}_ab", f"{judge}_ba")
    n_both = int(both.sum())
    flip = float((df.loc[both, f"{judge}_ab"] != df.loc[both, f"{judge}_ba"]).mean()) if n_both else float("nan")
    presented = pd.concat([df[f"{judge}_ab_presented"], df[f"{judge}_ba_presented"]], ignore_index=True)
    presented = presented[presented.isin(("A", "B", "tie"))]
    nontie = presented[presented.isin(("A", "B"))]
    n_nt = int(nontie.size)
    first_pref = float((nontie == "A").mean() - (nontie == "B").mean()) if n_nt else float("nan")
    return {
        "language": language, "judge": judge, "n_both_orders": n_both,
        "flip_rate": None if n_both == 0 else round(flip, 4),
        "n_nontie_presented": n_nt,
        "first_position_preference": None if n_nt == 0 else round(first_pref, 4),
    }


def length_row(df: pd.DataFrame, language: str, judge: str, col: str) -> dict:
    """Share of non-tie verdicts that pick the longer answer, plus a logit of A-vs-B on chars_a - chars_b."""
    sub = df[df[col].isin(("A", "B")) & df["chars_a"].notna() & df["chars_b"].notna()].copy()
    unequal = sub[sub.chars_a != sub.chars_b]
    n_uneq = int(len(unequal))
    if n_uneq:
        picked_longer = np.where(unequal[col] == "A", unequal.chars_a > unequal.chars_b,
                                 unequal.chars_b > unequal.chars_a)
        pct_longer = float(np.mean(picked_longer))
    else:
        pct_longer = float("nan")
    coef = pval = float("nan")
    n_logit = int(len(sub))
    if n_logit >= 20:
        y = (sub[col] == "A").astype(int).to_numpy()
        x = sm.add_constant((sub.chars_a - sub.chars_b).to_numpy())
        try:
            fit = sm.Logit(y, x).fit(disp=False)
            coef = float(fit.params[1])
            pval = float(fit.pvalues[1])
        except Exception:
            pass
    return {
        "language": language, "judge": judge, "n_nontie": n_logit, "n_unequal_length": n_uneq,
        "pct_choose_longer": None if n_uneq == 0 else round(100 * pct_longer, 2),
        "logit_coef_chars_a_minus_b": None if pd.isna(coef) else round(coef, 6),
        "logit_pvalue": None if pd.isna(pval) else round(pval, 6),
    }


def self_pref_row(df: pd.DataFrame, language: str, judge: str, col: str, family: str,
                  family_models: set[str]) -> dict:
    """On pairs where exactly one response is from the judge's family: % family picks vs humans."""
    in_a = df.model_a.isin(family_models)
    in_b = df.model_b.isin(family_models)
    exclusive = in_a ^ in_b
    sub = df.loc[exclusive & df[col].isin(LABELS) & df.human_majority.isin(LABELS)].copy()
    n = int(len(sub))
    family_is_a = sub.model_a.isin(family_models)

    def pick_rate(series: pd.Series) -> float:
        # A pick of the family model counts 1, the other model 0, a tie 0.5.
        chose_a = (series == "A").to_numpy()
        chose_b = (series == "B").to_numpy()
        fam_a = family_is_a.to_numpy()
        return float(np.mean(np.where(chose_a, fam_a, np.where(chose_b, ~fam_a, 0.5))))

    return {
        "language": language, "judge": judge, "family": family, "n_exclusive": n,
        "judge_family_pick_pct": None if n == 0 else round(100 * pick_rate(sub[col]), 2),
        "human_family_pick_pct": None if n == 0 else round(100 * pick_rate(sub["human_majority"]), 2),
    }


def win_rate(df: pd.DataFrame, verdict_col: str) -> pd.Series:
    """Per-model win rate, counting a tie as 0.5, over unique pairs with a valid verdict."""
    sub = df[df[verdict_col].isin(LABELS)]
    wins: dict[str, list[float]] = {}
    for row in sub.itertuples(index=False):
        v = getattr(row, verdict_col)
        a, b = row.model_a, row.model_b
        wins.setdefault(a, [])
        wins.setdefault(b, [])
        if v == "A":
            wins[a].append(1.0)
            wins[b].append(0.0)
        elif v == "B":
            wins[a].append(0.0)
            wins[b].append(1.0)
        else:
            wins[a].append(0.5)
            wins[b].append(0.5)
    return pd.Series({m: float(np.mean(s)) for m, s in wins.items()}).sort_values(ascending=False)


def leaderboard_block(df: pd.DataFrame, language: str, judge_cols: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Model win-rate table and Kendall's tau of each judge ranking vs the human ranking."""
    human = win_rate(df, "human_majority")
    table = pd.DataFrame({"language": language, "model": human.index, "human_win_rate": human.values})
    tau_rows = []
    for judge, col in judge_cols.items():
        wr = win_rate(df, col)
        table = table.merge(wr.rename(f"{judge}_win_rate").reset_index()
                            .rename(columns={"index": "model"}), on="model", how="outer")
        common = human.index.intersection(wr.index)
        tau, p = kendalltau(human.loc[common].rank(ascending=False),
                            wr.loc[common].rank(ascending=False))
        tau_rows.append({"language": language, "judge": judge, "n_models": int(len(common)),
                         "kendall_tau": round(float(tau), 4), "pvalue": round(float(p), 6)})
    return table, pd.DataFrame(tau_rows)


# ---------------------------------------------------------------------------------------------
# Clustered bootstrap
# ---------------------------------------------------------------------------------------------
def prompt_clusters(df: pd.DataFrame) -> tuple[np.ndarray, list[np.ndarray]]:
    """Prompt ids and, for each, the integer positions of its rows in `df`."""
    prompts = df.prompt_id.to_numpy()
    unique = np.unique(prompts)
    clusters = [np.flatnonzero(prompts == p) for p in unique]
    return unique, clusters


def resample_positions(clusters: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Draw clusters with replacement and concatenate their row positions (a cluster bootstrap)."""
    draw = rng.integers(0, len(clusters), size=len(clusters))
    return np.concatenate([clusters[i] for i in draw])


def bootstrap_metrics(df: pd.DataFrame, judge: str, col: str, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """2,000 prompt-clustered replicates of kappa, flip rate and tie rate for one judge."""
    _, clusters = prompt_clusters(df)
    human = df["human_majority"].to_numpy()
    judge_v = df[col].to_numpy()
    has_flip = f"{judge}_ab" in df.columns and f"{judge}_ba" in df.columns
    ab = df[f"{judge}_ab"].to_numpy() if has_flip else None
    ba = df[f"{judge}_ba"].to_numpy() if has_flip else None
    kappas = np.empty(N_BOOT)
    flips = np.full(N_BOOT, np.nan)
    ties = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pos = resample_positions(clusters, rng)
        j, h = judge_v[pos], human[pos]
        ok = pd.Series(j).isin(LABELS).to_numpy() & pd.Series(h).isin(LABELS).to_numpy()
        kappas[b] = kappa_np(j[ok], h[ok])
        ties[b] = np.mean(j[ok] == "tie") if ok.any() else np.nan
        if has_flip:
            a, bcol = ab[pos], ba[pos]
            both = pd.Series(a).isin(LABELS).to_numpy() & pd.Series(bcol).isin(LABELS).to_numpy()
            flips[b] = np.mean(a[both] != bcol[both]) if both.any() else np.nan
    return {"kappa": kappas, "flip_rate": flips, "tie_rate": ties}


def bootstrap_delta_same_lang(df: pd.DataFrame, col_a: str, col_b: str,
                              rng: np.random.Generator) -> np.ndarray:
    """Paired cluster bootstrap of kappa(col_a) - kappa(col_b) on the same resampled prompts."""
    _, clusters = prompt_clusters(df)
    human = df["human_majority"].to_numpy()
    va, vb = df[col_a].to_numpy(), df[col_b].to_numpy()
    diffs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pos = resample_positions(clusters, rng)
        h = human[pos]
        diffs[b] = kappa_np_ok(va[pos], h) - kappa_np_ok(vb[pos], h)
    return diffs


def kappa_np_ok(j: np.ndarray, h: np.ndarray) -> float:
    ok = pd.Series(j).isin(LABELS).to_numpy() & pd.Series(h).isin(LABELS).to_numpy()
    return kappa_np(j[ok], h[ok]) if ok.any() else float("nan")


def bootstrap_delta_two_langs(df_ta: pd.DataFrame, df_hi: pd.DataFrame, col: str,
                              rng: np.random.Generator) -> np.ndarray:
    """Independent prompt-clustered bootstrap of Tamil kappa minus Hindi kappa, same judge."""
    _, c_ta = prompt_clusters(df_ta)
    _, c_hi = prompt_clusters(df_hi)
    diffs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        p_ta = resample_positions(c_ta, rng)
        p_hi = resample_positions(c_hi, rng)
        k_ta = kappa_np_ok(df_ta[col].to_numpy()[p_ta], df_ta["human_majority"].to_numpy()[p_ta])
        k_hi = kappa_np_ok(df_hi[col].to_numpy()[p_hi], df_hi["human_majority"].to_numpy()[p_hi])
        diffs[b] = k_ta - k_hi
    return diffs


# ---------------------------------------------------------------------------------------------
# Tamil direct scores
# ---------------------------------------------------------------------------------------------
def load_direct_log(judge: str) -> pd.DataFrame:
    path = CLEAN / f"tamil_direct__{judge}.jsonl.gz"
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            ok = rec.get("parse_ok") is True
            scores = rec.get("direct_scores") if ok else None
            pid, model = rec["item_id"].split("|", 1)
            rows.append({
                "prompt_id": pid, "model": model,
                "LA": None if not scores else scores.get("LA"),
                "TQ": None if not scores else scores.get("TQ"),
                "H": None if not scores else scores.get("H"),
            })
    out = pd.DataFrame(rows)
    out["score"] = np.where(out.LA.notna() & out.TQ.notna() & out.H.notna(),
                            out.LA + out.TQ + 1 - out.H, np.nan)
    return out


def qwk(y_true, y_pred, labels) -> float:
    mask = pd.notna(y_true) & pd.notna(y_pred)
    if mask.sum() < 2:
        return float("nan")
    return float(cohen_kappa_score(y_true[mask].astype(int), y_pred[mask].astype(int),
                                   labels=list(labels), weights="quadratic"))


def spearman(y_true, y_pred) -> float:
    mask = pd.notna(y_true) & pd.notna(y_pred)
    if mask.sum() < 3:
        return float("nan")
    rho, _ = spearmanr(y_true[mask], y_pred[mask])
    return float(rho)


def direct_rows(humans: pd.DataFrame) -> list[dict]:
    rows = [_direct_one("gpt4_32k", humans, humans["gpt4_LA"], humans["gpt4_TQ"],
                        humans["gpt4_H"], humans["gpt4_score"])]
    for judge in LLM_JUDGES:
        log = load_direct_log(judge)
        merged = humans.merge(log, on=["prompt_id", "model"], how="left")
        rows.append(_direct_one(judge, merged, merged["LA"], merged["TQ"], merged["H"], merged["score"]))
    return rows


def _direct_one(judge: str, frame: pd.DataFrame, la, tq, h, score) -> dict:
    return {
        "judge": judge, "n_score": int((pd.notna(score) & pd.notna(frame.human_score)).sum()),
        "spearman_score": round(spearman(frame.human_score, score), 4),
        "qwk_score": round(qwk(frame.human_score, score, range(6)), 4),
        "spearman_LA": round(spearman(frame.human_LA, la), 4),
        "qwk_LA": round(qwk(frame.human_LA, la, range(3)), 4),
        "spearman_TQ": round(spearman(frame.human_TQ, tq), 4),
        "qwk_TQ": round(qwk(frame.human_TQ, tq, range(3)), 4),
        "spearman_H": round(spearman(frame.human_H, h), 4),
        "qwk_H": round(qwk(frame.human_H, h, range(2)), 4),
    }


# ---------------------------------------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------------------------------------
def write_summary(agreement: pd.DataFrame, ceiling: pd.DataFrame, position: pd.DataFrame,
                  length: pd.DataFrame, selfpref: pd.DataFrame, tau: pd.DataFrame,
                  contrasts: pd.DataFrame, direct: pd.DataFrame, dist: pd.DataFrame) -> None:
    def k(lang, judge):
        row = agreement[(agreement.language == lang) & (agreement.judge == judge)
                        & (agreement.view == ("ab" if judge == "gpt4_32k" else MAIN_VIEW))]
        return row.iloc[0] if len(row) else None

    lines = [
        "# Step 04 summary: agreement and bias",
        "",
        "Unique pairs only (Tamil 1,820, Hindi 1,000). Ties kept. `parse_ok=false` treated as missing. "
        "Main pairwise view is **swap-aggregated** (A/B and mapped B/A; disagreement -> tie). "
        "`gpt4_32k` is A/B order only. Seed 42, 2,000 prompt-clustered bootstrap replicates.",
        "",
        "## Agreement with human_majority (swap-aggregated; gpt4_32k = A/B)",
        "",
        "| language | judge | n | accuracy | kappa | accuracy (human non-tie) |",
        "|---|---|---|---|---|---|",
    ]
    for lang in ("tamil", "hindi"):
        for judge in (*LLM_JUDGES, "gpt4_32k"):
            r = k(lang, judge)
            if r is None:
                continue
            lines.append(f"| {lang} | {judge} | {r.n} | {r.accuracy:.3f} | {r.kappa:.3f} | "
                         f"{r.accuracy_human_nontie} |")
    lines += ["", "## Human ceiling (mean pairwise Cohen's kappa)", "",
              "| language | judge | human–human | judge–human | gap |",
              "|---|---|---|---|---|"]
    for _, r in ceiling.iterrows():
        lines.append(f"| {r.language} | {r.judge} | {r.human_human_kappa_mean:.3f} | "
                     f"{r.judge_human_kappa_mean:.3f} | {r.gap:.3f} |")
    # Tie rates from the distribution table.
    lines += ["", "## Verdict mix (% tie) vs humans", ""]
    human_tie = dist[(dist.judge == "human_majority") & (dist.label == "tie")]
    for _, r in human_tie.iterrows():
        lines.append(f"- {r.language} humans: {r.pct:.1f}% ties")
    for lang in ("tamil", "hindi"):
        bits = []
        for judge in (*LLM_JUDGES, "gpt4_32k"):
            row = dist[(dist.language == lang) & (dist.judge == judge) & (dist.label == "tie")]
            if len(row):
                bits.append(f"{judge} {row.iloc[0].pct:.1f}%")
        lines.append(f"- {lang} judges: " + ", ".join(bits))
    lines += ["", "## Position bias (LLM judges, both orders)", "",
              "| language | judge | flip rate | first-position preference |",
              "|---|---|---|---|"]
    for _, r in position.iterrows():
        lines.append(f"| {r.language} | {r.judge} | {r.flip_rate} | {r.first_position_preference} |")
    lines += ["", "## Length bias (% of non-tie verdicts choosing the longer answer)", "",
              "| language | judge | n | % longer | logit p |",
              "|---|---|---|---|---|"]
    for _, r in length.iterrows():
        lines.append(f"| {r.language} | {r.judge} | {r.n_nontie} | {r.pct_choose_longer} | {r.logit_pvalue} |")
    lines += ["", "## Self-preference (exactly one family member in the pair; tie = 0.5)", "",
              "| language | judge | family | n | judge % | human % |",
              "|---|---|---|---|---|---|"]
    for _, r in selfpref.iterrows():
        lines.append(f"| {r.language} | {r.judge} | {r.family} | {r.n_exclusive} | "
                     f"{r.judge_family_pick_pct} | {r.human_family_pick_pct} |")
    lines += ["", "## Leaderboard ranking agreement (Kendall's tau vs human win rates)", "",
              "| language | judge | n models | tau | p |",
              "|---|---|---|---|---|"]
    for _, r in tau.iterrows():
        lines.append(f"| {r.language} | {r.judge} | {r.n_models} | {r.kendall_tau:.3f} | {r.pvalue} |")
    lines += ["", "## Pre-specified contrasts (prompt-clustered bootstrap, Holm-adjusted)", "",
              "| contrast | estimate | 95% CI | p | p_Holm |",
              "|---|---|---|---|---|"]
    for _, r in contrasts.iterrows():
        lines.append(f"| {r.contrast} | {r.estimate:.3f} | [{r.ci_low:.3f}, {r.ci_high:.3f}] | "
                     f"{r.p_boot} | {r.p_holm} |")
    lines += ["", "## Tamil direct scores (0–5 = LA + TQ + (1 − H))", "",
              "| judge | n | Spearman | QWK | QWK LA | QWK TQ | QWK H |",
              "|---|---|---|---|---|---|---|"]
    for _, r in direct.iterrows():
        lines.append(f"| {r.judge} | {r.n_score} | {r.spearman_score:.3f} | {r.qwk_score:.3f} | "
                     f"{r.qwk_LA:.3f} | {r.qwk_TQ:.3f} | {r.qwk_H:.3f} |")
    lines += ["", "All numbers are computed from `results/judgements_clean/` and `data/processed/`. "
              "No new API calls.", ""]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------
def main() -> None:
    if not CLEAN.is_dir():
        sys.exit(f"{CLEAN} not found: run scripts/03b_package_judgements.py first")
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    families = {k: set(v) for k, v in config["model_families"].items() if not k.startswith("_")}
    judge_cfg = config["judges"]
    rng = np.random.default_rng(SEED)

    frames = {lang: attach_all_judges(load_pairs(lang), lang) for lang in ("tamil", "hindi")}

    agreement, ceiling, dist, position, length, selfpref = [], [], [], [], [], []
    lb_tables, tau_tables = [], []
    boot_ci = []
    # Point estimates.
    for lang, df in frames.items():
        # Humans as a reference distribution.
        dist.extend(verdict_rows(df, lang, "human_majority", "human_majority"))
        dist.extend(verdict_rows(df, lang, "gpt4_32k", "gpt4_32k_ab"))
        length.append(length_row(df, lang, "human_majority", "human_majority"))
        judge_cols = {"gpt4_32k": "gpt4_32k_ab"}
        for judge in LLM_JUDGES:
            agreement.append(agreement_row(df, lang, judge, "ab", f"{judge}_ab"))
            agreement.append(agreement_row(df, lang, judge, "ba", f"{judge}_ba"))
            agreement.append(agreement_row(df, lang, judge, "swap_agg", f"{judge}_swap"))
            ceiling.append(human_ceiling_row(df, lang, judge, f"{judge}_swap"))
            dist.extend(verdict_rows(df, lang, judge, f"{judge}_swap"))
            position.append(position_row(df, lang, judge))
            length.append(length_row(df, lang, judge, f"{judge}_swap"))
            fam = judge_cfg[judge]["family"]
            selfpref.append(self_pref_row(df, lang, judge, f"{judge}_swap", fam, families[fam]))
            judge_cols[judge] = f"{judge}_swap"
        agreement.append(agreement_row(df, lang, "gpt4_32k", "ab", "gpt4_32k_ab"))
        ceiling.append(human_ceiling_row(df, lang, "gpt4_32k", "gpt4_32k_ab"))
        length.append(length_row(df, lang, "gpt4_32k", "gpt4_32k_ab"))
        selfpref.append(self_pref_row(df, lang, "gpt4_32k", "gpt4_32k_ab", "gpt", families["gpt"]))
        lb, tau = leaderboard_block(df, lang, judge_cols)
        lb_tables.append(lb)
        tau_tables.append(tau)

        # Clustered bootstrap CIs for kappa, flip, tie (main view).
        for judge in (*LLM_JUDGES, "gpt4_32k"):
            col = "gpt4_32k_ab" if judge == "gpt4_32k" else f"{judge}_swap"
            print(f"  bootstrap {lang} {judge} ...", flush=True)
            samples = bootstrap_metrics(df, judge, col, rng)
            for metric, arr in samples.items():
                if np.all(np.isnan(arr)):
                    continue
                lo, hi = ci95(arr)
                boot_ci.append({"language": lang, "judge": judge, "metric": metric,
                                "estimate": round(float(np.nanmean(arr)), 4),
                                "ci_low": round(lo, 4), "ci_high": round(hi, 4)})

    # Pre-specified paired contrasts.
    print("  bootstrap contrasts ...", flush=True)
    contrast_rows = []
    # SEA-LION minus base_model kappa, each language (paired: same resampled prompts).
    for lang, df in frames.items():
        diffs = bootstrap_delta_same_lang(df, "sea_lion_swap", "base_model_swap", rng)
        lo, hi = ci95(diffs)
        contrast_rows.append({
            "contrast": f"kappa {lang}: sea_lion − base_model",
            "estimate": round(float(np.nanmean(diffs)), 4),
            "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "p_boot": round(two_sided_p(diffs), 4),
        })
    # Tamil minus Hindi kappa, each judge (independent prompt resamples per language).
    for judge, col in [(j, f"{j}_swap") for j in LLM_JUDGES] + [("gpt4_32k", "gpt4_32k_ab")]:
        diffs = bootstrap_delta_two_langs(frames["tamil"], frames["hindi"], col, rng)
        lo, hi = ci95(diffs)
        contrast_rows.append({
            "contrast": f"kappa {judge}: tamil − hindi",
            "estimate": round(float(np.nanmean(diffs)), 4),
            "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "p_boot": round(two_sided_p(diffs), 4),
        })
    contrasts = pd.DataFrame(contrast_rows)
    contrasts["p_holm"] = np.round(holm(list(contrasts.p_boot)), 4)

    print("  tamil direct scores ...", flush=True)
    humans = pd.read_csv(PROCESSED / "direct_tamil.csv")
    direct = pd.DataFrame(direct_rows(humans))

    agreement_df = pd.DataFrame(agreement)
    ceiling_df = pd.DataFrame(ceiling)
    dist_df = pd.DataFrame(dist)
    position_df = pd.DataFrame(position)
    length_df = pd.DataFrame(length)
    selfpref_df = pd.DataFrame(selfpref)
    tau_df = pd.concat(tau_tables, ignore_index=True)
    lb_df = pd.concat(lb_tables, ignore_index=True)
    boot_df = pd.DataFrame(boot_ci)

    agreement_df.to_csv(OUT / "agreement.csv", index=False)
    ceiling_df.to_csv(OUT / "human_ceiling.csv", index=False)
    dist_df.to_csv(OUT / "verdict_distribution.csv", index=False)
    position_df.to_csv(OUT / "position_bias.csv", index=False)
    length_df.to_csv(OUT / "length_bias.csv", index=False)
    selfpref_df.to_csv(OUT / "self_preference.csv", index=False)
    lb_df.to_csv(OUT / "leaderboard.csv", index=False)
    tau_df.to_csv(OUT / "leaderboard_tau.csv", index=False)
    boot_df.to_csv(OUT / "bootstrap_ci.csv", index=False)
    contrasts.to_csv(OUT / "bootstrap_contrasts.csv", index=False)
    direct.to_csv(OUT / "direct_agreement.csv", index=False)
    write_summary(agreement_df, ceiling_df, position_df, length_df, selfpref_df,
                  tau_df, contrasts, direct, dist_df)
    print(f"Wrote {len(list(OUT.glob('*')))} files under {OUT.relative_to(ROOT).as_posix()}/")


if __name__ == "__main__":
    main()
