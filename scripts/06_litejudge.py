"""Step 06 (objective O5): LiteJudge, a small tie-aware pairwise judge on frozen MuRIL embeddings.

PRE-REGISTRATION (fixed 10 Nov 2026, before any step 06 result was seen; also in README.md)
  * PRIMARY model = MuRIL embeddings only, NO length feature. This thesis is about judge biases and
    humans pick the longer answer 55-61% of the time, so the main model must not be handed the
    human length bias. The same model plus a length feature is an ABLATION only.
  * Model: Bradley-Terry with a tie option (Davidson, 1970). Each response gets a linear score
    s = w . x from its embedding x. For a pair with d = s_A - s_B and tie parameter nu = exp(tau):
        P(A) = e^(d/2) / Z,   P(B) = e^(-d/2) / Z,   P(tie) = nu / Z,   Z = e^(d/2) + e^(-d/2) + nu
    (this is Davidson's formula divided through by sqrt(pi_A * pi_B), with pi = e^s).
    There is no intercept on d, so swapping A and B swaps P(A) and P(B) exactly: the model judges
    both answer orders identically by construction (position bias is zero; checked below).
  * Training labels: the three individual human votes as soft labels (1/3 each). Evaluation is
    against human_majority, with step 04's own functions (imported, not copied).
  * Unique pairs only. Tamil 1,820 pairs; Hindi = the 1,000-pair sample used for the LLM judges.
  * Outer CV: 5 folds grouped by prompt_id (4 test prompts each), shuffled with seed 42.
    Inner CV: 4 folds grouped by prompt_id inside the 16 training prompts. It chooses the L2
    strength and an optional PCA (none / 16 / 32 / 64 components); the scaler and PCA are fitted
    on training responses only. Selection criterion: mean validation log-loss against the soft votes.
  * Verdict = the label with the highest probability (argmax). No tie threshold is tuned.
  * Baselines, scored on the same outer folds with the same metrics:
      longer      - always choose the longer response (characters); never a tie; equal length -> A;
      majority    - always predict the most common human_majority label of the training fold;
      model_id    - the same Davidson model on a one-hot "which model wrote this" vector (no text).
  * Pre-specified contrasts (paired prompt-clustered bootstrap, 2,000 replicates, Holm-adjusted over
    all of them): kappa(primary) minus kappa of each of: with_length, model_id, longer, and the four
    LLM judges (swap-aggregated), in each language.

EXPLORATORY (NOT pre-registered; added 10 Nov 2026 after the results above were seen)
  * Leave-one-model-out: for each response model, train LiteJudge (primary recipe) on the pairs
    that do not involve it and test on the pairs that do; same for the model-identity baseline,
    which gives the unseen model the average strength. Pooled predictions, step 04 metrics,
    unadjusted contrasts. Written to results/litejudge/exploratory/ only; it runs after every
    pre-registered file is written, on its own random stream, so it cannot change them.

Reads:  data/embeddings/muril_<language>.npz (step 05), data/processed/pairs_*.csv, responses.csv,
        results/judgements_clean/ (via step 04, for the LLM-judge contrasts),
        results/agreement/*.csv (step 04 outputs, for the side-by-side table)
Writes: results/litejudge/ (out-of-fold predictions, agreement tables, CV choices, summary.md)
Usage:  python scripts/06_litejudge.py
No API calls. No annotator IDs or justification texts are read or written.
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
EMB = ROOT / "data" / "embeddings"
STEP04_OUT = ROOT / "results" / "agreement"
OUT = ROOT / "results" / "litejudge"

# Load step 04 as a module so the metrics are computed by the very same code as for the LLM judges.
# (A file name starting with a digit cannot be imported with a plain `import` statement.)
_spec = importlib.util.spec_from_file_location("step04", ROOT / "scripts" / "04_agreement_bias.py")
s04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s04)

SEED = 42
LANGUAGES = ("tamil", "hindi")
LABELS = s04.LABELS                     # ("A", "B", "tie"), the same order everywhere
OUTER_FOLDS = 5
INNER_FOLDS = 4
PCA_GRID = (None, 16, 32, 64)           # None = keep all 768 standardised dimensions
L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
LITE_MODELS = ("litejudge", "litejudge_length", "model_id")  # the three trained (Davidson) models
PRIMARY = "litejudge"
ALL_NEW = ("litejudge", "litejudge_length", "longer", "majority", "model_id")


# ---------------------------------------------------------------------------------------------
# The Davidson (tie-aware Bradley-Terry) model
# ---------------------------------------------------------------------------------------------
def davidson_probs(d: np.ndarray, tau: float) -> np.ndarray:
    """P(A), P(B), P(tie) for score differences d: a 3-way softmax of the logits [d/2, -d/2, tau]."""
    logits = np.column_stack([d / 2, -d / 2, np.full_like(d, tau)])
    logits -= logits.max(axis=1, keepdims=True)  # subtract the row maximum: avoids overflow in exp
    expl = np.exp(logits)
    return expl / expl.sum(axis=1, keepdims=True)


def fit_davidson(delta: np.ndarray, targets: np.ndarray, l2: float) -> tuple[np.ndarray, float]:
    """Fit w and tau by minimising  mean soft-label cross-entropy + (l2 / 2) * ||w||^2.

    delta   (n_pairs, n_features): x_A - x_B, the feature difference of the two responses
    targets (n_pairs, 3): share of the three human votes that said A, B, tie
    tau is not penalised (it only sets how often ties are predicted).
    """
    n, k = delta.shape

    def loss_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
        w, tau = params[:k], params[k]
        p = davidson_probs(delta @ w, tau)
        loss = -np.mean(np.sum(targets * np.log(p + 1e-12), axis=1)) + 0.5 * l2 * (w @ w)
        g = (p - targets) / n                 # gradient of the mean cross-entropy w.r.t. the 3 logits
        g_d = 0.5 * g[:, 0] - 0.5 * g[:, 1]   # chain rule: logit A = d/2, logit B = -d/2
        grad_w = delta.T @ g_d + l2 * w
        grad_tau = g[:, 2].sum()
        return float(loss), np.append(grad_w, grad_tau)

    start = np.zeros(k + 1)  # w = 0 and tau = 0: every pair starts at P = 1/3 each
    result = minimize(loss_and_grad, start, jac=True, method="L-BFGS-B",
                      options={"maxiter": 2000})
    return result.x[:k], float(result.x[k])


def log_loss(p: np.ndarray, targets: np.ndarray) -> float:
    """Mean cross-entropy of predicted probabilities against the soft human votes (lower is better)."""
    return float(-np.mean(np.sum(targets * np.log(p + 1e-12), axis=1)))


# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------
def load_language(language: str) -> tuple[pd.DataFrame, dict]:
    """Unique pairs (with step 04's length columns and LLM verdicts) and response features."""
    pairs = s04.attach_all_judges(s04.load_pairs(language), language)
    votes = pairs[["human_1", "human_2", "human_3"]].to_numpy()
    # Soft labels: for each pair, the share of the three raters who voted A, B and tie.
    pairs[["t_A", "t_B", "t_tie"]] = np.column_stack([(votes == lab).mean(axis=1) for lab in LABELS])

    emb = np.load(EMB / f"muril_{language}.npz", allow_pickle=False)
    keys = list(zip(emb["prompt_id"], emb["model"]))
    index = {key: i for i, key in enumerate(keys)}
    responses = pd.read_csv(ROOT / "data" / "processed" / "responses.csv")
    responses = responses[responses.language == language].set_index(["prompt_id", "model"])
    chars = np.array([responses.loc[key, "response_chars"] for key in keys], dtype=float)
    models = sorted(set(emb["model"]))
    features = {
        "index": index,                                             # (prompt_id, model) -> row
        "prompt_id": emb["prompt_id"],
        "model": emb["model"],                                      # used by the exploratory check
        "embedding": emb["embeddings"].astype(np.float64),          # (n_responses, 768)
        "log_chars": np.log(chars),                                 # used by the ablation only
        "one_hot": np.array([[m == name for name in models] for m in emb["model"]], dtype=float),
    }
    # Row numbers of response A and response B for every pair.
    pairs["row_a"] = [index[(p, m)] for p, m in zip(pairs.prompt_id, pairs.model_a)]
    pairs["row_b"] = [index[(p, m)] for p, m in zip(pairs.prompt_id, pairs.model_b)]
    return pairs, features


def response_matrix(kind: str, features: dict, train_rows: np.ndarray, pca_k: int | None) -> np.ndarray:
    """Per-response feature matrix for one model kind, with any scaling fitted on train_rows only."""
    if kind == "model_id":
        return features["one_hot"]  # 0/1 indicators need no scaling
    scaler = StandardScaler().fit(features["embedding"][train_rows])
    x = scaler.transform(features["embedding"])
    if pca_k is not None:
        pca = PCA(n_components=pca_k, random_state=SEED).fit(x[train_rows])
        x = pca.transform(x)
        x = x / x[train_rows].std(axis=0)  # put every component on unit scale so one L2 fits all
    if kind == "litejudge_length":
        lc = features["log_chars"]
        z = (lc - lc[train_rows].mean()) / lc[train_rows].std()
        x = np.column_stack([x, z])  # the ablation's one extra feature: standardised log length
    return x


def pair_delta(x: np.ndarray, pairs: pd.DataFrame) -> np.ndarray:
    """x_A - x_B for every pair."""
    return x[pairs.row_a.to_numpy()] - x[pairs.row_b.to_numpy()]


def targets_of(pairs: pd.DataFrame) -> np.ndarray:
    return pairs[["t_A", "t_B", "t_tie"]].to_numpy()


def train_rows_of(features: dict, prompts: np.ndarray, exclude_model: str | None = None) -> np.ndarray:
    """Rows of the responses that belong to the given prompts (never test-prompt responses).

    exclude_model (exploratory leave-one-model-out only): also leave out that model's responses,
    so the scaler and PCA never see the held-out model. None = the pre-registered behaviour.
    """
    keep = np.isin(features["prompt_id"], prompts)
    if exclude_model is not None:
        keep &= features["model"] != exclude_model
    return np.flatnonzero(keep)


# ---------------------------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------------------------
def fit_and_predict(kind: str, features: dict, train: pd.DataFrame, test: pd.DataFrame,
                    pca_k: int | None, l2: float, exclude_model: str | None = None) -> tuple[np.ndarray, dict]:
    """Fit one Davidson model on `train` pairs and return probabilities for `test` pairs."""
    rows = train_rows_of(features, train.prompt_id.unique(), exclude_model)
    x = response_matrix(kind, features, rows, pca_k)
    w, tau = fit_davidson(pair_delta(x, train), targets_of(train), l2)
    return davidson_probs(pair_delta(x, test) @ w, tau), {"w": w, "tau": tau, "x": x}


def tune(kind: str, features: dict, train: pd.DataFrame,
         exclude_model: str | None = None) -> tuple[int | None, float, float]:
    """Inner grouped CV on the training prompts: pick (PCA, L2) with the lowest mean validation log-loss."""
    pca_grid = (None,) if kind == "model_id" else PCA_GRID  # PCA makes no sense for one-hot model ids
    inner = GroupKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
    splits = list(inner.split(train, groups=train.prompt_id))
    best = (None, None, np.inf)
    for pca_k in pca_grid:
        for l2 in L2_GRID:
            losses = []
            for tr, va in splits:
                p, _ = fit_and_predict(kind, features, train.iloc[tr], train.iloc[va], pca_k, l2, exclude_model)
                losses.append(log_loss(p, targets_of(train.iloc[va])))
            if np.mean(losses) < best[2]:
                best = (pca_k, l2, float(np.mean(losses)))
    return best


def to_verdict(p: np.ndarray) -> np.ndarray:
    """The label with the highest probability."""
    return np.array(LABELS)[p.argmax(axis=1)]


def run_cv(language: str, pairs: pd.DataFrame, features: dict) -> tuple[pd.DataFrame, list[dict], dict]:
    """Out-of-fold predictions for all five new judges, plus the chosen settings and timings."""
    outer = GroupKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
    preds = pd.DataFrame({"pairing_id": pairs.pairing_id, "prompt_id": pairs.prompt_id, "fold": -1})
    for judge in ALL_NEW:
        preds[judge] = None
    for kind in LITE_MODELS:
        preds[[f"{kind}_p_A", f"{kind}_p_B", f"{kind}_p_tie"]] = np.nan
    choices, timing = [], {"fit_seconds": 0.0, "predict_seconds": 0.0, "n_predicted": 0}
    swap_worst = 0.0

    for fold, (tr, te) in enumerate(outer.split(pairs, groups=pairs.prompt_id)):
        train, test = pairs.iloc[tr], pairs.iloc[te]
        preds.loc[preds.index[te], "fold"] = fold

        # Baseline: always the longer answer (step 04's character counts); never a tie.
        preds.loc[preds.index[te], "longer"] = np.where(test.chars_a >= test.chars_b, "A", "B")
        # Baseline: the most common human_majority label among the training pairs.
        preds.loc[preds.index[te], "majority"] = train.human_majority.value_counts().idxmax()

        for kind in LITE_MODELS:
            pca_k, l2, inner_loss = tune(kind, features, train)
            t0 = time.process_time()
            p, fitted = fit_and_predict(kind, features, train, test, pca_k, l2)
            t1 = time.process_time()
            # Order check (AGENTS.md: judge both orders): score the swapped pairs B/A, map back.
            swapped = davidson_probs(pair_delta(fitted["x"], test.rename(
                columns={"row_a": "row_b", "row_b": "row_a"})) @ fitted["w"], fitted["tau"])
            swap_worst = max(swap_worst, float(np.abs(swapped[:, [1, 0, 2]] - p).max()))
            if kind == PRIMARY:  # CPU time of the primary model, for agreement per US$ in step 07
                timing["fit_seconds"] += t1 - t0  # one refit on the outer training pairs
                t2 = time.process_time()
                davidson_probs(pair_delta(fitted["x"], test) @ fitted["w"], fitted["tau"])
                timing["predict_seconds"] += time.process_time() - t2
                timing["n_predicted"] += len(test)
            preds.loc[preds.index[te], [f"{kind}_p_A", f"{kind}_p_B", f"{kind}_p_tie"]] = p
            preds.loc[preds.index[te], kind] = to_verdict(p)
            choices.append({"language": language, "fold": fold, "judge": kind,
                            "test_prompts": int(test.prompt_id.nunique()), "n_train": len(train),
                            "n_test": len(test), "pca": "none" if pca_k is None else pca_k,
                            "l2": l2, "inner_log_loss": round(inner_loss, 4),
                            "test_log_loss": round(log_loss(p, targets_of(test)), 4),
                            "tau": round(fitted["tau"], 4)})
        print(f"  {language} fold {fold}: " + ", ".join(
            f"{c['judge']} pca={c['pca']} l2={c['l2']:g}" for c in choices[-len(LITE_MODELS):]), flush=True)

    if swap_worst > 1e-9:
        raise SystemExit(f"{language}: swapping A and B changed a prediction by {swap_worst:.1e}")
    timing["swap_check_max_abs_diff"] = swap_worst
    return preds, choices, timing


# ---------------------------------------------------------------------------------------------
# Evaluation with step 04's functions
# ---------------------------------------------------------------------------------------------
def evaluate(language: str, pairs: pd.DataFrame, preds: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Every step 04 metric that applies to the new judges, on the out-of-fold predictions."""
    df = pairs.merge(preds[["pairing_id", *ALL_NEW]], on="pairing_id", how="left")
    out = {k: [] for k in ("agreement", "ceiling", "dist", "length", "boot", "tau", "lb")}
    for judge in ALL_NEW:
        out["agreement"].append(s04.agreement_row(df, language, judge, "oof", judge))
        out["ceiling"].append(s04.human_ceiling_row(df, language, judge, judge))
        out["dist"].extend(s04.verdict_rows(df, language, judge, judge, view="oof"))
        out["length"].append(s04.length_row(df, language, judge, judge))
        for metric, arr in s04.bootstrap_metrics(df, judge, judge, rng).items():
            if np.all(np.isnan(arr)):
                continue  # flip rate: LiteJudge has no separate A/B and B/A runs
            lo, hi = s04.ci95(arr)
            out["boot"].append({"language": language, "judge": judge, "metric": metric,
                                "estimate": round(float(np.nanmean(arr)), 4),
                                "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
    # Leaderboard: win rates and Kendall's tau vs humans (the majority baseline ranks nobody).
    lb, tau = s04.leaderboard_block(df, language, {j: j for j in ALL_NEW if j != "majority"})
    out["lb"].append(lb)
    out["tau"].append(tau)
    return {"frame": df, **out}


def contrasts(language: str, df: pd.DataFrame, rng: np.random.Generator) -> list[dict]:
    """Pre-specified paired bootstrap contrasts: kappa(primary) minus kappa(other), same resamples."""
    others = ["litejudge_length", "model_id", "longer"] + [f"{j}_swap" for j in s04.LLM_JUDGES]
    rows = []
    for other in others:
        diffs = s04.bootstrap_delta_same_lang(df, PRIMARY, other, rng)
        lo, hi = s04.ci95(diffs)
        rows.append({"contrast": f"kappa {language}: {PRIMARY} − {other.removesuffix('_swap')}",
                     "estimate": round(float(np.nanmean(diffs)), 4),
                     "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                     "p_boot": round(s04.two_sided_p(diffs), 4)})
    return rows


def comparison_table(agreement: pd.DataFrame, ceiling: pd.DataFrame, dist: pd.DataFrame,
                     boot: pd.DataFrame, length: pd.DataFrame, tau: pd.DataFrame) -> pd.DataFrame:
    """One row per judge and language: new judges (this step) next to the LLM judges (step 04 CSVs)."""
    # Step 04 rows: the main view (swap-aggregated; gpt4_32k = A/B only).
    a4 = pd.read_csv(STEP04_OUT / "agreement.csv")
    a4 = a4[(a4.view == "swap_agg") | (a4.judge == "gpt4_32k")]
    d4 = pd.read_csv(STEP04_OUT / "verdict_distribution.csv")
    d4 = d4[(d4.view == "swap_agg") | (d4.judge == "gpt4_32k") | (d4.judge == "human_majority")]
    tables = {
        "agreement": pd.concat([a4, agreement], ignore_index=True),
        "ceiling": pd.concat([pd.read_csv(STEP04_OUT / "human_ceiling.csv"), ceiling], ignore_index=True),
        "dist": pd.concat([d4, dist], ignore_index=True),
        "boot": pd.concat([pd.read_csv(STEP04_OUT / "bootstrap_ci.csv"), boot], ignore_index=True),
        "length": pd.concat([pd.read_csv(STEP04_OUT / "length_bias.csv"), length], ignore_index=True),
        "tau": pd.concat([pd.read_csv(STEP04_OUT / "leaderboard_tau.csv"), tau], ignore_index=True),
    }
    ties = tables["dist"][tables["dist"].label == "tie"][["language", "judge", "pct"]].rename(columns={"pct": "pct_tie"})
    kci = tables["boot"][tables["boot"].metric == "kappa"][["language", "judge", "ci_low", "ci_high"]]
    kci = kci.rename(columns={"ci_low": "kappa_ci_low", "ci_high": "kappa_ci_high"})
    table = (tables["agreement"][["language", "judge", "n", "accuracy", "kappa"]]
             .merge(kci, on=["language", "judge"], how="left")
             .merge(tables["ceiling"][["language", "judge", "human_human_kappa_mean", "judge_human_kappa_mean", "gap"]],
                    on=["language", "judge"], how="left")
             .merge(ties, on=["language", "judge"], how="left")
             .merge(tables["length"][["language", "judge", "pct_choose_longer"]], on=["language", "judge"], how="left")
             .merge(tables["tau"][["language", "judge", "kendall_tau"]], on=["language", "judge"], how="left"))
    order = ["frontier", "low_cost", "sea_lion", "base_model", "gpt4_32k", *ALL_NEW]
    table["_o"] = table.judge.map({j: i for i, j in enumerate(order)})
    table["_l"] = table.language.map({l: i for i, l in enumerate(LANGUAGES)})
    human = (tables["dist"][(tables["dist"].judge == "human_majority") & (tables["dist"].label == "tie")]
             .set_index("language")["pct"])
    table["human_pct_tie"] = table.language.map(human)
    return table.sort_values(["_l", "_o"]).drop(columns=["_o", "_l"]).reset_index(drop=True)


def write_summary(table: pd.DataFrame, contrast_df: pd.DataFrame, choices: pd.DataFrame,
                  confusion: pd.DataFrame, timing: dict) -> None:
    def fmt(v, nd=3):
        return "" if pd.isna(v) else f"{v:.{nd}f}"
    lines = ["# Step 06 summary: LiteJudge vs LLM judges and baselines", "",
             "Out-of-fold predictions (5 outer folds grouped by prompt_id, seed 42); every metric is "
             "computed with step 04's functions on the same unique pairs. LLM judges are swap-aggregated "
             "(gpt4_32k = A/B only). Primary LiteJudge = MuRIL embeddings only (pre-registered).", ""]
    for lang in LANGUAGES:
        sub = table[table.language == lang]
        lines += [f"## {lang.capitalize()} (human ties {fmt(sub.human_pct_tie.iloc[0], 1)}%, "
                  f"human–human κ {fmt(sub.human_human_kappa_mean.iloc[0])})", "",
                  "| judge | n | accuracy | κ vs majority | 95% CI | κ vs raters | gap to ceiling | % tie | % longer | Kendall τ |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in sub.itertuples():
            lines.append(f"| {r.judge} | {r.n} | {fmt(r.accuracy)} | {fmt(r.kappa)} | "
                         f"[{fmt(r.kappa_ci_low)}, {fmt(r.kappa_ci_high)}] | {fmt(r.judge_human_kappa_mean)} | "
                         f"{fmt(r.gap)} | {fmt(r.pct_tie, 1)} | {fmt(r.pct_choose_longer, 1)} | {fmt(r.kendall_tau)} |")
        lines.append("")
    lines += ["## Pre-specified contrasts (paired prompt-clustered bootstrap, Holm over all rows)", "",
              "| contrast | estimate | 95% CI | p | p_Holm |", "|---|---|---|---|---|"]
    for r in contrast_df.itertuples():
        lines.append(f"| {r.contrast} | {fmt(r.estimate)} | [{fmt(r.ci_low)}, {fmt(r.ci_high)}] | {r.p_boot} | {r.p_holm} |")
    lines += ["", "## Settings chosen by the inner CV (per outer fold)", "",
              "| language | judge | PCA | L2 | test log-loss |", "|---|---|---|---|---|"]
    for (lang, judge), g in choices.groupby(["language", "judge"], sort=False):
        lines.append(f"| {lang} | {judge} | {', '.join(map(str, g.pca))} | {', '.join(f'{v:g}' for v in g.l2)} | "
                     f"{fmt(g.test_log_loss.mean())} |")
    lines += ["", "## Tamil confusion: primary LiteJudge vs human_majority", "",
              "| human \\ judge | A | B | tie | total |", "|---|---|---|---|---|"]
    for idx, r in confusion.iterrows():
        lines.append(f"| {idx} | {r['A']} | {r['B']} | {r['tie']} | {r['total']} |")
    lines += ["", f"Swap check: max |P(order A/B) − mapped P(order B/A)| = {timing['swap_max']:.1e} (order-invariant). "
              f"Primary CPU time: fit {timing['fit']:.2f} s, predict {timing['predict_ms_per_1000']:.2f} ms per 1,000 pairs "
              "(embedding time is in data/embeddings/MANIFEST.json).", ""]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# EXPLORATORY (not pre-registered, added 10 Nov 2026 after the main results were seen):
# leave-one-model-out. Does LiteJudge still work on a response model it has never seen?
# ---------------------------------------------------------------------------------------------
EXPLORATORY = OUT / "exploratory"
LOMO_JUDGES = ("litejudge_lomo", "model_id_lomo")


def model_id_unseen(features: dict, train: pd.DataFrame, test: pd.DataFrame, held_out: str,
                    l2: float) -> np.ndarray:
    """Model-identity Davidson model when the test model is unseen.

    It has no training pairs, so it has no learned strength. We give it the average strength of
    the models it was trained on (the user-specified rule), then score the test pairs as usual.
    """
    x = features["one_hot"]
    w, tau = fit_davidson(pair_delta(x, train), targets_of(train), l2)
    names = sorted(set(features["model"]))
    k = names.index(held_out)
    others = [i for i in range(len(names)) if i != k]
    w[k] = w[others].mean()
    return davidson_probs(pair_delta(x, test) @ w, tau)


def run_lomo(language: str, pairs: pd.DataFrame, features: dict) -> tuple[pd.DataFrame, list[dict]]:
    """For each response model: train on pairs without it, predict the pairs that involve it.

    A pair involves two models, so it is predicted twice (once per held-out model); the pooled
    table therefore holds every pair exactly twice, with `held_out` saying which model was unseen.
    Note: training pairs share prompts with the test pairs (this split is by model, not prompt).
    """
    models = sorted(set(pairs.model_a) | set(pairs.model_b))
    parts, choices = [], []
    for m in models:
        involved = (pairs.model_a == m) | (pairs.model_b == m)
        train, test = pairs[~involved], pairs[involved]
        # LiteJudge: the pre-registered recipe (inner grouped CV for PCA and L2), but the held-out
        # model's responses are also kept out of the scaler and PCA.
        pca_k, l2, inner_loss = tune(PRIMARY, features, train, exclude_model=m)
        p_lite, _ = fit_and_predict(PRIMARY, features, train, test, pca_k, l2, exclude_model=m)
        # Model identity: tune L2 the same way, then give the unseen model the average strength.
        _, l2_id, _ = tune("model_id", features, train)
        p_id = model_id_unseen(features, train, test, m, l2_id)
        part = test.copy()
        part["held_out"] = m
        part["litejudge_lomo"] = to_verdict(p_lite)
        part["model_id_lomo"] = to_verdict(p_id)
        part[["litejudge_lomo_p_A", "litejudge_lomo_p_B", "litejudge_lomo_p_tie"]] = p_lite
        parts.append(part)
        choices.append({"language": language, "held_out": m, "n_train": len(train), "n_test": len(test),
                        "pca": "none" if pca_k is None else pca_k, "l2": l2, "l2_model_id": l2_id,
                        "inner_log_loss": round(inner_loss, 4),
                        "max_p_tie": round(float(p_lite[:, 2].max()), 4)})
        print(f"  [exploratory] {language} held out {m}: pca={choices[-1]['pca']} l2={l2:g}", flush=True)
    return pd.concat(parts, ignore_index=True), choices


def evaluate_lomo(language: str, lomo: pd.DataFrame, main_preds: pd.DataFrame,
                  rng: np.random.Generator) -> dict:
    """Step 04 metrics on the pooled leave-one-model-out predictions, next to the main-CV verdicts.

    The main-CV verdicts of the same (duplicated) pairs are added as a reference, so all four
    columns are scored on exactly the same rows.
    """
    df = lomo.merge(main_preds[["pairing_id", "litejudge", "model_id"]], on="pairing_id", how="left")
    judges = {"litejudge_lomo": "lomo", "model_id_lomo": "lomo", "litejudge": "main_cv", "model_id": "main_cv"}
    agreement, ceiling, dist, boot, per_model = [], [], [], [], []
    for judge, scheme in judges.items():
        agreement.append(s04.agreement_row(df, language, judge, scheme, judge))
        ceiling.append(s04.human_ceiling_row(df, language, judge, judge))
        dist.extend(s04.verdict_rows(df, language, judge, judge, view=scheme))
        for metric, arr in s04.bootstrap_metrics(df, judge, judge, rng).items():
            if np.all(np.isnan(arr)):
                continue
            lo, hi = s04.ci95(arr)
            boot.append({"language": language, "judge": judge, "scheme": scheme, "metric": metric,
                         "estimate": round(float(np.nanmean(arr)), 4),
                         "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
        for m, sub in df.groupby("held_out"):
            per_model.append(s04.agreement_row(sub, language, judge, scheme, judge) | {"held_out": m})
    # Unadjusted paired contrasts (exploratory, so no Holm family).
    contrast = []
    for a, b in (("litejudge_lomo", "model_id_lomo"), ("litejudge_lomo", "litejudge")):
        diffs = s04.bootstrap_delta_same_lang(df, a, b, rng)
        lo, hi = s04.ci95(diffs)
        contrast.append({"contrast": f"kappa {language}: {a} − {b}", "estimate": round(float(np.nanmean(diffs)), 4),
                         "ci_low": round(lo, 4), "ci_high": round(hi, 4), "p_boot": round(s04.two_sided_p(diffs), 4)})
    return {"frame": df, "agreement": agreement, "ceiling": ceiling, "dist": dist, "boot": boot,
            "per_model": per_model, "contrast": contrast}


def write_lomo(results: dict, choices: list[dict], predictions: list[pd.DataFrame]) -> None:
    """Save the exploratory outputs (kept apart from the pre-registered files) and a short summary."""
    EXPLORATORY.mkdir(parents=True, exist_ok=True)
    tables = {k: pd.DataFrame(v) for k, v in results.items()}
    for name, frame in tables.items():
        frame.to_csv(EXPLORATORY / f"lomo_{name}.csv", index=False)
    pd.DataFrame(choices).to_csv(EXPLORATORY / "lomo_cv_choices.csv", index=False)
    pred = pd.concat(predictions, ignore_index=True)
    pred[["language", "pairing_id", "prompt_id", "held_out", *LOMO_JUDGES,
          "litejudge_lomo_p_A", "litejudge_lomo_p_B", "litejudge_lomo_p_tie"]].to_csv(
        EXPLORATORY / "lomo_predictions.csv", index=False)

    agr, ceil, boot, dist = tables["agreement"], tables["ceiling"], tables["boot"], tables["dist"]
    kci = boot[boot.metric == "kappa"].set_index(["language", "judge"])
    ties = dist[dist.label == "tie"].set_index(["language", "judge"])["pct"]
    ch = pd.DataFrame(choices)
    lines = ["# EXPLORATORY: leave-one-model-out check (not pre-registered)", "",
             "Added on 10 Nov 2026 after the pre-registered results were seen. For each response model, "
             "LiteJudge (primary recipe) and the model-identity baseline are trained on pairs that do not "
             "involve it and tested on the pairs that do. Every pair is tested twice (once per model), so "
             "rows = 2 x pairs. The model-identity baseline gives the unseen model the average strength. "
             "`main_cv` = the pre-registered out-of-fold verdicts scored on the same rows. "
             "Training pairs share prompts with test pairs. Contrasts are unadjusted.", "",
             "| language | judge | scheme | rows | κ vs majority | 95% CI | κ vs raters | gap to ceiling | % tie |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in agr.itertuples():
        c = kci.loc[(r.language, r.judge)]
        g = ceil[(ceil.language == r.language) & (ceil.judge == r.judge)].iloc[0]
        lines.append(f"| {r.language} | {r.judge} | {r.view} | {r.n} | {r.kappa:.3f} | "
                     f"[{c.ci_low:.3f}, {c.ci_high:.3f}] | {g.judge_human_kappa_mean:.3f} | {g.gap:.3f} | "
                     f"{ties.loc[(r.language, r.judge)]:.1f} |")
    lines += ["", "| contrast | estimate | 95% CI | p (unadjusted) |", "|---|---|---|---|"]
    for r in tables["contrast"].itertuples():
        lines.append(f"| {r.contrast} | {r.estimate:.3f} | [{r.ci_low:.3f}, {r.ci_high:.3f}] | {r.p_boot} |")
    lines += ["", "Max LiteJudge P(tie) over all held-out predictions: " + ", ".join(
        f"{lang} {g.max_p_tie.max():.3f}" for lang, g in ch.groupby("language", sort=False)) + ".", ""]
    (EXPLORATORY / "summary.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)            # bootstrap CIs
    rng_contrast = np.random.default_rng(SEED + 1)  # separate stream for the paired contrasts

    all_preds, all_choices, timings = [], [], {}
    res = {k: [] for k in ("agreement", "ceiling", "dist", "length", "boot", "tau", "lb")}
    contrast_rows, frames, data = [], {}, {}
    for language in LANGUAGES:
        print(f"{language}: loading pairs and embeddings", flush=True)
        pairs, features = load_language(language)
        data[language] = (pairs, features)  # kept for the exploratory check at the end
        preds, choices, timing = run_cv(language, pairs, features)
        evaluated = evaluate(language, pairs, preds, rng)
        frames[language] = evaluated["frame"]
        for k in res:
            res[k].extend(evaluated[k])
        contrast_rows.extend(contrasts(language, evaluated["frame"], rng_contrast))
        all_preds.append(preds.assign(language=language))
        all_choices.extend(choices)
        timings[language] = timing

    agreement = pd.DataFrame(res["agreement"])
    ceiling = pd.DataFrame(res["ceiling"])
    dist = pd.DataFrame(res["dist"])
    length = pd.DataFrame(res["length"])
    boot = pd.DataFrame(res["boot"])
    tau = pd.concat(res["tau"], ignore_index=True)
    lb = pd.concat(res["lb"], ignore_index=True)
    contrast_df = pd.DataFrame(contrast_rows)
    contrast_df["p_holm"] = np.round(s04.holm(list(contrast_df.p_boot)), 4)
    choices = pd.DataFrame(all_choices)
    table = comparison_table(agreement, ceiling, dist, boot, length, tau)
    confusion = s04.confusion_frame(frames["tamil"], PRIMARY)

    preds = pd.concat(all_preds, ignore_index=True)
    preds = preds[["language", "pairing_id", "prompt_id", "fold", *ALL_NEW,
                   *[c for c in preds.columns if "_p_" in c]]]
    preds.to_csv(OUT / "oof_predictions.csv", index=False)
    agreement.to_csv(OUT / "agreement.csv", index=False)
    ceiling.to_csv(OUT / "human_ceiling.csv", index=False)
    dist.to_csv(OUT / "verdict_distribution.csv", index=False)
    length.to_csv(OUT / "length_bias.csv", index=False)
    boot.to_csv(OUT / "bootstrap_ci.csv", index=False)
    tau.to_csv(OUT / "leaderboard_tau.csv", index=False)
    lb.to_csv(OUT / "leaderboard.csv", index=False)
    contrast_df.to_csv(OUT / "bootstrap_contrasts.csv", index=False)
    choices.to_csv(OUT / "cv_choices.csv", index=False)
    table.to_csv(OUT / "comparison.csv", index=False)
    confusion.to_csv(OUT / "confusion_tamil_litejudge.csv")

    # Run record: CPU time of the primary model (used for cost in step 07) and the swap check.
    fit = sum(t["fit_seconds"] for t in timings.values())
    pred_s = sum(t["predict_seconds"] for t in timings.values())
    n_pred = sum(t["n_predicted"] for t in timings.values())
    run = {"created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "seed": SEED,
           "outer_folds": OUTER_FOLDS, "inner_folds": INNER_FOLDS, "pca_grid": [str(k) for k in PCA_GRID],
           "l2_grid": list(L2_GRID), "primary": PRIMARY, "per_language": timings,
           "embeddings_manifest": json.loads((EMB / "MANIFEST.json").read_text())}
    (OUT / "run_info.json").write_text(json.dumps(run, indent=2, ensure_ascii=False, default=float) + "\n")
    write_summary(table, contrast_df, choices, confusion,
                  {"swap_max": max(t["swap_check_max_abs_diff"] for t in timings.values()),
                   "fit": fit, "predict_ms_per_1000": 1e6 * pred_s / n_pred})
    print(f"Wrote {len(list(OUT.glob('*')))} files under {OUT.relative_to(ROOT).as_posix()}/")

    # EXPLORATORY part, run only after every pre-registered file is written. It uses its own random
    # stream, so it cannot change any pre-registered number.
    rng_lomo = np.random.default_rng(SEED + 2)
    lomo_res = {k: [] for k in ("agreement", "ceiling", "dist", "boot", "per_model", "contrast")}
    lomo_choices, lomo_preds = [], []
    for language in LANGUAGES:
        lomo, chosen = run_lomo(language, data[language][0], data[language][1])
        main_preds = preds[preds.language == language]
        evaluated = evaluate_lomo(language, lomo, main_preds, rng_lomo)
        for k in lomo_res:
            lomo_res[k].extend(evaluated[k])
        lomo_choices.extend(chosen)
        lomo_preds.append(lomo.assign(language=language))
    write_lomo(lomo_res, lomo_choices, lomo_preds)
    print(f"[exploratory] wrote {EXPLORATORY.relative_to(ROOT).as_posix()}/")


if __name__ == "__main__":
    main()
