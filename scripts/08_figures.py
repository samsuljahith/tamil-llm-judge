"""Step 08: every figure in the report, as vector PDF + 300-dpi PNG, plus draft captions.

Figures (results/figures/):
  fig_a_kappa_ceiling     judge-vs-rater kappa with 95% CIs against the human ceiling (Tamil, Hindi)
  fig_b_tie_rate          share of tie verdicts per judge, with the human tie rate as reference
  fig_c_bias              position (flip rate), length and self-preference bias with 95% CIs
  fig_d_kendall_tau       leaderboard agreement (Kendall's tau vs the human ranking)
  fig_e_cost_kappa        US$ per 1,000 pairs (log scale) vs kappa; $0 judges in a separate strip
  fig_f_confusion_tamil   Tamil confusion matrix of the best single judge vs human_majority
  fig_g_lomo              EXPLORATORY leave-one-model-out: LiteJudge vs model identity

Which kappa is plotted next to the human ceiling
  The human ceiling (step 04) is the mean pairwise Cohen's kappa between the three raters. The
  matching judge number is the mean Cohen's kappa of the judge against each individual rater
  (step 04's `judge_human_kappa_mean`), NOT kappa against the majority label, which is on a
  different scale. So every figure that draws the ceiling plots judge-vs-rater kappa. Step 04
  has point estimates only for it, so this script adds 95% CIs with step 04's own
  prompt-clustered bootstrap (2,000 replicates, seed 42; judges and ceiling share the same
  resamples), and checks its point estimates against step 04's human_ceiling.csv.
  It also adds a bootstrap CI for LiteJudge's length bias (step 04's bootstrap_bias function).
  These numbers are saved in results/figures/figure_stats.csv.

Design rules: colourblind-safe categorical palette (validated with the dataviz validator:
adjacent-pair CVD dE >= 9.1), one fixed colour and row order per judge in every figure, judges
always named on an axis or by a direct label (colour is never the only cue), no titles inside
images (captions are in captions.md), axis labels with units, sized for A4 half-width (85 mm).

Usage:  python scripts/08_figures.py
Reads:  results/agreement/, results/litejudge/, results/ensembles/, data/processed/
Writes: results/figures/*.pdf, *.png, figure_stats.csv, captions.md
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files only (no window), the same on every machine
import matplotlib.pyplot as plt  # noqa: E402  (must come after choosing the backend)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
AGR = ROOT / "results" / "agreement"
LITE = ROOT / "results" / "litejudge"
ENS = ROOT / "results" / "ensembles"
OUT = ROOT / "results" / "figures"

_spec = importlib.util.spec_from_file_location("step04", ROOT / "scripts" / "04_agreement_bias.py")
s04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s04)

SEED = 42
LANGUAGES = ("tamil", "hindi")
LANG_NAME = {"tamil": "Tamil", "hindi": "Hindi"}

# ---- One order, one name and one colour per judge, used in every figure ----------------------
JUDGES = ("frontier", "low_cost", "sea_lion", "base_model", "gpt4_32k",
          "litejudge", "litejudge_length", "model_id", "longer", "majority")
NAME = {"frontier": "Frontier", "low_cost": "Low-cost", "sea_lion": "SEA-LION",
        "base_model": "Gemma base", "gpt4_32k": "GPT-4-32K", "litejudge": "LiteJudge",
        "litejudge_length": "LiteJudge + length", "model_id": "Model identity",
        "longer": "Always longer", "majority": "Majority class"}
# Categorical slots 1-6 of the validated reference palette, in its fixed order.
COLOUR = {"frontier": "#2a78d6", "low_cost": "#eb6834", "sea_lion": "#1baf7a",
          "base_model": "#eda100", "gpt4_32k": "#e87ba4", "litejudge": "#008300",
          "litejudge_length": "#008300",  # same entity as LiteJudge; told apart by a hollow marker
          "model_id": "#52514e", "longer": "#898781", "majority": "#898781"}  # baselines in grey
HOLLOW = {"litejudge_length", "longer"}       # drawn with a white centre
MARKER = {j: "o" for j in JUDGES} | {"model_id": "s", "longer": "D", "majority": "D", "litejudge_length": "o"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
ENSEMBLE_COLOUR = INK2  # ensembles (cost figure only): dark grey diamonds, always direct-labelled

# Verdict column for each judge in the merged per-pair table.
COLUMN = {**{j: f"{j}_swap" for j in s04.LLM_JUDGES}, "gpt4_32k": "gpt4_32k_ab",
          **{j: j for j in ("litejudge", "litejudge_length", "model_id", "longer", "majority")}}

WIDTH = 3.35  # inches = 85 mm, half an A4 text width
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 0, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,   # embed fonts as TrueType so text stays text (searchable, editable)
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


# ---------------------------------------------------------------------------------------------
# Data for the figures
# ---------------------------------------------------------------------------------------------
def pair_table(language: str) -> pd.DataFrame:
    """Unique pairs with every judge's verdict: LLMs via step 04, new judges from step 06."""
    df = s04.attach_all_judges(s04.load_pairs(language), language)
    oof = pd.read_csv(LITE / "oof_predictions.csv")
    cols = ["litejudge", "litejudge_length", "model_id", "longer", "majority"]
    return df.merge(oof.loc[oof.language == language, ["pairing_id", *cols]], on="pairing_id", how="left")


def codes(values: np.ndarray) -> np.ndarray:
    """A/B/tie -> 0/1/2; anything else (missing verdict) -> -1."""
    out = np.full(len(values), -1, dtype=np.int8)
    for k, lab in enumerate(s04.LABELS):
        out[values == lab] = k
    return out


def kappa_codes(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa on rows where both codes are valid (the same formula as s04.kappa_np)."""
    ok = (a >= 0) & (b >= 0)
    n = int(ok.sum())
    if n == 0:
        return float("nan")
    mat = np.zeros((3, 3))
    np.add.at(mat, (a[ok], b[ok]), 1)
    po = mat.trace() / n
    pe = (mat.sum(axis=1) * mat.sum(axis=0)).sum() / (n * n)
    return float((po - pe) / (1 - pe)) if pe < 1 else (1.0 if po >= 1 else 0.0)


def rater_kappas(df: pd.DataFrame, columns: dict[str, str], rng: np.random.Generator,
                 language: str, scheme: str) -> list[dict]:
    """Mean kappa of each judge against the three raters, and the human-human ceiling, with CIs.

    Point estimates use all rows; CIs come from prompt-clustered resamples shared by every judge
    and the ceiling, so judge and ceiling intervals are directly comparable.
    """
    raters = [codes(df[f"human_{k}"].to_numpy()) for k in (1, 2, 3)]
    judges = {name: codes(df[col].to_numpy()) for name, col in columns.items()}

    def ceiling(pos: np.ndarray) -> float:
        r = [x[pos] for x in raters]
        return float(np.mean([kappa_codes(r[0], r[1]), kappa_codes(r[0], r[2]), kappa_codes(r[1], r[2])]))

    def vs_raters(j: np.ndarray, pos: np.ndarray) -> float:
        return float(np.mean([kappa_codes(j[pos], r[pos]) for r in raters]))

    everything = np.arange(len(df))
    point = {"human_ceiling": ceiling(everything)} | {n: vs_raters(j, everything) for n, j in judges.items()}
    _, clusters = s04.prompt_clusters(df)
    draws = {name: np.empty(s04.N_BOOT) for name in point}
    for b in range(s04.N_BOOT):
        pos = s04.resample_positions(clusters, rng)
        draws["human_ceiling"][b] = ceiling(pos)
        for name, j in judges.items():
            draws[name][b] = vs_raters(j, pos)
    rows = []
    for name, est in point.items():
        lo, hi = s04.ci95(draws[name])
        rows.append({"language": language, "scheme": scheme, "judge": name, "metric": "kappa_vs_raters",
                     "estimate": round(est, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
    return rows


def check_against_step04(stats: pd.DataFrame) -> None:
    """Our point estimates must equal step 04's / step 06's human_ceiling tables (4 decimals)."""
    ref = pd.concat([pd.read_csv(AGR / "human_ceiling.csv"), pd.read_csv(LITE / "human_ceiling.csv")])
    # Only judges that step 04/06 report (the extra A/B-only rows for the cost plot have no reference).
    main = stats[(stats.scheme == "main") & (stats.metric == "kappa_vs_raters") & stats.judge.isin(JUDGES)]
    merged = main.merge(ref, on=["language", "judge"])
    worst = (merged.estimate - merged.judge_human_kappa_mean).abs().max()
    ceil = stats[(stats.scheme == "main") & (stats.judge == "human_ceiling")].set_index("language").estimate
    ceil_ref = ref.groupby("language").human_human_kappa_mean.first()
    worst = max(worst, (ceil - ceil_ref).abs().max())
    if len(merged) != len(main) or worst > 1e-4:
        raise SystemExit(f"Rater-kappa check failed (max difference {worst:.1e}); figures would not match step 04.")
    print(f"  rater kappas match step 04/06 tables (max difference {worst:.1e})")


def compute_stats(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Every number the figures need that is not already in a step 04-07 CSV."""
    rng = np.random.default_rng(SEED)
    rows = []
    for lang, df in frames.items():
        cols = dict(COLUMN) | {f"{j}_ab": f"{j}_ab" for j in s04.LLM_JUDGES}  # A/B-only views for the cost plot
        rows += rater_kappas(df, cols, rng, lang, "main")
        # Length bias CI for the two LiteJudge variants (step 04's own bootstrap; no family needed).
        for j in ("litejudge", "litejudge_length"):
            arr = s04.bootstrap_bias(df, j, set(), rng)["length_minus_human"]
            lo, hi = s04.ci95(arr)
            rows.append({"language": lang, "scheme": "main", "judge": j, "metric": "length_minus_human",
                         "estimate": round(float(np.nanmean(arr)), 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
    # Exploratory leave-one-model-out rows (every pair appears twice, once per held-out model).
    lomo = pd.read_csv(LITE / "exploratory" / "lomo_predictions.csv")
    oof = pd.read_csv(LITE / "oof_predictions.csv")
    for lang, df in frames.items():
        sub = lomo[lomo.language == lang].merge(
            oof.loc[oof.language == lang, ["pairing_id", "litejudge", "model_id"]], on="pairing_id")
        sub = sub.merge(df[["pairing_id", "human_1", "human_2", "human_3"]], on="pairing_id")
        rows += rater_kappas(sub, {"litejudge_lomo": "litejudge_lomo", "model_id_lomo": "model_id_lomo",
                                   "litejudge": "litejudge", "model_id": "model_id"}, rng, lang, "lomo")
    stats = pd.DataFrame(rows)
    check_against_step04(stats)
    return stats


# ---------------------------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------------------------
def style_x_grid(ax) -> None:
    """Solid hairline vertical gridlines behind the data; no y grid for dot plots."""
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def dot(ax, x, y, judge: str, size: float = 4.5, lo=None, hi=None, hollow: bool | None = None,
        marker: str | None = None) -> None:
    """One judge's point with an optional 95% CI whisker, in its fixed colour and marker."""
    colour = COLOUR[judge]
    hollow = judge in HOLLOW if hollow is None else hollow
    if lo is not None and hi is not None and np.isfinite(lo):
        ax.plot([lo, hi], [y, y], color=colour, linewidth=1.2, solid_capstyle="round", zorder=2)
    ax.plot([x], [y], marker=marker or MARKER[judge], markersize=size, color=colour,
            markerfacecolor="white" if hollow else colour, markeredgewidth=1.1, linestyle="none", zorder=3)


def judge_axis(ax, judges: list[str], labels: bool = True) -> dict[str, int]:
    """Judges as rows from top to bottom; returns judge -> y position.

    labels=False is for the 2nd+ panel of a figure whose panels do NOT share the y-axis.
    With a shared y-axis, call this with labels=True on the first panel only (blanking the labels
    on a shared axis would blank them on every panel).
    """
    ypos = {j: len(judges) - 1 - i for i, j in enumerate(judges)}
    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels([NAME.get(j, j) for j in ypos] if labels else [])
    ax.set_ylim(-0.6, len(judges) - 0.4)
    ax.tick_params(axis="y", length=0)
    return ypos


def ceiling_line(ax, est: float, lo: float, hi: float, vertical: bool = True, label: bool = True) -> None:
    """The human ceiling: a black reference line with its 95% CI as a light band."""
    if vertical:
        ax.axvspan(lo, hi, color=INK, alpha=0.07, linewidth=0, zorder=0)
        ax.axvline(est, color=INK, linewidth=0.9, zorder=1)
        if label:
            ax.text(est, 0.01, " human\n ceiling", transform=ax.get_xaxis_transform(), ha="left", va="bottom",
                    fontsize=6, color=INK)
    else:
        ax.axhspan(lo, hi, color=INK, alpha=0.07, linewidth=0, zorder=0)
        ax.axhline(est, color=INK, linewidth=0.9, zorder=1)
        if label:
            ax.text(1.0, est, "human ceiling ", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                    fontsize=6, color=INK)


LANG_MARKER = {"tamil": "o", "hindi": "s"}  # language is shown by marker shape, never by fill


def language_legend(ax, loc="lower right") -> None:
    """Legend for the language encoding used when both languages share one panel."""
    handles = [Line2D([], [], marker=LANG_MARKER[lang], color=MUTED, linestyle="none", markersize=4.5,
                      label=LANG_NAME[lang]) for lang in LANGUAGES]
    ax.legend(handles=handles, loc=loc, frameon=False, handletextpad=0.2, borderaxespad=0.2)


def save(fig, name: str) -> list[str]:
    paths = []
    for ext in ("pdf", "png"):
        path = OUT / f"{name}.{ext}"
        # Fixed metadata so re-running gives byte-identical PDFs (no creation date stamped in).
        meta = {"CreationDate": None} if ext == "pdf" else {"Software": None}
        fig.savefig(path, metadata=meta)
        paths.append(path.relative_to(ROOT).as_posix())
    plt.close(fig)
    return paths


def stat(stats: pd.DataFrame, lang: str, judge: str, metric: str = "kappa_vs_raters", scheme: str = "main"):
    row = stats[(stats.language == lang) & (stats.judge == judge) & (stats.metric == metric) & (stats.scheme == scheme)]
    return row.iloc[0] if len(row) else None


# ---------------------------------------------------------------------------------------------
# The figures
# ---------------------------------------------------------------------------------------------
def fig_a(stats: pd.DataFrame) -> list[str]:
    """Judge-vs-rater kappa with CIs, one panel per language, human ceiling as reference."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.6), sharex=True, gridspec_kw={"wspace": 0.08})
    for i, (ax, lang) in enumerate(zip(axes, LANGUAGES)):
        ypos = judge_axis(ax, list(JUDGES), labels=i == 0)
        c = stat(stats, lang, "human_ceiling")
        ceiling_line(ax, c.estimate, c.ci_low, c.ci_high, label=i == 0)
        for j in JUDGES:
            r = stat(stats, lang, j)
            dot(ax, r.estimate, ypos[j], j, lo=r.ci_low, hi=r.ci_high)
        style_x_grid(ax)
        ax.set_xlim(-0.05, 0.6)
        ax.set_xlabel(f"{LANG_NAME[lang]}: Cohen's κ vs raters", labelpad=2)
    return save(fig, "fig_a_kappa_ceiling")


def fig_b() -> list[str]:
    """Share of tie verdicts per judge, human tie rate as the reference line."""
    d4 = pd.read_csv(AGR / "verdict_distribution.csv")
    d6 = pd.read_csv(LITE / "verdict_distribution.csv")
    dist = pd.concat([d4[(d4.view == "swap_agg") | (d4.judge.isin(["gpt4_32k", "human_majority"]))], d6])
    ties = dist[dist.label == "tie"].set_index(["language", "judge"]).pct
    judges = [j for j in JUDGES if j not in ("longer", "majority")]  # those two never say tie by design
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.1), sharex=True, gridspec_kw={"wspace": 0.08})
    for i, (ax, lang) in enumerate(zip(axes, LANGUAGES)):
        ypos = judge_axis(ax, judges, labels=i == 0)
        for j in judges:
            v = ties.loc[(lang, j)]
            hollow = j in HOLLOW  # the ablation: outlined bar, like its hollow marker elsewhere
            ax.barh(ypos[j], v, height=0.55, facecolor="white" if hollow else COLOUR[j],
                    edgecolor=COLOUR[j], linewidth=1.0 if hollow else 0, zorder=2)
        human = ties.loc[(lang, "human_majority")]
        ax.axvline(human, color=INK, linewidth=0.9, zorder=3)
        xmax = 35
        right_half = human > xmax / 2  # put the label on the side of the line with room for it
        ax.text(human, 0.01, "humans " if right_half else " humans", transform=ax.get_xaxis_transform(),
                ha="right" if right_half else "left", va="bottom", fontsize=6, color=INK)
        style_x_grid(ax)
        ax.set_xlim(0, xmax)
        ax.set_xlabel(f"{LANG_NAME[lang]}: tie verdicts (%)", labelpad=2)
    return save(fig, "fig_b_tie_rate")


def fig_c(stats: pd.DataFrame) -> list[str]:
    """Position (flip rate), length and self-preference bias with 95% CIs, both languages."""
    boot = pd.read_csv(AGR / "bootstrap_ci.csv")
    bias = pd.read_csv(AGR / "bootstrap_bias.csv")
    judges = ["frontier", "low_cost", "sea_lion", "base_model", "gpt4_32k", "litejudge", "litejudge_length"]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.3), sharey=True, gridspec_kw={"wspace": 0.2})
    panels = [("flip_rate", "Flip rate when\norder swaps (%)"),
              ("length_minus_human", "Longer answer:\njudge − human (pp)"),
              ("selfpref_minus_human", "Own family:\njudge − human (pp)")]
    ypos = judge_axis(axes[0], judges)  # shared y-axis: rows and names set once
    for k, (ax, (metric, xlabel)) in enumerate(zip(axes, panels)):
        if metric != "flip_rate":
            ax.axvline(0, color=AXIS, linewidth=0.8, zorder=1)  # "no different from humans"
        for lang, offset in (("tamil", 0.17), ("hindi", -0.17)):
            for j in judges:
                if metric == "flip_rate":
                    if j.startswith("litejudge"):  # order-invariant by construction: exactly 0
                        dot(ax, 0, ypos[j] + offset, j, marker=LANG_MARKER[lang])
                        continue
                    r = boot[(boot.language == lang) & (boot.judge == j) & (boot.metric == metric)]
                elif j.startswith("litejudge"):
                    r = stats[(stats.language == lang) & (stats.judge == j) & (stats.metric == metric)]
                else:
                    r = bias[(bias.language == lang) & (bias.judge == j) & (bias.metric == metric)]
                if r.empty:
                    continue  # no flip rate for GPT-4-32K (one order); no model family for LiteJudge
                r = r.iloc[0]
                dot(ax, 100 * r.estimate, ypos[j] + offset, j, size=3.8, lo=100 * r.ci_low, hi=100 * r.ci_high,
                    marker=LANG_MARKER[lang])
        style_x_grid(ax)
        ax.set_xlabel(xlabel, labelpad=2)
    axes[0].set_xlim(-1, 20)
    language_legend(axes[2], loc="lower left")
    return save(fig, "fig_c_bias")


def fig_d() -> list[str]:
    """Kendall's tau between each judge's model ranking and the human ranking."""
    tau = pd.concat([pd.read_csv(AGR / "leaderboard_tau.csv"), pd.read_csv(LITE / "leaderboard_tau.csv")])
    judges = [j for j in JUDGES if j != "majority"]  # a constant verdict ranks nobody
    fig, ax = plt.subplots(figsize=(WIDTH, 2.2))
    ypos = judge_axis(ax, judges)
    for lang, offset in (("tamil", 0.15), ("hindi", -0.15)):
        for j in judges:
            v = tau[(tau.language == lang) & (tau.judge == j)].kendall_tau.iloc[0]
            dot(ax, v, ypos[j] + offset, j, size=4, marker=LANG_MARKER[lang])
    style_x_grid(ax)
    ax.set_xlim(-0.05, 1.0)
    ax.set_xlabel("Kendall's τ: judge's model ranking vs human ranking", labelpad=2)
    language_legend(ax, loc="upper left")
    return save(fig, "fig_d_kendall_tau")


def place_labels(ax, points: list[tuple[float, float, str]], log_x: bool) -> None:
    """Direct labels that do not collide: points at (nearly) the same x form a group; within a
    group the highest label goes above its point, the lowest below, any middle one level with it.

    points: (x, kappa, text). "Nearly the same x" = within 0.15 decades on the log axis, or the
    same x in the $0 strip.
    """
    def key(x):
        return np.log10(x) if log_x else x
    remaining = sorted(points, key=lambda p: key(p[0]))
    groups: list[list[tuple[float, float, str]]] = []
    for p in remaining:
        if groups and abs(key(p[0]) - key(groups[-1][0][0])) < (0.15 if log_x else 1e-9):
            groups[-1].append(p)
        else:
            groups.append([p])
    for group in groups:
        group.sort(key=lambda p: p[1], reverse=True)  # highest kappa first
        for rank, (x, k, text) in enumerate(group):
            if len(group) == 1:
                dy, va = 0, "center"
            elif rank == 0:
                dy, va = 3, "bottom"
            elif rank == len(group) - 1:
                dy, va = -3, "top"
            else:
                dy, va = 0, "center"
            if log_x:
                ax.annotate(text, (x, k), xytext=(5, dy), textcoords="offset points", fontsize=6,
                            color=INK2, ha="left", va=va)
            else:  # $0 strip: centred over / under the point, there is no room to the side
                ax.annotate(text, (x, k), xytext=(0, 4 if dy >= 0 else -4), textcoords="offset points",
                            fontsize=6, color=INK2, ha="center", va="bottom" if dy >= 0 else "top")


def fig_e(stats: pd.DataFrame) -> list[str]:
    """Cost per 1,000 pairs (log) vs judge-vs-rater kappa; $0 judges in a strip on the left."""
    cost = pd.read_csv(ENS / "cost_agreement.csv")
    ens_ceiling = pd.read_csv(ENS / "ensemble_human_ceiling.csv")
    ens_names = {"llm4": "4 LLMs", "cheap3": "Cheap 3", "llm4_lite": "4 LLMs + Lite", "free_lite": "Free + Lite"}
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 3.9), sharey="row",
                             gridspec_kw={"width_ratios": [1.5, 5], "wspace": 0.06, "hspace": 0.45})
    for row, lang in enumerate(LANGUAGES):
        zero_ax, log_ax = axes[row]
        c = stat(stats, lang, "human_ceiling")
        for ax in (zero_ax, log_ax):
            ceiling_line(ax, c.estimate, c.ci_low, c.ci_high, vertical=False, label=False)
            ax.grid(axis="y", color=GRID, linewidth=0.5)
            ax.set_axisbelow(True)
        zero_labels, log_labels = [], []
        for r in cost[cost.language == lang].itertuples():
            if r.kind == "single":
                single_order = r.view == "ab"
                k = stat(stats, lang, f"{r.judge}_ab" if single_order else r.judge).estimate
                free = r.usd_per_1000_pairs == 0
                ax = zero_ax if free else log_ax
                # In the $0 strip the A/B-only point sits a little to the right of the both-orders point.
                x = (0.6 if single_order else 0.0) if free else r.usd_per_1000_pairs
                dot(ax, x, k, r.judge, size=4 if single_order else 5, hollow=single_order)
                if not single_order:  # name each judge once, at its main (both-orders) point
                    (zero_labels if free else log_labels).append((x, k, NAME[r.judge]))
            else:
                k = ens_ceiling[(ens_ceiling.language == lang)
                                & (ens_ceiling.judge == r.judge)].judge_human_kappa_mean.iloc[0]
                log_ax.plot([r.usd_per_1000_pairs], [k], marker="D", markersize=4, color=ENSEMBLE_COLOUR,
                            linestyle="none", zorder=3)
                log_labels.append((r.usd_per_1000_pairs, k, ens_names[r.judge]))
        place_labels(zero_ax, zero_labels, log_x=False)
        place_labels(log_ax, log_labels, log_x=True)
        log_ax.set_xscale("log")
        log_ax.set_xlim(0.04, 300)  # room on the right for the labels of the most expensive points
        log_ax.set_xticks([0.1, 1, 10, 100])
        log_ax.set_xticklabels(["$0.10", "$1", "$10", "$100"])
        zero_ax.set_xlim(-0.6, 1.0)
        zero_ax.set_xticks([0])
        zero_ax.set_xticklabels(["$0"])
        zero_ax.spines["right"].set_visible(True)
        zero_ax.spines["right"].set_linestyle((0, (2, 2)))  # marks the break between $0 and the log axis
        zero_ax.set_ylabel(f"{LANG_NAME[lang]}: Cohen's κ vs raters")
        log_ax.tick_params(axis="y", length=0)
        log_ax.set_xlabel("API cost per 1,000 pairs (US$, log scale)", labelpad=2)
        zero_ax.set_ylim(0.27, 0.52)
    handles = [Line2D([], [], marker="o", color=MUTED, linestyle="none", markersize=5, label="both orders"),
               Line2D([], [], marker="o", color=MUTED, markerfacecolor="white", markeredgewidth=1.1,
                      linestyle="none", markersize=4, label="A/B order only"),
               Line2D([], [], marker="D", color=ENSEMBLE_COLOUR, linestyle="none", markersize=4, label="ensemble"),
               Line2D([], [], color=INK, linewidth=0.9, label="human ceiling (band: 95% CI)")]
    # Legend below both rows, outside the plots, so it can never cover a point.
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False,
               handletextpad=0.3, columnspacing=1.2)
    return save(fig, "fig_e_cost_kappa")


def fig_f() -> tuple[list[str], str]:
    """Tamil confusion matrix of the best single judge (highest kappa vs human_majority)."""
    agr = pd.read_csv(AGR / "agreement.csv")
    main = agr[(agr.language == "tamil") & ((agr.view == "swap_agg") | (agr.judge == "gpt4_32k"))]
    best = main.sort_values("kappa", ascending=False).judge.iloc[0]
    df = pair_table("tamil")
    cm = s04.confusion_frame(df, COLUMN[best]).loc[list(s04.LABELS), list(s04.LABELS)]
    share = cm.div(cm.sum(axis=1), axis=0) * 100  # row % (of each human label)
    fig, ax = plt.subplots(figsize=(WIDTH * 0.62, WIDTH * 0.55))
    from matplotlib.colors import LinearSegmentedColormap
    blues = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#86b6ef", "#2a78d6", "#104281"])
    image = ax.imshow(share.to_numpy(), cmap=blues, vmin=0, vmax=100)
    for i in range(3):
        for k in range(3):
            v = share.iloc[i, k]
            ax.text(k, i, f"{cm.iloc[i, k]}\n{v:.0f}%", ha="center", va="center", fontsize=6.5,
                    color="white" if v > 55 else INK)
    labels = ["A better", "B better", "Tie"]
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(3), labels)
    ax.set_xlabel(f"{NAME[best]} verdict (swap-aggregated)", labelpad=2)
    ax.set_ylabel("Human majority label")
    ax.tick_params(length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.04)
    bar.set_label("Share of the human label (%)", fontsize=6)
    bar.ax.tick_params(labelsize=6, length=2)
    bar.outline.set_visible(False)
    return save(fig, "fig_f_confusion_tamil"), best


def fig_g(stats: pd.DataFrame) -> list[str]:
    """EXPLORATORY leave-one-model-out: LiteJudge vs model identity, main CV vs unseen model."""
    rows = [("litejudge", "main", "LiteJudge"), ("litejudge_lomo", "lomo", "LiteJudge, unseen model"),
            ("model_id", "main", "Model identity"), ("model_id_lomo", "lomo", "Model identity, unseen")]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 1.55), sharex=True, gridspec_kw={"wspace": 0.08})
    for i, (ax, lang) in enumerate(zip(axes, LANGUAGES)):
        ypos = {name: len(rows) - 1 - n for n, (name, _, _) in enumerate(rows)}
        ax.set_yticks(list(ypos.values()))
        ax.set_yticklabels([label for _, _, label in rows] if i == 0 else [])
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.tick_params(axis="y", length=0)
        c = stat(stats, lang, "human_ceiling", scheme="lomo")
        ceiling_line(ax, c.estimate, c.ci_low, c.ci_high, label=i == 0)
        for name, _, _ in rows:
            r = stat(stats, lang, name, scheme="lomo")
            judge = "litejudge" if name.startswith("litejudge") else "model_id"
            dot(ax, r.estimate, ypos[name], judge, lo=r.ci_low, hi=r.ci_high, hollow=name.endswith("_lomo"))
        style_x_grid(ax)
        ax.set_xlim(0.1, 0.55)
        ax.set_xlabel(f"{LANG_NAME[lang]}: Cohen's κ vs raters", labelpad=2)
    axes[1].text(0.98, 0.04, "exploratory", transform=axes[1].transAxes, ha="right", va="bottom",
                 fontsize=6, color=INK2, style="italic")
    return save(fig, "fig_g_lomo")


# ---------------------------------------------------------------------------------------------
# Captions (drafts for the report; the numbers are filled in from the result tables)
# ---------------------------------------------------------------------------------------------
def write_captions(stats: pd.DataFrame, best: str) -> None:
    s = lambda lang, j, scheme="main": stat(stats, lang, j, scheme=scheme)  # noqa: E731
    ta_c, hi_c = s("tamil", "human_ceiling"), s("hindi", "human_ceiling")
    cost = pd.read_csv(ENS / "cost_agreement.csv")
    usd = cost[(cost.view == "swap_agg")].set_index(["language", "judge"]).usd_per_1000_pairs
    names = ("Frontier = GPT-5.5; Low-cost = GPT-5.4 mini; SEA-LION = Gemma-SEA-LION-v4-27B-IT; "
             "Gemma base = Gemma 3 27B IT; GPT-4-32K = PARIKSHA's original verdict (A/B order only)")
    text = f"""# Figure captions (drafts)

All figures: unique pairs only (Tamil 1,820, Hindi 1,000), ties kept. LLM judges are
swap-aggregated (A/B and mapped B/A; disagreement = tie) unless stated. {names}. LiteJudge
results are out-of-fold (5 folds grouped by prompt). 95% CIs are prompt-clustered bootstrap
intervals (2,000 replicates, seed 42).

**Figure (a) - `fig_a_kappa_ceiling`.** Agreement of each judge with the human raters, as the
mean Cohen's κ of the judge against each of the three individual raters, with 95% CIs. The
black line and band show the human ceiling, the mean pairwise κ between the three raters
(Tamil {ta_c.estimate:.2f} [{ta_c.ci_low:.2f}, {ta_c.ci_high:.2f}], Hindi {hi_c.estimate:.2f}
[{hi_c.ci_low:.2f}, {hi_c.ci_high:.2f}]); judges and ceiling use the same bootstrap resamples. Grey
markers are baselines (model identity = a Bradley–Terry model that only knows which model wrote
each answer; always longer; majority class); the hollow green marker is the LiteJudge ablation
with a length feature. κ against the majority label (the main table metric) is reported in
Table X. The figure uses κ against the individual raters, not κ against the majority label used in the
main tables, because only the former is on the same scale as the human ceiling (rater vs rater):
the comparison is like-for-like.

**Figure (b) - `fig_b_tie_rate`.** Share of pairs each judge calls a tie, against the share of
human-majority ties (black line: Tamil 31.7%, Hindi 6.3%). All judges under-call ties in Tamil.
The always-longer and majority-class baselines never output a tie and are omitted.

**Figure (c) - `fig_c_bias`.** Three judge biases, circles = Tamil, squares = Hindi, with 95% CIs
(hollow = the LiteJudge length ablation).
Left: flip rate, the share of pairs whose verdict changes when the answer order is swapped
(LiteJudge is order-invariant by construction, so 0; GPT-4-32K has one order only). Middle:
share of non-tie verdicts that pick the longer answer, judge minus human majority, in
percentage points. Right: share of picks for the judge's own model family (GPT or Gemma) on
pairs with exactly one family member (tie = half a pick), judge minus humans, in percentage
points. Zero = same as humans.

**Figure (d) - `fig_d_kendall_tau`.** Kendall's τ between the ranking of the response models
by each judge's win rate (tie = 0.5) and the ranking by human-majority win rate (Tamil 14
models, Hindi 20). Circles = Tamil, squares = Hindi; hollow = the LiteJudge length ablation and the always-longer baseline. The model-identity baseline ranks well by
construction: it learns exactly the per-model strength.

**Figure (e) - `fig_e_cost_kappa`.** Logged API cost per 1,000 unique pairs (log scale) against
mean κ vs the raters, with the human ceiling. Filled markers judge both orders, hollow markers
the A/B order only (half the cost). Judges with no API cost sit in the $0 strip: SEA-LION (free,
rate-limited API) and LiteJudge (CPU only; about 29 s for Tamil and 45 s for Hindi, mostly
the one-off MuRIL embedding). Diamonds are majority-vote ensembles (cost = sum of members).
The figure uses κ against the individual raters, not κ against the majority label used in the
main tables, because only the former is on the same scale as the human ceiling (rater vs rater):
the comparison is like-for-like.
Gemma base matches the frontier judge on Tamil at ${usd.loc[('tamil', 'base_model')]:.2f} vs
${usd.loc[('tamil', 'frontier')]:.2f} per 1,000 pairs.

**Figure (f) - `fig_f_confusion_tamil`.** Confusion matrix of the best single judge on Tamil
({NAME[best]}, highest κ against the human majority) against the human-majority label: counts and
row percentages. Most disagreement is human ties that the judge resolves to A or B.

**Figure (g) - `fig_g_lomo`.** EXPLORATORY (not pre-registered). Leave-one-model-out check: for
each response model, LiteJudge and the model-identity baseline are trained on pairs without that
model and tested on the pairs with it (hollow markers; the unseen model gets the average
strength in the identity baseline). Filled markers are the pre-registered cross-validated
predictions scored on the same rows. Every pair is tested twice, once per model it contains.
LiteJudge keeps its agreement on unseen models (Tamil {s('tamil', 'litejudge_lomo', 'lomo').estimate:.2f},
Hindi {s('hindi', 'litejudge_lomo', 'lomo').estimate:.2f}) while model identity falls
(Tamil {s('tamil', 'model_id_lomo', 'lomo').estimate:.2f}, Hindi {s('hindi', 'model_id_lomo', 'lomo').estimate:.2f}).
Training pairs share prompts with the test pairs. The figure uses κ against the individual raters, not κ against the majority label used in the
main tables, because only the former is on the same scale as the human ceiling (rater vs rater):
the comparison is like-for-like.
"""
    (OUT / "captions.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Computing rater kappas and CIs (prompt-clustered bootstrap) ...", flush=True)
    frames = {lang: pair_table(lang) for lang in LANGUAGES}
    stats = compute_stats(frames)
    stats.to_csv(OUT / "figure_stats.csv", index=False)
    paths = []
    paths += fig_a(stats)
    paths += fig_b()
    paths += fig_c(stats)
    paths += fig_d()
    paths += fig_e(stats)
    f_paths, best = fig_f()
    paths += f_paths
    paths += fig_g(stats)
    write_captions(stats, best)
    for p in paths:
        print(" ", p)
    print("  results/figures/captions.md\n  results/figures/figure_stats.csv")


if __name__ == "__main__":
    main()
