# Figure captions (drafts)

All figures: unique pairs only (Tamil 1,820, Hindi 1,000), ties kept. LLM judges are
swap-aggregated (A/B and mapped B/A; disagreement = tie) unless stated. Frontier = GPT-5.5; Low-cost = GPT-5.4 mini; SEA-LION = Gemma-SEA-LION-v4-27B-IT; Gemma base = Gemma 3 27B IT; GPT-4-32K = PARIKSHA's original verdict (A/B order only). LiteJudge
results are out-of-fold (5 folds grouped by prompt). 95% CIs are prompt-clustered bootstrap
intervals (2,000 replicates, seed 42).

**Figure (a) - `fig_a_kappa_ceiling`.** Agreement of each judge with the human raters, as the
mean Cohen's κ of the judge against each of the three individual raters, with 95% CIs. The
black line and band show the human ceiling, the mean pairwise κ between the three raters
(Tamil 0.41 [0.35, 0.48], Hindi 0.43
[0.39, 0.48]); judges and ceiling use the same bootstrap resamples. Grey
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
Gemma base matches the frontier judge on Tamil at $0.12 vs
$16.44 per 1,000 pairs.

**Figure (f) - `fig_f_confusion_tamil`.** Confusion matrix of the best single judge on Tamil
(SEA-LION, highest κ against the human majority) against the human-majority label: counts and
row percentages. Most disagreement is human ties that the judge resolves to A or B.

**Figure (g) - `fig_g_lomo`.** EXPLORATORY (not pre-registered). Leave-one-model-out check: for
each response model, LiteJudge and the model-identity baseline are trained on pairs without that
model and tested on the pairs with it (hollow markers; the unseen model gets the average
strength in the identity baseline). Filled markers are the pre-registered cross-validated
predictions scored on the same rows. Every pair is tested twice, once per model it contains.
LiteJudge keeps its agreement on unseen models (Tamil 0.30,
Hindi 0.47) while model identity falls
(Tamil 0.22, Hindi 0.23).
Training pairs share prompts with the test pairs. The figure uses κ against the individual raters, not κ against the majority label used in the
main tables, because only the former is on the same scale as the human ceiling (rater vs rater):
the comparison is like-for-like.
