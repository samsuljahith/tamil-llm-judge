# EXPLORATORY: leave-one-model-out check (not pre-registered)

Added on 10 Nov 2026 after the pre-registered results were seen. For each response model, LiteJudge (primary recipe) and the model-identity baseline are trained on pairs that do not involve it and tested on the pairs that do. Every pair is tested twice (once per model), so rows = 2 x pairs. The model-identity baseline gives the unseen model the average strength. `main_cv` = the pre-registered out-of-fold verdicts scored on the same rows. Training pairs share prompts with test pairs. Contrasts are unadjusted.

| language | judge | scheme | rows | κ vs majority | 95% CI | κ vs raters | gap to ceiling | % tie |
|---|---|---|---|---|---|---|---|---|
| tamil | litejudge_lomo | lomo | 3640 | 0.319 | [0.245, 0.390] | 0.296 | -0.117 | 6.9 |
| tamil | model_id_lomo | lomo | 3640 | 0.236 | [0.177, 0.292] | 0.222 | -0.192 | 4.4 |
| tamil | litejudge | main_cv | 3640 | 0.331 | [0.258, 0.408] | 0.304 | -0.110 | 6.1 |
| tamil | model_id | main_cv | 3640 | 0.366 | [0.300, 0.431] | 0.342 | -0.072 | 6.4 |
| hindi | litejudge_lomo | lomo | 2000 | 0.551 | [0.523, 0.578] | 0.469 | 0.038 | 0.0 |
| hindi | model_id_lomo | lomo | 2000 | 0.272 | [0.235, 0.309] | 0.233 | -0.198 | 0.0 |
| hindi | litejudge | main_cv | 2000 | 0.535 | [0.479, 0.589] | 0.457 | 0.026 | 0.0 |
| hindi | model_id | main_cv | 2000 | 0.480 | [0.417, 0.549] | 0.401 | -0.030 | 0.0 |

| contrast | estimate | 95% CI | p (unadjusted) |
|---|---|---|---|
| kappa tamil: litejudge_lomo − model_id_lomo | 0.082 | [0.037, 0.133] | 0.0 |
| kappa tamil: litejudge_lomo − litejudge | -0.013 | [-0.036, 0.010] | 0.274 |
| kappa hindi: litejudge_lomo − model_id_lomo | 0.278 | [0.240, 0.318] | 0.0 |
| kappa hindi: litejudge_lomo − litejudge | 0.015 | [-0.023, 0.053] | 0.431 |

Max LiteJudge P(tie) over all held-out predictions: tamil 0.365, hindi 0.099.
