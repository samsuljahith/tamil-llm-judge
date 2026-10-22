# Step 04 summary: agreement and bias

Unique pairs only (Tamil 1,820, Hindi 1,000). Ties kept. `parse_ok=false` treated as missing. Main pairwise view is **swap-aggregated** (A/B and mapped B/A; disagreement -> tie). `gpt4_32k` is A/B order only. Seed 42, 2,000 prompt-clustered bootstrap replicates.

## Agreement with human_majority (swap-aggregated; gpt4_32k = A/B)

| language | judge | n | accuracy | kappa | accuracy (human non-tie) |
|---|---|---|---|---|---|
| tamil | frontier | 1819 | 0.664 | 0.492 | 0.86 |
| tamil | low_cost | 1820 | 0.631 | 0.442 | 0.8399 |
| tamil | sea_lion | 1820 | 0.676 | 0.510 | 0.8552 |
| tamil | base_model | 1818 | 0.672 | 0.505 | 0.8533 |
| tamil | gpt4_32k | 1820 | 0.613 | 0.413 | 0.8608 |
| hindi | frontier | 1000 | 0.727 | 0.503 | 0.7641 |
| hindi | low_cost | 1000 | 0.690 | 0.453 | 0.7215 |
| hindi | sea_lion | 1000 | 0.714 | 0.511 | 0.7417 |
| hindi | base_model | 1000 | 0.708 | 0.506 | 0.7353 |
| hindi | gpt4_32k | 1000 | 0.774 | 0.583 | 0.8164 |

## Per-call mix and swap-aggregation (RQ3)

Each LLM call is one order. Swap-aggregation sets a tie when A/B and mapped B/A disagree, which raises the tie rate and (on Tamil) kappa. `kappa_gain` = swap κ minus the mean of the two single-order κ.

| language | judge | view | n | %A | %B | %tie | accuracy | kappa | kappa_gain |
|---|---|---|---|---|---|---|---|---|---|
| tamil | frontier | ab | 1820 | 50.05 | 44.89 | 5.05 | 0.635 | 0.448 |  |
| tamil | frontier | ba | 1819 | 45.9 | 48.82 | 5.28 | 0.642 | 0.456 |  |
| tamil | frontier | swap_agg | 1819 | 44.8 | 43.82 | 11.38 | 0.664 | 0.492 | 0.040 |
| tamil | low_cost | ab | 1820 | 51.32 | 48.08 | 0.6 | 0.599 | 0.392 |  |
| tamil | low_cost | ba | 1820 | 49.62 | 50.0 | 0.38 | 0.604 | 0.399 |  |
| tamil | low_cost | swap_agg | 1820 | 45.49 | 44.07 | 10.44 | 0.631 | 0.442 | 0.046 |
| tamil | sea_lion | ab | 1820 | 45.71 | 50.6 | 3.68 | 0.636 | 0.448 |  |
| tamil | sea_lion | ba | 1820 | 49.84 | 45.55 | 4.62 | 0.650 | 0.471 |  |
| tamil | sea_lion | swap_agg | 1820 | 42.8 | 43.08 | 14.12 | 0.676 | 0.510 | 0.051 |
| tamil | base_model | ab | 1818 | 45.6 | 50.99 | 3.41 | 0.636 | 0.447 |  |
| tamil | base_model | ba | 1820 | 50.27 | 45.77 | 3.96 | 0.647 | 0.465 |  |
| tamil | base_model | swap_agg | 1818 | 42.63 | 43.01 | 14.36 | 0.672 | 0.505 | 0.049 |
| tamil | gpt4_32k | ab | 1820 | 51.7 | 45.38 | 2.91 | 0.613 | 0.413 |  |
| hindi | frontier | ab | 1000 | 45.9 | 53.8 | 0.3 | 0.742 | 0.514 |  |
| hindi | frontier | ba | 1000 | 46.4 | 53.3 | 0.3 | 0.733 | 0.498 |  |
| hindi | frontier | swap_agg | 1000 | 43.9 | 51.3 | 4.8 | 0.727 | 0.503 | -0.003 |
| hindi | low_cost | ab | 1000 | 46.7 | 53.3 | 0.0 | 0.710 | 0.453 |  |
| hindi | low_cost | ba | 1000 | 49.5 | 50.5 | 0.0 | 0.716 | 0.466 |  |
| hindi | low_cost | swap_agg | 1000 | 43.7 | 47.5 | 8.8 | 0.690 | 0.453 | -0.006 |
| hindi | sea_lion | ab | 1000 | 42.5 | 56.6 | 0.9 | 0.761 | 0.551 |  |
| hindi | sea_lion | ba | 1000 | 52.2 | 46.6 | 1.2 | 0.753 | 0.540 |  |
| hindi | sea_lion | swap_agg | 1000 | 41.3 | 45.4 | 13.3 | 0.714 | 0.511 | -0.035 |
| hindi | base_model | ab | 1000 | 41.8 | 57.0 | 1.2 | 0.758 | 0.546 |  |
| hindi | base_model | ba | 1000 | 53.5 | 45.4 | 1.1 | 0.760 | 0.553 |  |
| hindi | base_model | swap_agg | 1000 | 40.8 | 44.3 | 14.9 | 0.708 | 0.506 | -0.044 |
| hindi | gpt4_32k | ab | 1000 | 51.2 | 46.4 | 2.4 | 0.774 | 0.583 |  |

## Human ceiling (mean pairwise Cohen's kappa)

| language | judge | human–human | judge–human | gap |
|---|---|---|---|---|
| tamil | frontier | 0.414 | 0.423 | 0.010 |
| tamil | low_cost | 0.414 | 0.386 | -0.028 |
| tamil | sea_lion | 0.414 | 0.449 | 0.035 |
| tamil | base_model | 0.414 | 0.439 | 0.025 |
| tamil | gpt4_32k | 0.414 | 0.385 | -0.029 |
| hindi | frontier | 0.431 | 0.418 | -0.013 |
| hindi | low_cost | 0.431 | 0.373 | -0.058 |
| hindi | sea_lion | 0.431 | 0.433 | 0.002 |
| hindi | base_model | 0.431 | 0.428 | -0.003 |
| hindi | gpt4_32k | 0.431 | 0.487 | 0.056 |

## Verdict mix (% tie) vs humans

- tamil humans: 31.7% ties
- hindi humans: 6.3% ties
- tamil judges (swap-agg / gpt4 A/B): frontier 11.4%, low_cost 10.4%, sea_lion 14.1%, base_model 14.4%, gpt4_32k 2.9%
- hindi judges (swap-agg / gpt4 A/B): frontier 4.8%, low_cost 8.8%, sea_lion 13.3%, base_model 14.9%, gpt4_32k 2.4%

## Position bias (LLM judges, both orders)

| language | judge | flip rate | first-position preference |
|---|---|---|---|
| tamil | frontier | 0.0918 | 0.0426 |
| tamil | low_cost | 0.1038 | 0.0182 |
| tamil | sea_lion | 0.1187 | -0.0479 |
| tamil | base_model | 0.1254 | -0.0514 |
| hindi | frontier | 0.046 | -0.005 |
| hindi | low_cost | 0.088 | -0.028 |
| hindi | sea_lion | 0.129 | -0.0995 |
| hindi | base_model | 0.144 | -0.1179 |

## Length bias (% of non-tie verdicts choosing the longer answer)

| language | judge | n | % longer | logit p |
|---|---|---|---|---|
| tamil | human_majority | 1243 | 55.23 | 0.0 |
| tamil | frontier | 1612 | 49.5 | 0.1104 |
| tamil | low_cost | 1630 | 48.71 | 0.78701 |
| tamil | sea_lion | 1563 | 52.59 | 0.000649 |
| tamil | base_model | 1557 | 52.86 | 0.00012 |
| tamil | gpt4_32k | 1767 | 59.43 | 0.0 |
| hindi | human_majority | 937 | 61.32 | 0.0 |
| hindi | frontier | 952 | 52.89 | 0.064335 |
| hindi | low_cost | 912 | 49.95 | 0.952088 |
| hindi | sea_lion | 867 | 56.12 | 0.000495 |
| hindi | base_model | 851 | 56.47 | 0.000125 |
| hindi | gpt4_32k | 976 | 58.26 | 4e-06 |

## Self-preference (exactly one family member in the pair; tie = 0.5)

| language | judge | family | n | judge % | human % |
|---|---|---|---|---|---|
| tamil | frontier | gpt | 660 | 58.18 | 54.02 |
| tamil | low_cost | gpt | 660 | 56.36 | 54.02 |
| tamil | sea_lion | gemma | 800 | 60.0 | 61.25 |
| tamil | base_model | gemma | 799 | 59.7 | 61.2 |
| tamil | gpt4_32k | gpt | 660 | 62.12 | 54.02 |
| hindi | frontier | gpt | 260 | 67.5 | 58.27 |
| hindi | low_cost | gpt | 260 | 68.27 | 58.27 |
| hindi | sea_lion | gemma | 345 | 39.86 | 48.41 |
| hindi | base_model | gemma | 345 | 41.45 | 48.41 |
| hindi | gpt4_32k | gpt | 260 | 61.92 | 58.27 |

## Bootstrap 95% CIs: judge minus human (length % and self-preference)

| language | judge | metric | estimate | 95% CI |
|---|---|---|---|---|
| tamil | frontier | length_minus_human | -0.057 | [-0.086, -0.026] |
| tamil | frontier | selfpref_minus_human | 0.042 | [0.000, 0.083] |
| tamil | low_cost | length_minus_human | -0.065 | [-0.090, -0.039] |
| tamil | low_cost | selfpref_minus_human | 0.023 | [-0.022, 0.064] |
| tamil | sea_lion | length_minus_human | -0.026 | [-0.055, 0.004] |
| tamil | sea_lion | selfpref_minus_human | -0.013 | [-0.051, 0.024] |
| tamil | base_model | length_minus_human | -0.023 | [-0.053, 0.005] |
| tamil | base_model | selfpref_minus_human | -0.015 | [-0.052, 0.019] |
| tamil | gpt4_32k | length_minus_human | 0.043 | [0.006, 0.082] |
| tamil | gpt4_32k | selfpref_minus_human | 0.081 | [0.036, 0.127] |
| hindi | frontier | length_minus_human | -0.084 | [-0.114, -0.053] |
| hindi | frontier | selfpref_minus_human | 0.095 | [0.006, 0.179] |
| hindi | low_cost | length_minus_human | -0.114 | [-0.146, -0.082] |
| hindi | low_cost | selfpref_minus_human | 0.101 | [0.021, 0.182] |
| hindi | sea_lion | length_minus_human | -0.052 | [-0.085, -0.023] |
| hindi | sea_lion | selfpref_minus_human | -0.086 | [-0.144, -0.033] |
| hindi | base_model | length_minus_human | -0.049 | [-0.079, -0.021] |
| hindi | base_model | selfpref_minus_human | -0.070 | [-0.126, -0.020] |
| hindi | gpt4_32k | length_minus_human | -0.030 | [-0.064, 0.001] |
| hindi | gpt4_32k | selfpref_minus_human | 0.036 | [-0.017, 0.096] |

## Leaderboard ranking agreement (Kendall's tau vs human win rates)

| language | judge | n models | tau | p |
|---|---|---|---|---|
| tamil | gpt4_32k | 14 | 0.714 | 0.000145 |
| tamil | frontier | 14 | 0.890 | 0.0 |
| tamil | low_cost | 14 | 0.872 | 1.8e-05 |
| tamil | sea_lion | 14 | 0.890 | 0.0 |
| tamil | base_model | 14 | 0.868 | 1e-06 |
| hindi | gpt4_32k | 20 | 0.744 | 5e-06 |
| hindi | frontier | 20 | 0.726 | 1e-06 |
| hindi | low_cost | 20 | 0.681 | 2.8e-05 |
| hindi | sea_lion | 20 | 0.695 | 3e-06 |
| hindi | base_model | 20 | 0.716 | 1e-06 |

## Pre-specified contrasts (prompt-clustered bootstrap, Holm-adjusted)

| contrast | estimate | 95% CI | p | p_Holm |
|---|---|---|---|---|
| kappa tamil: sea_lion − base_model | 0.006 | [-0.009, 0.021] | 0.427 | 1.0 |
| kappa hindi: sea_lion − base_model | 0.005 | [-0.012, 0.021] | 0.579 | 1.0 |
| kappa frontier: tamil − hindi | -0.011 | [-0.099, 0.075] | 0.803 | 1.0 |
| kappa low_cost: tamil − hindi | -0.010 | [-0.103, 0.084] | 0.828 | 1.0 |
| kappa sea_lion: tamil − hindi | -0.002 | [-0.101, 0.094] | 0.972 | 1.0 |
| kappa base_model: tamil − hindi | -0.001 | [-0.091, 0.082] | 0.972 | 1.0 |
| kappa gpt4_32k: tamil − hindi | -0.170 | [-0.258, -0.086] | 0.0 | 0.0 |

## Tamil direct scores (0–5 = LA + TQ + (1 − H))

| judge | n | Spearman | QWK | QWK LA | QWK TQ | QWK H |
|---|---|---|---|---|---|---|
| gpt4_32k | 280 | 0.639 | 0.474 | 0.494 | 0.339 | 0.456 |
| frontier | 280 | 0.736 | 0.670 | 0.569 | 0.716 | 0.317 |
| low_cost | 280 | 0.685 | 0.583 | 0.544 | 0.665 | 0.127 |
| sea_lion | 280 | 0.764 | 0.676 | 0.632 | 0.670 | 0.177 |
| base_model | 280 | 0.771 | 0.677 | 0.643 | 0.670 | 0.184 |

## Tamil confusion (swap-aggregated judge vs human_majority)

Rows = human majority; columns = judge. Off-diagonal human-tie cells are the missed-tie mass.

### sea_lion

| human \ judge | A | B | tie | total |
|---|---|---|---|---|
| A | 510 | 39 | 46 | 595 |
| B | 51 | 553 | 44 | 648 |
| tie | 218 | 192 | 167 | 577 |
| total | 779 | 784 | 257 | 1820 |

### frontier

| human \ judge | A | B | tie | total |
|---|---|---|---|---|
| A | 513 | 48 | 34 | 595 |
| B | 58 | 556 | 34 | 648 |
| tie | 244 | 193 | 139 | 576 |
| total | 815 | 797 | 207 | 1819 |

All numbers are computed from `results/judgements_clean/` and `data/processed/`. No new API calls.
