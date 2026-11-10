# Step 06 summary: LiteJudge vs LLM judges and baselines

Out-of-fold predictions (5 outer folds grouped by prompt_id, seed 42); every metric is computed with step 04's functions on the same unique pairs. LLM judges are swap-aggregated (gpt4_32k = A/B only). Primary LiteJudge = MuRIL embeddings only (pre-registered).

## Tamil (human ties 31.7%, human–human κ 0.414)

| judge | n | accuracy | κ vs majority | 95% CI | κ vs raters | gap to ceiling | % tie | % longer | Kendall τ |
|---|---|---|---|---|---|---|---|---|---|
| frontier | 1819 | 0.664 | 0.492 | [0.438, 0.543] | 0.423 | 0.010 | 11.4 | 49.5 | 0.890 |
| low_cost | 1820 | 0.631 | 0.442 | [0.380, 0.502] | 0.386 | -0.028 | 10.4 | 48.7 | 0.872 |
| sea_lion | 1820 | 0.676 | 0.510 | [0.438, 0.582] | 0.449 | 0.035 | 14.1 | 52.6 | 0.890 |
| base_model | 1818 | 0.672 | 0.505 | [0.434, 0.572] | 0.439 | 0.025 | 14.4 | 52.9 | 0.868 |
| gpt4_32k | 1820 | 0.613 | 0.413 | [0.344, 0.481] | 0.385 | -0.029 | 2.9 | 59.4 | 0.714 |
| litejudge | 1820 | 0.559 | 0.331 | [0.256, 0.406] | 0.304 | -0.110 | 6.1 | 57.1 | 0.538 |
| litejudge_length | 1820 | 0.549 | 0.317 | [0.250, 0.389] | 0.292 | -0.122 | 6.8 | 57.9 | 0.538 |
| longer | 1820 | 0.378 | 0.055 | [0.004, 0.113] | 0.052 | -0.362 | 0.0 | 100.0 | 0.022 |
| majority | 1820 | 0.356 | 0.000 | [0.000, 0.000] | 0.000 | -0.414 | 0.0 | 49.5 |  |
| model_id | 1820 | 0.582 | 0.366 | [0.297, 0.432] | 0.342 | -0.072 | 6.4 | 54.4 | 0.956 |

## Hindi (human ties 6.3%, human–human κ 0.431)

| judge | n | accuracy | κ vs majority | 95% CI | κ vs raters | gap to ceiling | % tie | % longer | Kendall τ |
|---|---|---|---|---|---|---|---|---|---|
| frontier | 1000 | 0.727 | 0.503 | [0.436, 0.569] | 0.418 | -0.013 | 4.8 | 52.9 | 0.726 |
| low_cost | 1000 | 0.690 | 0.453 | [0.386, 0.516] | 0.373 | -0.058 | 8.8 | 50.0 | 0.681 |
| sea_lion | 1000 | 0.714 | 0.511 | [0.454, 0.573] | 0.433 | 0.002 | 13.3 | 56.1 | 0.695 |
| base_model | 1000 | 0.708 | 0.506 | [0.450, 0.561] | 0.428 | -0.003 | 14.9 | 56.5 | 0.716 |
| gpt4_32k | 1000 | 0.774 | 0.583 | [0.524, 0.638] | 0.487 | 0.056 | 2.4 | 58.3 | 0.744 |
| litejudge | 1000 | 0.753 | 0.535 | [0.480, 0.586] | 0.457 | 0.026 | 0.0 | 63.3 | 0.695 |
| litejudge_length | 1000 | 0.769 | 0.565 | [0.515, 0.608] | 0.475 | 0.043 | 0.0 | 64.9 | 0.695 |
| longer | 1000 | 0.574 | 0.199 | [0.137, 0.259] | 0.177 | -0.254 | 0.0 | 100.0 | 0.301 |
| majority | 1000 | 0.487 | 0.000 | [0.000, 0.000] | 0.000 | -0.431 | 0.0 | 50.0 |  |
| model_id | 1000 | 0.724 | 0.480 | [0.417, 0.553] | 0.401 | -0.030 | 0.0 | 60.0 | 0.926 |

## Pre-specified contrasts (paired prompt-clustered bootstrap, Holm over all rows)

| contrast | estimate | 95% CI | p | p_Holm |
|---|---|---|---|---|
| kappa tamil: litejudge − litejudge_length | 0.014 | [-0.005, 0.033] | 0.142 | 0.852 |
| kappa tamil: litejudge − model_id | -0.035 | [-0.089, 0.017] | 0.187 | 0.852 |
| kappa tamil: litejudge − longer | 0.277 | [0.230, 0.330] | 0.0 | 0.0 |
| kappa tamil: litejudge − frontier | -0.160 | [-0.220, -0.101] | 0.0 | 0.0 |
| kappa tamil: litejudge − low_cost | -0.110 | [-0.159, -0.060] | 0.0 | 0.0 |
| kappa tamil: litejudge − sea_lion | -0.179 | [-0.230, -0.130] | 0.0 | 0.0 |
| kappa tamil: litejudge − base_model | -0.174 | [-0.222, -0.120] | 0.0 | 0.0 |
| kappa hindi: litejudge − litejudge_length | -0.030 | [-0.064, 0.004] | 0.085 | 0.595 |
| kappa hindi: litejudge − model_id | 0.056 | [-0.026, 0.140] | 0.166 | 0.852 |
| kappa hindi: litejudge − longer | 0.337 | [0.276, 0.406] | 0.0 | 0.0 |
| kappa hindi: litejudge − frontier | 0.032 | [-0.046, 0.106] | 0.441 | 1.0 |
| kappa hindi: litejudge − low_cost | 0.082 | [-0.003, 0.173] | 0.063 | 0.504 |
| kappa hindi: litejudge − sea_lion | 0.025 | [-0.054, 0.107] | 0.55 | 1.0 |
| kappa hindi: litejudge − base_model | 0.028 | [-0.044, 0.100] | 0.445 | 1.0 |

## Settings chosen by the inner CV (per outer fold)

| language | judge | PCA | L2 | test log-loss |
|---|---|---|---|---|
| tamil | litejudge | none, none, none, none, none | 1, 1, 1, 1, 1 | 0.934 |
| tamil | litejudge_length | none, none, none, 16, 32 | 1, 1, 1, 0.01, 0.01 | 0.940 |
| tamil | model_id | none, none, none, none, none | 0.001, 0.001, 0.001, 0.001, 0.001 | 0.906 |
| hindi | litejudge | none, none, none, 64, 64 | 1, 1, 0.1, 0.01, 0.01 | 0.716 |
| hindi | litejudge_length | 32, 16, 16, 64, 16 | 0.001, 0.001, 0.001, 0.01, 0.001 | 0.700 |
| hindi | model_id | none, none, none, none, none | 0.001, 0.001, 0.001, 0.001, 0.001 | 0.745 |

## Tamil confusion: primary LiteJudge vs human_majority

| human \ judge | A | B | tie | total |
|---|---|---|---|---|
| A | 461 | 104 | 30 | 595 |
| B | 107 | 508 | 33 | 648 |
| tie | 269 | 260 | 48 | 577 |
| total | 837 | 872 | 111 | 1820 |

Swap check: max |P(order A/B) − mapped P(order B/A)| = 0.0e+00 (order-invariant). Primary CPU time: fit 0.14 s, predict 1.24 ms per 1,000 pairs (embedding time is in data/embeddings/MANIFEST.json).
