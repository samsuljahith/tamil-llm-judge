# Step 07 summary: ensembles and cost

Plurality vote (no clear winner -> tie) of swap-aggregated LLM verdicts and LiteJudge's out-of-fold verdict; step 04 metrics on the same unique pairs.

## Main ensembles

| language | ensemble | κ vs majority | 95% CI | gap to ceiling | % tie |
|---|---|---|---|---|---|
| tamil | llm4 | 0.524 | [0.452, 0.588] | 0.037 | 18.6 |
| tamil | cheap3 | 0.513 | [0.443, 0.583] | 0.034 | 13.8 |
| tamil | llm4_lite | 0.509 | [0.441, 0.576] | 0.029 | 12.2 |
| tamil | free_lite | 0.508 | [0.438, 0.577] | 0.030 | 14.1 |
| hindi | llm4 | 0.486 | [0.429, 0.546] | -0.024 | 17.4 |
| hindi | cheap3 | 0.513 | [0.456, 0.569] | 0.002 | 12.9 |
| hindi | llm4_lite | 0.538 | [0.481, 0.597] | 0.023 | 8.5 |
| hindi | free_lite | 0.519 | [0.467, 0.574] | 0.009 | 13.6 |

## Pre-specified contrasts vs frontier (Holm over 8)

| contrast | estimate | 95% CI | p | p_Holm |
|---|---|---|---|---|
| kappa tamil: llm4 − frontier | 0.031 | [-0.013, 0.073] | 0.161 | 1.0 |
| kappa tamil: cheap3 − frontier | 0.022 | [-0.033, 0.073] | 0.407 | 1.0 |
| kappa tamil: llm4_lite − frontier | 0.017 | [-0.033, 0.058] | 0.451 | 1.0 |
| kappa tamil: free_lite − frontier | 0.017 | [-0.038, 0.069] | 0.534 | 1.0 |
| kappa hindi: llm4 − frontier | -0.017 | [-0.052, 0.019] | 0.363 | 1.0 |
| kappa hindi: cheap3 − frontier | 0.011 | [-0.027, 0.047] | 0.589 | 1.0 |
| kappa hindi: llm4_lite − frontier | 0.036 | [0.002, 0.070] | 0.044 | 0.352 |
| kappa hindi: free_lite − frontier | 0.015 | [-0.024, 0.054] | 0.458 | 1.0 |

## Cost vs agreement (US$ per 1,000 unique pairs)

| language | judge | view | κ | US$ / 1,000 pairs | κ per US$ | compute s |
|---|---|---|---|---|---|---|
| tamil | frontier | ab | 0.448 | 8.19 | 0.0546 |  |
| tamil | frontier | swap_agg | 0.492 | 16.44 | 0.0299 |  |
| tamil | low_cost | ab | 0.392 | 1.05 | 0.3728 |  |
| tamil | low_cost | swap_agg | 0.442 | 2.08 | 0.2119 |  |
| tamil | sea_lion | ab | 0.448 | 0.00 |  |  |
| tamil | sea_lion | swap_agg | 0.510 | 0.00 |  |  |
| tamil | base_model | ab | 0.447 | 0.06 | 7.6444 |  |
| tamil | base_model | swap_agg | 0.505 | 0.12 | 4.3108 |  |
| tamil | litejudge | oof | 0.331 | 0.00 |  | 28.9 |
| tamil | llm4 | ensemble | 0.524 | 18.64 | 0.0281 |  |
| tamil | cheap3 | ensemble | 0.513 | 2.20 | 0.2333 |  |
| tamil | llm4_lite | ensemble | 0.509 | 18.64 | 0.0273 | 28.9 |
| tamil | free_lite | ensemble | 0.508 | 0.12 | 4.3424 | 28.9 |
| hindi | frontier | ab | 0.514 | 7.51 | 0.0685 |  |
| hindi | frontier | swap_agg | 0.503 | 15.03 | 0.0335 |  |
| hindi | low_cost | ab | 0.453 | 1.01 | 0.4503 |  |
| hindi | low_cost | swap_agg | 0.453 | 2.01 | 0.2249 |  |
| hindi | sea_lion | ab | 0.551 | 0.00 |  |  |
| hindi | sea_lion | swap_agg | 0.511 | 0.00 |  |  |
| hindi | base_model | ab | 0.546 | 0.07 | 7.6100 |  |
| hindi | base_model | swap_agg | 0.506 | 0.14 | 3.5566 |  |
| hindi | litejudge | oof | 0.535 | 0.00 |  | 45.2 |
| hindi | llm4 | ensemble | 0.486 | 17.19 | 0.0283 |  |
| hindi | cheap3 | ensemble | 0.513 | 2.16 | 0.2380 |  |
| hindi | llm4_lite | ensemble | 0.538 | 17.19 | 0.0313 | 45.2 |
| hindi | free_lite | ensemble | 0.519 | 0.14 | 3.6451 | 45.2 |

## Order-swapping: κ gain for the second order

| language | judge | κ A/B | κ swap-agg | gain | extra US$ / 1,000 pairs |
|---|---|---|---|---|---|
| tamil | frontier | 0.448 | 0.492 | 0.044 | 8.25 |
| tamil | low_cost | 0.392 | 0.442 | 0.050 | 1.03 |
| tamil | sea_lion | 0.448 | 0.510 | 0.062 | 0.00 |
| tamil | base_model | 0.447 | 0.505 | 0.058 | 0.06 |
| hindi | frontier | 0.514 | 0.503 | -0.011 | 7.52 |
| hindi | low_cost | 0.453 | 0.453 | -0.000 | 1.01 |
| hindi | sea_lion | 0.551 | 0.511 | -0.040 | 0.00 |
| hindi | base_model | 0.546 | 0.506 | -0.040 | 0.07 |

## EXPLORATORY: best five voter subsets by κ (no tests)

- tamil: frontier+sea_lion κ 0.548 (tie 25.9%); frontier+base_model κ 0.545 (tie 26.6%); frontier+low_cost+sea_lion+base_model κ 0.524 (tie 18.6%); frontier+sea_lion+base_model+litejudge κ 0.520 (tie 15.9%); frontier+sea_lion+base_model κ 0.520 (tie 14.6%)
- hindi: frontier+sea_lion+litejudge κ 0.549 (tie 7.1%); frontier+base_model+litejudge κ 0.546 (tie 8.3%); low_cost+sea_lion+litejudge κ 0.546 (tie 9.3%); low_cost+base_model+litejudge κ 0.539 (tie 10.4%); frontier+low_cost+sea_lion+base_model+litejudge κ 0.538 (tie 8.5%)

SEA-LION is a free, rate-limited API ($0). LiteJudge has no API cost; its compute time (embedding wall-clock + fit/predict CPU) is shown instead of a dollar figure.
