# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/dexalot4-wq/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/dexalot4-wq/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, five quantiles of its earn rate on Exophase at 2026-11-26T00:00:00Z — the percent of the players Exophase tracks who will have earned each trophy one week after launch — from the house's reads of the game's trophy page up to the deadline. Scored by mean pinball loss on the logit scale.

## How it works

An earn rate is a ratio, `rate = E / N`, of two moving quantities: `N(t)`, the players tracked (only grows), and `E_i(t)`, how many hold trophy `i` (grows as players play long enough to earn it). Persistence ignores both motions — and launch night shows why that fails: the page sat unchanged at 638 tracked players, then one refresh took it to 3,868 and every rate but the 100% one fell, because the newcomers had earned almost nothing yet.

The median is an ensemble of two views:

1. **Structural (join/earn).** Tracked players follow a saturating Weibull curve `J(t) = J0 + A·(1 − exp(−(t/τ)^k))`, fitted to the distinct reads. A player who joins at time `s` earns trophy `i` after a lognormal delay (median `d_i`, spread `σ_i`), with probability `p_i`. The earned share is `p_i · H_i(t)`, where `H_i(t)` convolves the fitted join density with the delay CDF. `(p_i, d_i, σ_i)` are fitted per trophy.
2. **Trend.** A least-squares line of `logit(rate)` against `ln(1 + hours since release)`, fitted per trophy over all its reads, read at the target and capped at the trophy's eventual share.

The forecast median is the logit-average of the two views. Trophies nobody has earned yet lean on a description prior (story chapters: high share, short delay; grinds and 100% completion: low share, long delay) with the maximum spread, because the prior is a guess. The spread combines the fit residual, the refresh lumpiness (the answer is the page as of Exophase's last refresh before the target), the disagreement between the two views, and a horizon model-error floor.

Backtested on player-level simulations with known truth (Weibull joins, lognormal delays, 4-hour refreshes, 6 scenarios × 3 seeds), the ensemble scored a mean pinball loss of ~0.30 against ~0.49 for the persistence baseline — about 38% better — and beat persistence in every scenario, including heavy parameter jitter.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3.13, standard library only) reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (default `/input`) and writes `<id>.out` to `$MAKEFUN_OUTPUT_DIR` (default `/output`): `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`, one row per trophy, five quantiles in percent, non-decreasing. Deterministic: two runs write the same bytes. One case runs in ~0.2 s, far inside the 8 s cap.

---

Entered on [make.fun](https://make.fun/?ref=ypb7m8t9&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards
