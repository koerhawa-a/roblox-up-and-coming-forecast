# Roblox Up-and-Coming Top-10 Forecaster

Entry for make.fun's `roblox-up-and-coming-forecast-01`: a program that reads
the history of Roblox's Up-and-Coming chart and forecasts its top 10, with a
probability for each game at each place, for the 72 hours after entries close.

## How it works

`main.py` (Python 3.13, standard library only, deterministic, no network):

1. **Rank velocity.** For every game on the last snapshot, fit its recent rank
   velocity by exponentially time-weighted linear regression over its observed
   ranks (12-hour half-life, last 12 observations within 6 days), clamped so
   no trend explodes.
2. **Momentum decay.** Expected rank at future hour `h` follows the velocity
   with an 18-hour decay constant: `rank_h = rank0 + v * 18 * (1 - exp(-h/18))`.
3. **Growing uncertainty.** Rank uncertainty widens with horizon:
   `sigma_h = 1.6 + 0.25 * sqrt(h)`.
4. **Place probabilities.** P(game at place k at hour h) is the normal mass of
   `[k-0.5, k+0.5]` around the expected rank; the forecast averages that over
   all 72 hourly slots of the scored window.
5. **Ageing out.** Games with a known creation date stop being forecast after
   `created + 28 days` (the chart only lists games published in the last
   28 days).
6. **Room for newcomers.** Each place keeps 30% unnamed, so games that join
   the chart later still share it instead of scoring the log-score floor.

`top10` orders games by their window-averaged probability of holding any
top-10 place.

## Local backtest

Split the kit history at two cutoffs and score the forecast against the later
snapshots with the competition's log-score rule:

| cutoff | this forecast | house baseline (no-change) |
| ------ | ------------- | -------------------------- |
| 2026-11-05 (9 reads) | -40.5 | -52.0 |
| 2026-11-06 (5 reads) | -31.5 | -32.9 |

See `backtest.py` in the builder's workspace for the harness (not part of the
entry; the entry is `main.py` alone).

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads `$MAKEFUN_INPUT_DIR/history.ndjson` and `task.json`, writes
`$MAKEFUN_OUTPUT_DIR/forecast.json`. Two runs on the same input write the
same bytes.

## License

MIT — see `LICENSE`.
