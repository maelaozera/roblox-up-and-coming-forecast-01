# Roblox Up-and-Coming Forecast — @maelaozera

Entry for make.fun `roblox-up-and-coming-forecast-01` ($500): forecasts the
top 10 of Roblox's Up-and-Coming chart at every hour of the 72 hours after
entries close.

## How it works

`main.py` (Python 3.13, standard library only, no network) reads
`/input/history.ndjson` and `/input/task.json` and writes
`/output/forecast.json` (one object with `probabilities` and `top10`).

The forecast is a blend of two views of the chart:

1. **Rank diffusion with drift.** Each game's rank is modelled as a random
   walk: drift is its rank velocity over the last 24 hours (regression
   slope, shrunk by 0.3, capped at ±3 ranks/hour), diffusion is
   σ = 0.5 + 0.6·√hours. Probabilities are the time average over the
   window's 72 hourly slots. Games that age out (published more than 28
   days earlier, from `details.created` when the history carries it)
   contribute nothing after they leave.
2. **Persistence** — the house baseline's shape (0.5 on the game at #k,
   0.15 on its neighbours, 0.3 on #2 at place 1).

Blend: 65% diffusion, 35% persistence. Each place's probabilities sum to
0.8 (0.2 left unnamed for games that join the chart during the window),
and no game totals more than 1.0 over the ten places.

Two findings from the practice history drove the design:

- Playing-count *growth* does not predict rank moves (corr ≈ −0.02), so it
  is not used; rank velocity persists a little (corr ≈ 0.22).
- Brand-new games essentially never jump straight into the top 10
  (1 of 330 top-10 slots); entrants come from ranks 11–30, which the
  diffusion model covers.

## Local validation (walk-forward backtests on the practice history)

Forecast built at a cut, scored against the reads of the next 72 hours
with the competition's own log-score rule:

| cut | this entry | house baseline |
|-----|-----------|----------------|
| 2026-11-02 | −37.78 | −47.34 |
| 2026-11-04 | −39.16 | −50.88 |
| 2026-11-05 | −40.72 | −51.97 |
| 2026-11-06 | −31.01 | −37.71 |
| 2026-11-07 | −27.63 | −30.43 |

Mean over 8 cuts: about −33.7 vs −40.8 for the baseline (higher is
better; the baseline's stated practice score is −43.51). The program is
deterministic (two runs give byte-identical output) and runs in ~0.1 s
on the practice history, far inside the 300 s limit.

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
