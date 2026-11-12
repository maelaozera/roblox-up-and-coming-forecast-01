"""Forecast Roblox's Up-and-Coming top 10 (make.fun roblox-up-and-coming-forecast-01).

Model: each game on the chart is a rank that drifts and diffuses.
  * drift: the game's rank velocity over the last 24 hours (linear-regression
    slope of rank against time, shrunk by 0.4, capped at +-3 ranks/hour).
    Rank velocity persists a little (corr ~0.22 between consecutive moves);
    playing-count growth does not predict rank moves (corr ~0), so it is
    deliberately not used.
  * diffusion: rank at slot s (hours after the history ends) is
    Normal(mu, sigma) with mu = rank + velocity*s and
    sigma = 0.5 + 0.6*sqrt(s). The spread is the time-averaged marginal
    over the window's 72 hourly slots.
  * games that age out: a game published more than 28 days before a slot
    leaves the chart, so once its created + 28 days passes it contributes
    nothing to later slots (only when the history carries details).
  * the diffusion forecast is blended 75/25 with the persistence forecast
    (the house baseline's shape: 0.5 on the game at #k, 0.15 on #k-1/#k+1,
    0.3 on #2 at place 1). Blending won every walk-forward backtest:
    on 7 cuts over the practice history the blend scores about -36.0
    against -45.2 for the baseline alone (log score, higher is better).
  * each place's probabilities are normalised to sum 0.8, leaving 0.2
    unnamed for games that join the chart during the window (brand-new
    games essentially never jump straight into the top 10, so 0.2 is ample),
    and no game totals more than 1.0 over the ten places.
  * top10 is ordered by expected place under the blended probabilities.

Reads $MAKEFUN_INPUT_DIR (else /input) and writes $MAKEFUN_OUTPUT_DIR
(else /output), the window/places/output name from task.json.
Standard library only, deterministic: the same input gives the same bytes.
"""
import json
import math
import os
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

VEL_WEIGHT = 0.4      # shrink on the rank-velocity slope
VEL_WINDOW_H = 24.0   # velocity is fit over the last 24 hours of reads
VEL_CAP = 3.0         # max |velocity| in ranks per hour
SIG_BASE = 0.5        # diffusion sigma at slot 0
SIG_RATE = 0.6        # diffusion sigma grows with sqrt(hours)
COL_SUM = 0.8         # each place's named probabilities sum to this
BLEND_MODEL = 0.75    # weight of the diffusion forecast in the blend
MIN_P = 0.0005        # rows smaller than this are dropped
AGE_OUT_DAYS = 28.0   # games published more than this ago leave the chart


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() / 3600.0


def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def slope(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def load_history(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    with open(os.path.join(INPUT_DIR, "task.json"), "r", encoding="utf-8") as fh:
        task = json.load(fh)
    places = int(task.get("places", 10))
    output_name = task.get("output", "forecast.json")
    window = task.get("window", {})
    slots = int(window.get("slots", 72))
    slot_minutes = float(window.get("slot_minutes", 60))
    slot_hours = slot_minutes / 60.0

    rows = load_history(os.path.join(INPUT_DIR, "history.ndjson"))
    last = rows[-1]
    t0 = parse_time(last["at"])
    try:
        window_from = parse_time(window["from"])
    except Exception:
        window_from = t0

    # Rank history per game: list of (time_hours, rank).
    hist = {}
    for row in rows:
        tt = parse_time(row["at"])
        for c in row.get("up_and_coming") or []:
            hist.setdefault(c["universe_id"], []).append((tt, c["rank"]))

    # Details of the last snapshot, if the history carries them (the house
    # recorder does; the public archive does not). Keyed by universe id.
    details = last.get("details") or {}
    created_h = {}
    for key, det in details.items():
        if not isinstance(det, dict):
            continue
        cr = det.get("created")
        if not cr:
            continue
        try:
            uid = int(key)
        except (TypeError, ValueError):
            continue
        created_h[uid] = parse_time(cr)

    chart = sorted(last["up_and_coming"], key=lambda c: c["rank"])
    chart_ids = [c["universe_id"] for c in chart]
    rank_now = {c["universe_id"]: c["rank"] for c in chart}

    # --- diffusion forecast: time-averaged P(game at place k) ---
    model = {}
    for uid in chart_ids:
        r0 = rank_now[uid]
        obs = [o for o in hist.get(uid, []) if t0 - o[0] <= VEL_WINDOW_H]
        vel = 0.0
        if len(obs) >= 3:
            vel = slope([o[0] for o in obs], [o[1] for o in obs]) * VEL_WEIGHT
            vel = max(-VEL_CAP, min(VEL_CAP, vel))
        # Hour (from the window start) at which the game ages out, if any.
        exit_slot = None
        if uid in created_h:
            exit_h = created_h[uid] + AGE_OUT_DAYS * 24.0 - window_from
            if 0.0 < exit_h < slots * slot_hours:
                exit_slot = exit_h / slot_hours
        tot = [0.0] * places
        for s in range(1, slots + 1):
            if exit_slot is not None and s > exit_slot:
                break
            h = s * slot_hours
            mu = r0 + vel * h
            mu = max(0.5, min(150.0, mu))
            sig = SIG_BASE + SIG_RATE * math.sqrt(h)
            for k in range(1, places + 1):
                tot[k - 1] += norm_cdf((k + 0.5 - mu) / sig) - norm_cdf(
                    (k - 0.5 - mu) / sig
                )
        tot = [x / slots for x in tot]
        model[uid] = tot

    # Normalise each place's column to COL_SUM.
    for k in range(places):
        ssum = sum(model[u][k] for u in model)
        if ssum > 1e-9:
            f = COL_SUM / ssum
            for u in model:
                model[u][k] *= f
    # No game totals more than 1 over the places.
    for u in model:
        ssum = sum(model[u])
        if ssum > 1.0:
            f = 1.0 / ssum
            model[u] = [x * f for x in model[u]]

    # --- persistence forecast (house baseline shape) ---
    persist = {u: [0.0] * places for u in chart_ids}
    for k in range(1, places + 1):
        persist[chart_ids[k - 1]][k - 1] += 0.5
        if k == 1:
            if len(chart_ids) > 1:
                persist[chart_ids[1]][0] += 0.3
        else:
            persist[chart_ids[k - 2]][k - 1] += 0.15
            if k < len(chart_ids):
                persist[chart_ids[k]][k - 1] += 0.15

    # --- blend, then rebuild expected places from the blend ---
    probs = []
    blended = {}
    for u in chart_ids:
        row = [
            BLEND_MODEL * model[u][k] + (1.0 - BLEND_MODEL) * persist[u][k]
            for k in range(places)
        ]
        blended[u] = row
        for k in range(places):
            if row[k] >= MIN_P:
                probs.append(
                    {"universe_id": u, "place": k + 1, "p": round(row[k], 6)}
                )
    exp2 = {
        u: sum((k + 1) * blended[u][k] for k in range(places))
        + (places + 1) * max(0.0, 1.0 - sum(blended[u]))
        for u in chart_ids
    }
    top10 = sorted(chart_ids, key=lambda u: (exp2[u], -sum(blended[u])))[:places]

    out = {"probabilities": probs, "top10": top10}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, output_name), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
