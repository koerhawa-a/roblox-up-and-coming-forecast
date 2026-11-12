"""Roblox Up-and-Coming top-10 forecaster (Python 3.13, standard library only).

Reads /input/history.ndjson (hourly/archive snapshots of the chart) and writes
/output/forecast.json: for each of places 1..10, a probability for each game
named being at that place, plus the top 10 in order.

Model (deterministic, no network, no clock use):
  - For every game on the last snapshot, fit its recent rank velocity by
    exponentially time-weighted linear regression over its observed ranks
    (half-life ~24h), clamped so no trend explodes.
  - Expected rank at future hour h follows the velocity with momentum decay
    (time constant TAU): rank_h = rank0 + v * TAU * (1 - exp(-h / TAU)).
  - Rank uncertainty grows with horizon: sigma_h = SIGMA0 + GROW * sqrt(h).
  - P(game at place k at hour h) is the normal mass of [k-0.5, k+0.5] around
    rank_h; the forecast averages that over the 72 hourly slots.
  - Games with a known creation date stop being forecast after created+28d
    (the chart only lists games published in the last 28 days).
  - Each place keeps UNNAMED share unnamed, so games that join the chart
    later (never named) still share it instead of scoring the floor.
  - top10 orders games by their window-averaged probability of being in
    the top 10.
"""

import json
import math
import os

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

VEL_HALF_LIFE_H = 12.0   # exponential weight half-life for the velocity fit
VEL_MAX = 0.35           # clamp |velocity| in ranks per hour
TAU_H = 18.0             # momentum decay time constant (hours)
SIGMA0 = 1.6             # rank uncertainty at h=0
GROW = 0.25              # sigma growth per sqrt(hour)
UNNAMED = 0.30           # share of each place left unnamed
HOURS = 72               # scored window length
MIN_P = 0.0005           # drop rows below this (keeps the file small)


def parse_time(s):
    from datetime import datetime

    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def fit_velocity(points):
    """Weighted linear fit of rank vs hours-before-last; returns ranks/hour.

    points: list of (dt_hours_ago, rank), dt >= 0. Recent points weigh more.
    """
    if len(points) < 2:
        return 0.0
    wsum = 0.0
    wdt = 0.0
    wr = 0.0
    for dt, rank in points:
        w = 0.5 ** (dt / VEL_HALF_LIFE_H)
        wsum += w
        wdt += w * dt
        wr += w * rank
    mdt = wdt / wsum
    mr = wr / wsum
    num = 0.0
    den = 0.0
    for dt, rank in points:
        w = 0.5 ** (dt / VEL_HALF_LIFE_H)
        num += w * (dt - mdt) * (rank - mr)
        den += w * (dt - mdt) ** 2
    if den <= 0.0:
        return 0.0
    # rank falls as dt falls for a climbing game: slope vs dt is positive
    # when the game climbs, so velocity (drank/dfuture) = -slope.
    v = -num / den
    return max(-VEL_MAX, min(VEL_MAX, v))


def forecast(snaps, places=10, hours=HOURS):
    """snaps: history snapshots oldest first. Returns (probabilities, top10)."""
    snaps = sorted(snaps, key=lambda s: s["at"])
    last = snaps[-1]
    last_t = parse_time(last["at"])
    last_chart = sorted(last["up_and_coming"], key=lambda r: r["rank"])

    # Per-game rank history (hours before last snapshot) and creation dates.
    hist = {}
    created = {}
    for s in snaps:
        t = parse_time(s["at"])
        dt = (last_t - t).total_seconds() / 3600.0
        for r in s["up_and_coming"]:
            hist.setdefault(r["universe_id"], []).append((dt, r["rank"]))
        det = s.get("details") or {}
        for k, d in det.items():
            if d and d.get("created"):
                try:
                    created[int(k)] = parse_time(d["created"])
                except (ValueError, TypeError):
                    pass

    # Expected rank path per game: (rank0, velocity, expiry_hour or None).
    games = []
    for r in last_chart:
        uid = r["universe_id"]
        # Velocity from recent observations (last ~6 days, at most 12 points):
        # with hourly house reads the history is dense, so cap the count.
        pts = [p for p in hist.get(uid, []) if p[0] <= 144.0][-12:]
        v = fit_velocity(pts)
        expiry = None
        if uid in created:
            from datetime import timedelta

            exp_t = created[uid] + timedelta(days=28)
            expiry = (exp_t - last_t).total_seconds() / 3600.0
            if expiry <= 0:
                continue  # already aged out of the chart
        games.append((uid, float(r["rank"]), v, expiry))

    # Average P(place k) over the 72 hourly slots.
    probs = {uid: [0.0] * places for uid, _, _, _ in games}
    for h in range(hours):
        sigma = SIGMA0 + GROW * math.sqrt(h)
        decay = TAU_H * (1.0 - math.exp(-h / TAU_H)) if TAU_H > 0 else float(h)
        for uid, rank0, v, expiry in games:
            if expiry is not None and h >= expiry:
                continue
            mu = rank0 + v * decay
            mu = max(1.0, min(130.0, mu))
            row = probs[uid]
            for k in range(1, places + 1):
                row[k - 1] += (
                    phi((k + 0.5 - mu) / sigma) - phi((k - 0.5 - mu) / sigma)
                ) / hours

    # Emit rows, scaling each place down to at most (1 - UNNAMED).
    out = []
    for k in range(1, places + 1):
        col = [(uid, probs[uid][k - 1]) for uid, _, _, _ in games]
        col = [(u, p) for u, p in col if p >= MIN_P]
        total = sum(p for _, p in col)
        cap = 1.0 - UNNAMED
        scale = cap / total if total > cap else 1.0
        for uid, p in col:
            q = math.floor(p * scale * 1e6) / 1e6
            if q >= MIN_P:
                out.append({"universe_id": uid, "place": k, "p": q})

    # top10 by window-averaged probability of holding any top-10 place.
    scored = sorted(
        games,
        key=lambda g: -sum(probs[g[0]][k] for k in range(places)),
    )
    top10 = [uid for uid, _, _, _ in scored[:places]]
    return out, top10


def main():
    with open(os.path.join(INPUT_DIR, "task.json")) as f:
        task = json.load(f)
    snaps = []
    with open(os.path.join(INPUT_DIR, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if line:
                snaps.append(json.loads(line))
    probabilities, top10 = forecast(snaps, places=task["places"])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, task["output"]), "w") as f:
        f.write(json.dumps({"probabilities": probabilities, "top10": top10}) + "\n")


if __name__ == "__main__":
    main()
