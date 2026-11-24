"""Tally "other emissions" into frequency x bandwidth buckets.

Frequency buckets are fixed `step_hz` slices of the capture (default 100 kHz). Bandwidth
buckets are < 25 kHz (FHSS hops, carriers), 25-200 kHz, 200-600 kHz and >= 600 kHz.
Rates and duty cycle are per hour of *channel capture* in the window (from the
channel_start/channel_stop events), not wall-clock hours, because nothing is recorded during
sweeps or downtime.
"""
from __future__ import annotations

BW_EDGES_HZ = [25e3, 200e3, 600e3]  # bucket i: [edge[i-1], edge[i])


def bw_bucket_label(i: int) -> str:
    k = [e / 1e3 for e in BW_EDGES_HZ]
    if i == 0:
        return f"< {k[0]:g}"
    if i >= len(k):
        return f"≥ {k[-1]:g}"
    return f"{k[i - 1]:g}–{k[i]:g}"


def capture_seconds(db, t0: float, now: float | None = None) -> float:
    """Seconds of channel capture since t0. A session with no stop event (still running, or
    killed) ends at the next start, or at its last band_seconds row."""
    import time
    now = time.time() if now is None else now
    ev = db.query("SELECT kind, ts FROM events WHERE kind IN ('channel_start', 'channel_stop') ORDER BY ts")
    total, start = 0.0, None

    def add(a, b):
        nonlocal total
        a, b = max(a, t0), min(b, now)
        if b > a:
            total += b - a

    for e in ev:
        if e["kind"] == "channel_start":
            if start is not None:
                add(start, _last_second(db, start, e["ts"]))
            start = e["ts"]
        elif start is not None:
            add(start, e["ts"])
            start = None
    if start is not None:
        add(start, _last_second(db, start, now))
    return total


def _last_second(db, a: float, b: float) -> float:
    r = db.query("SELECT max(ts) m FROM band_seconds WHERE ts >= ? AND ts < ?", (int(a), b))[0]["m"]
    return a if r is None else r + 1


def buckets(db, t0: float, step_hz: float = 100e3) -> dict:
    case = " ".join(f"WHEN bw_hz < {e} THEN {i}" for i, e in enumerate(BW_EDGES_HZ))
    rows = db.query(
        f"""SELECT CAST(f_center_hz / ? AS INTEGER) fb, CASE {case} ELSE {len(BW_EDGES_HZ)} END bb,
                   count(*) n, sum(ts_end - ts) air, avg(ts_end - ts) dur, avg(bw_hz) bw,
                   avg(peak_dbfs) peak, max(peak_dbfs) peak_max, avg(above_floor_db) above,
                   sum(in_channel) in_ch, min(ts) first, max(ts) last
            FROM emissions WHERE ts >= ? GROUP BY fb, bb""", (step_hz, t0))
    obs = capture_seconds(db, t0)
    out = []
    for r in rows:
        lo = r["fb"] * step_hz
        out.append({
            "f_lo_hz": lo, "f_hi_hz": lo + step_hz, "bw_bucket": r["bb"], "bw_label": bw_bucket_label(r["bb"]),
            "count": r["n"], "per_hour": r["n"] / (obs / 3600) if obs else None,
            "airtime_s": r["air"], "duty_pct": 100 * r["air"] / obs if obs else None,
            "dur_ms": r["dur"] * 1e3, "bw_khz": r["bw"] / 1e3, "peak_dbfs": r["peak"], "peak_max_dbfs": r["peak_max"],
            "above_db": r["above"], "in_channel": r["in_ch"] > 0, "first": r["first"], "last": r["last"],
        })
    out.sort(key=lambda b: (-b["count"], b["f_lo_hz"], b["bw_bucket"]))
    return {"buckets": out, "observed_s": obs, "step_hz": step_hz,
            "total": sum(b["count"] for b in out)}
