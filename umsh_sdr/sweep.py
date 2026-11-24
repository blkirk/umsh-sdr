"""Wideband survey with rtl_power (900-935 MHz by default).

Covers the whole 902-928 ISM band plus the high-power neighbours most
likely to desense an SX1262 near 917 MHz: 929-932 MHz paging and the
top of 869-894 MHz cellular downlink. Levels are rtl_power's relative dB
(same gain each run, so runs are comparable; not dBm).
"""
from __future__ import annotations

import subprocess
import time

import numpy as np

from .db import DB, f32

KNOWN_SERVICES = [  # (lo MHz, hi MHz, label) annotations for reports
    (869.0, 894.0, "Cellular DL (B5/B26)"),
    (902.0, 928.0, "ISM 902-928"),
    (906.875 - 0.125, 906.875 + 0.125, "Meshtastic US LongFast"),
    (908.42 - 0.05, 908.42 + 0.05, "Z-Wave US"),
    (910.525 - 0.031, 910.525 + 0.031, "MeshCore US"),
    (917.25, 917.75, "UMSH"),
    (929.0, 932.0, "Paging"),
]


def run_rtl_power(start_hz: int, stop_hz: int, bin_hz: int, gain_db: float, ppm: float,
                  interval_s: int, duration_s: float, crop: float = 0.25) -> list[str]:
    cmd = ["rtl_power", "-f", f"{start_hz}:{stop_hz}:{bin_hz}", "-g", str(gain_db), "-p", str(int(round(ppm))),
           "-i", str(interval_s), "-e", f"{int(duration_s)}s", "-c", str(crop)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=duration_s + 120)
    if proc.returncode != 0 and not proc.stdout:
        raise RuntimeError(f"rtl_power failed: {proc.stderr.strip()[-400:]}")
    return proc.stdout.splitlines()


def parse_rtl_power(lines: list[str]):
    """Return (f0, df, spectra[n_sweeps, n_bins]) of complete sweeps.

    A sweep is a run of hops with increasing start frequency; a new sweep
    begins when the start frequency resets. (Timestamps can't be used: one
    sweep's hops often straddle a second boundary.)"""
    sweeps, cur, last_lo = [], [], None
    for ln in lines:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 7:
            continue
        try:
            lo, step = float(parts[2]), float(parts[4])
            vals = [float(v) if v not in ("nan", "-nan") else np.nan for v in parts[6:]]
        except ValueError:
            continue
        if last_lo is not None and lo <= last_lo:
            sweeps.append(cur)
            cur = []
        cur.append((lo, step, vals))
        last_lo = lo
    if cur:
        sweeps.append(cur)
    if not sweeps:
        return None, None, None
    layout = max((tuple((h[0], len(h[2])) for h in sw) for sw in sweeps), key=len)
    rows = [np.concatenate([h[2] for h in sw]) for sw in sweeps
            if tuple((h[0], len(h[2])) for h in sw) == layout]
    first = next(sw for sw in sweeps if tuple((h[0], len(h[2])) for h in sw) == layout)
    step = first[0][1]
    return first[0][0] + step / 2, step, np.vstack(rows)


def survey(db: DB, settings, kind: str = "survey", gain_db: float | None = None, duration_s: float | None = None):
    sc, cal = settings.schedule, settings.calibration
    g = cal.sweep_gain_db if gain_db is None else gain_db
    dur = sc.sweep_minutes * 60 if duration_s is None else duration_s
    t0 = time.time()
    lines = run_rtl_power(sc.sweep_start_hz, sc.sweep_stop_hz, sc.sweep_bin_hz, g, cal.ppm, sc.sweep_interval_s, dur)
    f0, df, S = parse_rtl_power(lines)
    if S is None:
        db.event(t0, "sweep_empty", {"lines": len(lines)})
        return None
    row = {"ts": t0, "ts_end": time.time(), "f0_hz": f0, "df_hz": df, "n": S.shape[1],
           "p50": f32(np.nanpercentile(S, 50, axis=0)), "p90": f32(np.nanpercentile(S, 90, axis=0)),
           "maxhold": f32(np.nanmax(S, axis=0)), "gain_db": g, "kind": kind}
    row["id"] = db.insert("sweeps", row)
    return row, S


def overload_check(db: DB, settings, step_db: float = 10.0, duration_s: float = 60):
    """Sweep at G and G-step, alternating twice (hi, lo, hi, lo). Only bins that are steady
    across the repeats are judged, so bursty traffic (mesh packets, pagers) can't masquerade
    as intermod. Real signals drop ~step dB; 3rd-order products made inside an overloaded
    RTL drop ~3*step. The median drop of steady strong bins is also the tuner's real step."""
    g = settings.calibration.sweep_gain_db
    g_lo = max(0.0, g - step_db)
    part = max(10.0, duration_s / 4)
    runs = {}
    for i, gain in enumerate((g, g_lo, g, g_lo)):
        row, S = survey(db, settings, kind="overload", gain_db=gain, duration_s=part)
        runs[i] = (row, np.nanmedian(S, 0))
    n = min(v[1].size for v in runs.values())
    hi1, lo1, hi2, lo2 = (runs[i][1][:n] for i in range(4))
    hi, lo = (hi1 + hi2) / 2, (lo1 + lo2) / 2
    steady = (np.abs(hi1 - hi2) < 2.0) & (np.abs(lo1 - lo2) < 2.0)
    strong = hi > np.nanpercentile(hi, 20) + 10
    drop = hi - lo
    judged = steady & strong
    flagged = judged & (drop > 2.0 * step_db)
    freqs = runs[0][0]["f0_hz"] + np.arange(n) * runs[0][0]["df_hz"]
    res = {"gain_db": g, "low_gain_db": g_lo, "n_strong": int(strong.sum()), "n_steady_strong": int(judged.sum()),
           "n_flagged": int(flagged.sum()),
           "measured_step_db": float(np.nanmedian(drop[judged])) if judged.any() else None,
           "flagged_mhz": [round(f / 1e6, 3) for f in freqs[flagged]][:200],
           "verdict": (f"inconclusive: only {int(judged.sum())} steady strong bins to judge" if judged.sum() < 10 else
                       "overload: RTL-generated products present; lower sweep gain" if flagged.sum() > 3 else
                       "ok: no RTL intermod detected at this gain")}
    db.event(time.time(), "overload_check", res)
    return res
