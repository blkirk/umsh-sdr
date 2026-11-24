"""Calibration with the test node (umshctl ping).

ppm        median CFO of decoded frames -> crystal correction (new = old - cfo/f)
gain       per-gain decode count, channel floor and background clipping
level      dBFS -> dBm offset from the node's own RSSI for the ping replies it hears
selfnoise  terminated-input floor (fit a 50-ohm load on the SDR first)
"""
from __future__ import annotations

import re
import statistics
import subprocess
import threading
import time

from . import config as C
from .db import DB

REPLY_RE = re.compile(r"seq (\d+): reply in ([\d.]+) s, .*?(-?\d+(?:\.\d+)?) dBm, snr (-?[\d.]+) dB")


class Pinger:
    def __init__(self, settings, count: int, interval: float = 3.0):
        if not settings.ping_target:
            raise SystemExit("set ping_target (a node key that answers `umshctl ping`) in config.toml")
        self.cmd = [settings.umshctl, "-p", settings.node_port, "ping", "-c", str(count), "-i", str(int(interval)),
                    settings.ping_target]
        self.replies: list[dict] = []
        self.lines: list[str] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self, *_):
        time.sleep(2.0)  # let the flowgraph settle first
        self.thread.start()

    def _run(self):
        proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            self.lines.append(line.rstrip())
            m = REPLY_RE.search(line)
            if m:
                self.replies.append({"t": time.time(), "seq": int(m[1]), "rtt": float(m[2]),
                                     "rssi_dbm": float(m[3]), "snr_db": float(m[4])})
        proc.wait()

    def join(self, timeout=None):
        if self.thread.is_alive():
            self.thread.join(timeout)


def _session_with_pings(settings, db, pings: int, extra_s: float = 12.0, gain: float | None = None):
    from .__main__ import channel_session
    if gain is not None:
        settings.calibration.gain_db = gain
    pinger = Pinger(settings, pings)
    t0 = time.time()
    channel_session(settings, db, 2.0 + pings * 3.0 + extra_s, on_start=lambda fg, m: threading.Thread(
        target=pinger.start, daemon=True).start())
    pinger.join(10)
    return t0, time.time(), pinger


def _pkts(db, t0, t1):
    return db.query("SELECT * FROM packets WHERE decoded_at BETWEEN ? AND ? ORDER BY ts", (t0, t1))


def cal_ppm(a, s, db):
    t0, t1, pinger = _session_with_pings(s, db, a.pings)
    cfos = [r["cfo_hz"] for r in _pkts(db, t0, t1) if r["crc_ok"] and r["cfo_hz"] is not None]
    if len(cfos) < 3:
        print(f"only {len(cfos)} decoded frames; check antenna/gain. ping said: {pinger.lines[-3:]}")
        return
    med = statistics.median(cfos)
    q = statistics.quantiles(cfos, n=4)
    old = s.calibration.ppm
    new = round(old - med / C.CHANNEL_HZ * 1e6, 2)
    print(f"{len(cfos)} frames: median CFO {med:+.0f} Hz (IQR {q[0]:+.0f}..{q[2]:+.0f}); "
          f"transmitter spread is real crystal variation between nodes")
    print(f"ppm correction {old} -> {new}")
    return {"ppm": new}


def cal_gain(a, s, db):
    gains = [float(g) for g in a.gains.split(",")]
    rows = []
    for g in gains:
        t0, t1, pinger = _session_with_pings(s, db, a.pings, gain=g)
        pk = _pkts(db, t0, t1)
        bs = db.query("SELECT avg(floor_dbfs) fl, avg(clip) clip FROM band_seconds WHERE band='channel' "
                      "AND ts BETWEEN ? AND ? AND busy_umsh = 0", (int(t0) + 3, int(t1)))
        rows.append({"gain": g, "decoded": len(pk), "crc_ok": sum(r["crc_ok"] for r in pk),
                     "replies": len(pinger.replies), "floor": bs[0]["fl"], "bg_clip": bs[0]["clip"] or 0.0})
        r = rows[-1]
        print(f"gain {g:5.1f}: crc_ok {r['crc_ok']:3d}/{r['decoded']:3d}  replies {r['replies']}/{a.pings}  "
              f"floor {r['floor'] if r['floor'] is None else round(r['floor'], 1)} dBFS/500k  "
              f"background clip {r['bg_clip']:.1e}", flush=True)
    best_ok = max(r["crc_ok"] for r in rows) or 1
    ok = [r for r in rows if r["bg_clip"] < 1e-4 and r["crc_ok"] >= 0.9 * best_ok]
    if not ok:
        print("no gain met both criteria (background clip < 1e-4, >=90% of best decode count)")
        return
    pick = max(ok, key=lambda r: r["gain"])
    for lo, hi in zip(rows, rows[1:]):
        if lo["floor"] is not None and hi["floor"] is not None:
            slope = (hi["floor"] - lo["floor"]) / (hi["gain"] - lo["gain"])
            print(f"  floor slope {lo['gain']}->{hi['gain']}: {slope:.2f} dB/dB "
                  f"({'external noise visible' if slope > 0.7 else 'mostly SDR self-noise'})")
    print(f"recommended channel gain: {pick['gain']} dB")
    return {"gain_db": pick["gain"], "sweep_gain_db": pick["gain"]}


def cal_level(a, s, db):
    cal = s.calibration
    target = cal.gain_db
    candidates = sorted([float(g) for g in a.gains.split(",")] + [20.7, 16.6, 12.5, 8.7], reverse=True)
    for g in [g for g in candidates if g <= target]:
        t0, t1, pinger = _session_with_pings(s, db, a.pings, gain=g)
        pk = [r for r in _pkts(db, t0, t1) if r["matched"] and r["crc_ok"]]
        pairs = []
        for rep in pinger.replies:
            near = [r for r in pk if rep["t"] - 1.5 <= r["ts_end"] <= rep["t"]]
            if near:
                r = max(near, key=lambda r: r["ts_end"])
                if (r["clipped"] or 0) < 1e-3 and r["rssi_dbfs"] < -6:
                    pairs.append(rep["rssi_dbm"] - r["rssi_dbfs"])
        print(f"gain {g}: {len(pairs)} unclipped reply pairs {[round(p, 1) for p in pairs]}", flush=True)
        if len(pairs) >= 3:
            off = statistics.median(pairs)
            at_target = off - (target - g)
            print(f"offset at {g} dB: {off:+.1f} dB (spread {min(pairs) - off:+.1f}/{max(pairs) - off:+.1f}); "
                  f"carried to channel gain {target} dB using the nominal tuner step: {at_target:+.1f} dB")
            print("note: node-referenced (its antenna, its RSSI calibration): treat dBm as approximate, +/-6 dB. "
                  "Desense uses the self-noise excess instead (calibrate selfnoise).")
            return {"dbm_offset": round(at_target, 1)}
    print("could not find a gain where ping replies are unclipped; move the SDR away from the node or add attenuation")


def cal_selfnoise(a, s, db):
    from .__main__ import channel_session
    input("Fit a 50-ohm terminator on the SDR input (no antenna), then press Enter...")
    t0 = time.time()
    channel_session(s, db, a.seconds)
    out = {}
    for band in C.BANDS:
        r = db.query("SELECT avg(floor_dbfs) f FROM band_seconds WHERE band=? AND ts BETWEEN ? AND ?",
                     (band, int(t0) + 3, int(time.time())))
        if r and r[0]["f"] is not None:
            out[band] = round(r[0]["f"], 2)
    print(f"self-noise floor at gain {s.calibration.gain_db} dB (dBFS/500k): {out}")
    print("Refit the antenna before running the monitor.")
    return {"self_floor_dbfs": out}


def main(a, settings):
    db = DB(settings.db_path)
    fn = {"ppm": cal_ppm, "gain": cal_gain, "level": cal_level, "selfnoise": cal_selfnoise}[a.what]
    saved_gain = settings.calibration.gain_db
    res = fn(a, settings, db)
    settings.calibration.gain_db = saved_gain
    db.event(time.time(), f"calibrate_{a.what}", res)
    if res and a.apply:
        for k, v in res.items():
            setattr(settings.calibration, k, v)
        C.save(settings)
        print(f"saved to {C.CONFIG_PATH}: {res}")
    elif res:
        print("(dry run; add --apply to save)")
