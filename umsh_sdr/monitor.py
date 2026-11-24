"""Channel-mode analysis: consumes Frames and DecodedFrames, writes SQLite.

Everything that depends on knowing whether a moment was a UMSH packet runs
``LAG_S`` behind the stream, so packet decodes (which arrive after their
burst ends) can claim their bursts first.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config as C
from . import lora
from .db import DB, f32
from .dsp import Burst, BurstDetector, Emission, EmissionTracker, Frame, FrameAnalyzer, db, undb
from .flowgraph import DecodedFrame
from .loratap import PcapWriter
from .umsh_addr import parse as parse_addr

LAG_S = 8.0
MATCH_TOL_S = 0.04  # true matches measured within +/-10 ms of the expected end
CLIP_INVALID = 1e-3  # a frame with more clipped samples than this is excluded from noise stats
STRONG_DBFS = -12.0  # packets above this make the dongle generate spurs
WARMUP_S = 4.0  # the R820T floor settles for a few seconds after (re)tuning


@dataclass
class _Second:
    frames: list = field(default_factory=list)


class Monitor:
    def __init__(self, db_: DB, settings: C.Settings, publish=None, dissector=None, pcap_dir: Path | None = None):
        self.db = db_
        self.s = settings
        self.cal = settings.calibration
        self.publish = publish or (lambda kind, data: None)
        self.dissector = dissector
        self.q: queue.Queue = queue.Queue(maxsize=20000)
        self.analyzer = FrameAnalyzer()
        fs = self.analyzer.frame_seconds
        self.bursts = BurstDetector(fs)
        self.emit_tracker = EmissionTracker(self.analyzer)
        self.closed_bursts: deque[Burst] = deque(maxlen=400)
        self.umsh_iv: deque[tuple[float, float]] = deque(maxlen=2000)
        self.pending_pkts: list[DecodedFrame] = []
        self.pending_emis: list[Emission] = []
        self.frames: deque[Frame] = deque()
        self.ref_floor = {b: deque(maxlen=60) for b in C.BANDS}
        self.minute = None
        self.min_acc = None
        self.latest_t = 0.0
        self.t_first = None
        self.last_drift_note = 0.0
        self.wf_acc = None
        self.wf_n = 0
        self.pcap_dir = Path(pcap_dir or Path(settings.data_dir) / "pcap")
        self._pcap = None
        self._pcap_day = None
        self.stats = {"frames": 0, "packets": 0, "crc_ok": 0, "matched": 0, "emissions": 0, "dropped": 0}
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="monitor", daemon=True)
        self.realtime = True

    # --- producer side (GNU Radio threads) ---
    def on_frame(self, f: Frame):
        try:
            self.q.put_nowait(("f", f))
        except queue.Full:
            self.stats["dropped"] += 1

    def on_packet(self, p: DecodedFrame):
        self.q.put(("p", p))

    # --- lifecycle ---
    def start(self):
        self.thread.start()

    def stop(self, flush=True):
        self._stop.set()
        self.thread.join(timeout=30)
        if flush:
            self._flush_all()
        if self._pcap:
            self._pcap.close()
            self._pcap = None

    def drain(self, timeout=60.0):
        """Wait until queued items are processed (used for file replay)."""
        t_end = time.time() + timeout
        while not self.q.empty() and time.time() < t_end:
            time.sleep(0.05)

    def _run(self):
        while not self._stop.is_set() or not self.q.empty():
            try:
                kind, item = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if kind == "f":
                    self._frame(item)
                else:
                    self.pending_pkts.append(item)
                    self.stats["packets"] += 1
                    self._match_packets(final=False)
            except Exception as e:  # keep monitoring; record the failure
                self.db.event(time.time(), "monitor_error", repr(e))

    # --- frame path ---
    def _frame(self, f: Frame):
        self.stats["frames"] += 1
        self.latest_t = f.t
        if self.t_first is None:
            self.t_first = f.t
        if f.t - self.t_first < WARMUP_S:
            self._waterfall(f)
            return
        if self.realtime and f.t - self.last_drift_note > 60:
            drift = time.time() - f.t
            if abs(drift) > 2.0:
                self.db.event(time.time(), "timebase_drift", {"seconds": round(drift, 2)})
                self.last_drift_note = f.t
        f.valid = f.clip <= CLIP_INVALID
        self.frames.append(f)
        b = self.bursts.push(f.t, f.band["channel"], f.clip)
        if b is not None:
            self.closed_bursts.append(b)
        if f.valid:  # a clipped frame's spectrum is full of ADC splatter / intermod
            for e in self.emit_tracker.push(f):
                self.pending_emis.append(e)
        self._waterfall(f)
        self._match_packets(final=False)
        self._finalize(f.t - LAG_S)

    def _waterfall(self, f: Frame):
        self.wf_acc = f.psd.copy() if self.wf_acc is None else np.maximum(self.wf_acc, f.psd)
        self.wf_n += 1
        if self.wf_n >= 10:
            self.publish("wf", {"t": f.t, "row": np.round(db(self.wf_acc), 1).tolist(),
                                "band": {k: round(v, 1) for k, v in f.band.items()}})
            self.wf_acc, self.wf_n = None, 0

    # --- packets ---
    # frame_sync's header tag lands ~HDR_LAG_SYM symbols after the preamble ends
    # (sync word 4.25 symbols + measured ~7 symbols of pipeline lag).
    HDR_LAG_SYM = 11.25

    def _expected_end(self, p: DecodedFrame) -> float | None:
        if p.t_header is None:
            return None
        ts = lora.symbol_time()
        return p.t_header - self.HDR_LAG_SYM * ts + lora.airtime(len(p.payload), preamble=0) + 4.25 * ts

    def _match_packets(self, final: bool):
        still = []
        for p in self.pending_pkts:
            burst = self._find_burst(p)
            end = self._expected_end(p)
            waited = (self.latest_t > end + 0.4) if end is not None else (time.time() - p.decoded_at > 4.0)
            if burst is not None or final or waited:
                self._store_packet(p, burst)
            else:
                still.append(p)
        self.pending_pkts = still

    def _find_burst(self, p: DecodedFrame) -> Burst | None:
        n = len(p.payload)
        lo = lora.airtime(n, preamble=8) - 0.02   # bursts run ~1 frame long
        hi = lora.airtime(n, preamble=34) + 0.035
        end = self._expected_end(p)
        best, best_err = None, None
        for b in self.closed_bursts:
            if b.claimed_by is not None or not (lo <= b.duration <= hi):
                continue
            err = abs(b.t_end - end) if end is not None else abs(time.time() - b.t_end)
            if end is not None and err > MATCH_TOL_S:
                continue
            if best is None or err < best_err:
                best, best_err = b, err
        return best

    def _window_power(self, t0: float, t1: float):
        """Channel power over [t0, t1] and the floor just before, from retained frames."""
        body = [f.band["channel"] for f in self.frames if t0 + 0.02 <= f.t <= t1 - 0.03]
        busy = self._umsh_intervals(t0 - 0.7, t0)
        pre = [f.band["channel"] for f in self.frames if t0 - 0.6 <= f.t <= t0 - 0.05
               and not any(a <= f.t <= b for a, b in busy)]
        if not body:
            return None, None
        # back-to-back mesh repeats can leave no quiet frames before a packet; use the rolling floor
        noise = float(np.median(pre)) if pre else self.bursts.floor()
        return float(db(np.mean(undb(body)))), noise

    def _store_packet(self, p: DecodedFrame, b: Burst | None):
        n = len(p.payload)
        row = {
            "decoded_at": p.decoded_at, "pay_len": n, "crc_ok": int(p.crc_ok), "cr": p.cr,
            "payload": p.payload, "cfo_hz": p.cfo_hz,
            # cfo_hz > 0: transmitter's carrier sits above the SDR's (ppm-corrected) tuning
            "cfo_ppm": None if p.cfo_hz is None else p.cfo_hz / C.CHANNEL_HZ * 1e6,
            "gain_db": self.cal.gain_db, "matched": int(b is not None),
        }
        if b is not None:
            t0, t1 = b.t_start, b.t_end
            rssi, noise, clip = b.power_dbfs, b.floor_dbfs, b.clip
            row["airtime_ms"] = b.duration * 1e3
            self.stats["matched"] += 1
        else:
            end = self._expected_end(p)
            if end is None:
                end = p.decoded_at
            t1, t0 = end, end - lora.airtime(n)
            rssi, noise = self._window_power(t0, t1)
            clip = None
        row.update(ts=t0, ts_end=t1, rssi_dbfs=rssi, noise_dbfs=noise, clipped=clip)
        snr_pow = None
        if rssi is not None and noise is not None and undb(rssi) > undb(noise) * 1.05:
            snr_pow = float(db((undb(rssi) - undb(noise)) / undb(noise)))
        row["snr_power_db"] = snr_pow
        # Preamble SNR is right for weak frames but saturates (and clipping distorts it) on
        # strong ones, where the power ratio is reliable.
        if snr_pow is not None and (snr_pow >= 10.0 or p.snr_preamble_db is None):
            row["snr_db"] = snr_pow
        else:
            row["snr_db"] = p.snr_preamble_db
        if self.cal.dbm_offset is not None and rssi is not None:
            row["rssi_dbm"] = rssi + self.cal.dbm_offset
        pid = self.db.insert("packets", row)
        if b is not None:
            b.claimed_by = pid
        strong = (clip or 0) > 0 or (rssi is not None and rssi > STRONG_DBFS)
        self.umsh_iv.append((t0, t1, strong))
        if p.crc_ok:
            self.stats["crc_ok"] += 1
            self._write_pcap(row)
            if self.dissector is not None:
                self.dissector.submit(pid, row["ts"], p.payload, row.get("rssi_dbm"), row.get("snr_db"))
        pub = {k: v for k, v in row.items() if k != "payload"}
        pub.update(id=pid, payload_hex=p.payload.hex(), addr=parse_addr(p.payload) if p.crc_ok else None)
        self.publish("packet", pub)

    def _write_pcap(self, row):
        day = time.strftime("%Y%m%d", time.localtime(row["ts"]))
        if day != self._pcap_day:
            if self._pcap:
                self._pcap.close()
            self._pcap = PcapWriter(self.pcap_dir / f"sdr-{day}.pcap")
            self._pcap_day = day
        self._pcap.write_lora(row["ts"], row["payload"], freq_hz=C.CHANNEL_HZ, bw_hz=C.BW_HZ, sf=C.SF,
                              rssi_dbm=row.get("rssi_dbm"), snr_db=row.get("snr_db"), sync_word=C.SYNC_WORD)

    # --- lagged aggregation ---
    def _umsh_intervals(self, t_lo, t_hi):
        pad = 0.03
        return [(a - pad, b + pad) for a, b, *_ in self.umsh_iv if b + pad >= t_lo and a - pad <= t_hi]

    def _strong_intervals(self, t_lo, t_hi):
        pad = 0.05
        return [(a - pad, b + pad) for a, b, strong in self.umsh_iv if strong and b + pad >= t_lo and a - pad <= t_hi]

    def _finalize(self, horizon: float):
        """Aggregate every whole second that ended before ``horizon``."""
        cut = float("inf") if horizon == float("inf") else float(int(horizon))
        if not self.frames or self.frames[0].t >= cut:
            return
        batch = []
        while self.frames and int(self.frames[0].t) < cut:
            batch.append(self.frames.popleft())
        iv = self._umsh_intervals(batch[0].t, batch[-1].t + 1)
        for f in batch:
            f.umsh = any(a <= f.t <= b for a, b in iv)
        self._emissions_ready(horizon)
        # group by second
        by_sec: dict[int, list[Frame]] = {}
        for f in batch:
            by_sec.setdefault(int(f.t), []).append(f)
        for sec, frames in by_sec.items():
            self._second(sec, frames)
            self._minute(sec, frames)

    def _second(self, sec: int, frames: list[Frame]):
        rows = []
        clip = float(np.mean([f.clip for f in frames]))
        good = [f for f in frames if f.valid and not f.umsh]
        n = len(frames)
        for band in C.BANDS:
            vals = np.array([f.band[band] for f in good]) if good else np.array([])
            if vals.size:
                floor = float(np.percentile(vals, 10))
                ref = float(np.median(self.ref_floor[band])) if self.ref_floor[band] else floor
                rows.append((sec, band, floor, float(np.median(vals)), float(vals.max()),
                             float(db(np.mean(undb(vals)))), float(np.mean(vals > ref + 6.0)) * len(good) / n,
                             float(np.mean([f.umsh for f in frames])) if band == "channel" else None,
                             clip, self.cal.gain_db))
                self.ref_floor[band].append(floor)
            else:
                rows.append((sec, band, None, None, None, None, None,
                             float(np.mean([f.umsh for f in frames])) if band == "channel" else None,
                             clip, self.cal.gain_db))
        self.db.executemany("INSERT OR REPLACE INTO band_seconds VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        self.publish("second", {"ts": sec, "bands": {r[1]: {"floor": r[2], "median": r[3], "max": r[4],
                                                           "busy_other": r[6], "busy_umsh": r[7]} for r in rows},
                                "clip": clip, "stats": dict(self.stats)})

    def _minute(self, sec: int, frames: list[Frame]):
        minute = sec - sec % 60
        if self.minute is not None and minute != self.minute:
            self._write_minute()
        if self.minute != minute:
            self.minute = minute
            self.min_acc = {"sum": None, "n": 0, "max": None, "secs": []}
        a = self.min_acc
        good = [f.psd for f in frames if f.valid and not f.umsh]
        allp = np.stack([f.psd for f in frames])
        a["max"] = allp.max(0) if a["max"] is None else np.maximum(a["max"], allp.max(0))
        if good:
            g = np.stack(good)
            a["sum"] = g.sum(0) if a["sum"] is None else a["sum"] + g.sum(0)
            a["n"] += len(good)
            a["secs"].append(g.mean(0))

    def _write_minute(self):
        a = self.min_acc
        if not a or a["n"] == 0:
            return
        mean = a["sum"] / a["n"]
        p10 = np.percentile(np.stack(a["secs"]), 10, axis=0)
        fr = self.analyzer.freqs
        self.db.execute("INSERT OR REPLACE INTO spectra VALUES (?,?,?,?,?,?,?)",
                        (self.minute, float(fr[0]), float(self.analyzer.df), f32(db(mean)), f32(db(a["max"])),
                         f32(db(p10)), self.cal.gain_db))

    def _emissions_ready(self, horizon: float):
        keep, rows = [], []
        ch_lo, ch_hi = C.BANDS["channel"]
        for e in self.pending_emis:
            if e.t_end > horizon:
                keep.append(e)
                continue
            in_ch = e.f_lo < ch_hi and e.f_hi > ch_lo
            iv = self._umsh_intervals(e.t_start, e.t_end)
            overlap = sum(max(0.0, min(b, e.t_end) - max(a, e.t_start)) for a, b in iv)
            is_umsh = in_ch and overlap > 0.5 * (e.t_end - e.t_start)
            if is_umsh:
                continue  # UMSH packets live in the packets table
            # Out-of-channel "emissions" during a strong (often clipped) local packet are
            # spurs/intermod made inside the dongle; keep them out of the interference log.
            if any(a <= e.t_start and e.t_end <= b for a, b in self._strong_intervals(e.t_start, e.t_end)):
                self.stats["artifacts"] = self.stats.get("artifacts", 0) + 1
                continue
            rows.append((e.t_start, e.t_end, e.f_center, e.bw, e.peak_dbfs, e.above_db, e.cls(), int(in_ch), 0))
        self.pending_emis = keep
        if rows:
            self.db.executemany("INSERT INTO emissions (ts, ts_end, f_center_hz, bw_hz, peak_dbfs, above_floor_db,"
                                " cls, in_channel, umsh) VALUES (?,?,?,?,?,?,?,?,?)", rows)
            self.stats["emissions"] += len(rows)
            for r in rows:
                self.publish("emission", {"ts": r[0], "dur": r[1] - r[0], "f": r[2], "bw": r[3],
                                          "peak": r[4], "above": r[5], "cls": r[6], "in_channel": r[7]})

    def _flush_all(self):
        self._match_packets(final=True)
        self.pending_emis.extend(self.emit_tracker.flush())
        self._finalize(float("inf"))
        self._write_minute()
