"""Hardware-independent signal analysis (pure numpy, unit-tested).

Frame:      10.24 ms averaged periodogram (dBFS/bin) plus band powers.
BurstDetector: on/off detection of energy in one band (UMSH packet bursts).
EmissionTracker: finds and tracks any emission across the 2 MHz window and
classifies it by bandwidth and duration.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import config as C

DC_GUARD_HZ = 6_000
EDGE_GUARD_HZ = 80_000


def db(x):
    return 10.0 * np.log10(np.maximum(x, 1e-20))


def undb(x):
    return 10.0 ** (np.asarray(x) / 10.0)


@dataclass
class Frame:
    t: float                    # frame start (unix s)
    psd: np.ndarray             # linear power per bin (fraction of full scale), fftshifted
    band: dict[str, float]      # dBFS per 500 kHz, per band
    clip: float                 # fraction of samples at ADC rails
    # filled in later by the monitor:
    umsh: bool = False
    valid: bool = True


class FrameAnalyzer:
    """Turns a stream of complex samples into Frames."""

    def __init__(self, sample_rate=C.SAMPLE_RATE, lo_hz=C.LO_HZ, fft_size=C.FFT_SIZE,
                 frame_ffts=C.FRAME_FFTS, bands=None):
        self.sr, self.lo, self.n, self.k = sample_rate, lo_hz, fft_size, frame_ffts
        self.win = np.hanning(fft_size).astype(np.float32)
        self.norm = fft_size * float(np.sum(self.win ** 2))
        self.freqs = lo_hz + np.fft.fftshift(np.fft.fftfreq(fft_size, 1.0 / sample_rate))
        self.df = sample_rate / fft_size
        rel = self.freqs - lo_hz
        self.usable = (np.abs(rel) > DC_GUARD_HZ) & (np.abs(rel) < sample_rate / 2 - EDGE_GUARD_HZ)
        self.band_masks = {name: (self.freqs >= lo) & (self.freqs < hi) & self.usable
                           for name, (lo, hi) in (bands or C.BANDS).items()}
        # Bands differ in width; report each as power per 500 kHz (the UMSH channel width)
        # so floors are directly comparable.
        self.band_norm = {name: float(C.BW_HZ / (m.sum() * self.df)) for name, m in self.band_masks.items()}
        self.frame_len = fft_size * frame_ffts

    @property
    def frame_seconds(self) -> float:
        return self.frame_len / self.sr

    def frame_from(self, x: np.ndarray, t: float) -> Frame:
        blocks = x[: self.frame_len].reshape(self.k, self.n) * self.win
        p = np.abs(np.fft.fft(blocks, axis=1)) ** 2
        psd = np.fft.fftshift(p.mean(axis=0)) / self.norm
        band = {name: float(db(psd[m].sum() * self.band_norm[name])) for name, m in self.band_masks.items()}
        clip = float(np.mean((np.abs(x.real) >= 0.99) | (np.abs(x.imag) >= 0.99)))
        return Frame(t, psd.astype(np.float32), band, clip)


class StreamFramer:
    """Accumulates samples and emits Frames with sample-accurate timestamps."""

    def __init__(self, analyzer: FrameAnalyzer, t0: float):
        self.a = analyzer
        self.t0 = t0
        self.n_done = 0  # samples consumed into frames
        self._buf = np.zeros(0, dtype=np.complex64)

    def feed(self, x: np.ndarray) -> list[Frame]:
        self._buf = np.concatenate([self._buf, x]) if self._buf.size else np.asarray(x, np.complex64)
        out, L = [], self.a.frame_len
        nfull = self._buf.size // L
        for i in range(nfull):
            t = self.t0 + (self.n_done + i * L) / self.a.sr
            out.append(self.a.frame_from(self._buf[i * L:(i + 1) * L], t))
        if nfull:
            self._buf = self._buf[nfull * L:].copy()
            self.n_done += nfull * L
        return out

    def now(self) -> float:
        return self.t0 + self.n_done / self.a.sr


@dataclass
class Burst:
    t_start: float
    t_end: float
    power_dbfs: float      # linear mean over the burst body
    floor_dbfs: float      # band floor just before the burst
    clip: float
    claimed_by: int | None = None

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


class BurstDetector:
    """Energy bursts in one band: threshold = rolling p10 floor + ``rise_db``."""

    def __init__(self, frame_s: float, rise_db=6.0, floor_window_s=10.0, start_frames=2, end_frames=3):
        self.frame_s = frame_s
        self.rise = rise_db
        self.hist = deque(maxlen=max(10, int(floor_window_s / frame_s)))
        self.recent = deque(maxlen=max(5, int(1.0 / frame_s)))
        self.start_n, self.end_n = start_frames, end_frames
        self.active: list[tuple[float, float, float]] | None = None  # (t, dbfs, clip)
        self.above = 0
        self.below = 0
        self.pending: list[tuple[float, float, float]] = []
        self.floor_at_start = None

    def floor(self) -> float | None:
        if len(self.hist) < 5:
            return None
        return float(np.percentile(np.fromiter(self.hist, float), 10))

    def push(self, t: float, dbfs: float, clip: float = 0.0) -> Burst | None:
        fl = self.floor()
        hot = fl is not None and dbfs > fl + self.rise
        if self.active is None:
            if hot:
                self.pending.append((t, dbfs, clip))
                if len(self.pending) >= self.start_n:
                    self.active = list(self.pending)
                    self.floor_at_start = float(np.median(np.fromiter(self.recent, float))) if self.recent else fl
                    self.pending = []
                    self.below = 0
                return None
            for p in self.pending:  # false start: these were background after all
                self.hist.append(p[1])
            self.pending = []
            self.hist.append(dbfs)
            self.recent.append(dbfs)
            return None
        # inside a burst
        self.active.append((t, dbfs, clip))
        self.below = 0 if hot else self.below + 1
        if self.below < self.end_n:
            return None
        frames = self.active[: -self.end_n]
        self.active = None
        self.below = 0
        body = frames[1:-1] if len(frames) > 3 else frames
        p = float(db(np.mean(undb([f[1] for f in body]))))
        return Burst(frames[0][0], frames[-1][0] + self.frame_s, p, self.floor_at_start,
                     float(np.mean([f[2] for f in frames])))


@dataclass
class Emission:
    t_start: float
    t_last: float
    f_lo: float
    f_hi: float
    peak_dbfs: float
    above_db: float
    frame_s: float
    seen: int = 1

    @property
    def f_center(self):
        return 0.5 * (self.f_lo + self.f_hi)

    @property
    def bw(self):
        return self.f_hi - self.f_lo

    @property
    def t_end(self):
        return self.t_last + self.frame_s

    def cls(self) -> str:
        if self.t_end - self.t_start >= 10.0:
            return "continuous"
        if self.bw < 60_000:
            return "narrowband"
        if self.bw <= 600_000:
            return "medium"
        return "wideband"


class EmissionTracker:
    """Per-bin floor (p20 of 1 s means over the last minute), 10 dB detection
    threshold, contiguous-bin grouping, frame-to-frame tracking."""

    def __init__(self, analyzer: FrameAnalyzer, threshold_db=10.0, gap_frames=3,
                 floor_window_s=60, max_duration_s=60.0):
        self.a = analyzer
        self.thr = undb(threshold_db)
        self.gap_s = gap_frames * analyzer.frame_seconds
        self.max_dur = max_duration_s
        self.sec_means = deque(maxlen=floor_window_s)
        self._acc = None
        self._acc_n = 0
        self._acc_sec = None
        self._floor = None
        self.active: list[Emission] = []

    def _update_floor(self, f: Frame):
        sec = int(f.t)
        if self._acc_sec is None:
            self._acc_sec = sec
        if sec != self._acc_sec and self._acc_n:
            self.sec_means.append(self._acc / self._acc_n)
            self._acc, self._acc_n, self._acc_sec = None, 0, sec
            stack = np.stack(self.sec_means)
            self._floor = np.percentile(stack, 20, axis=0) if len(stack) >= 3 else np.median(stack, axis=0)
        self._acc = f.psd.astype(np.float64) if self._acc is None else self._acc + f.psd
        self._acc_n += 1

    def floor(self) -> np.ndarray | None:
        return self._floor

    def push(self, f: Frame) -> list[Emission]:
        self._update_floor(f)
        closed = []
        if self._floor is None:
            return closed
        hot = (f.psd > self._floor * self.thr) & self.a.usable
        runs = _runs(hot, merge_gap=2)
        df = self.a.df
        seen_now = set()
        for i0, i1 in runs:
            seg = f.psd[i0:i1 + 1]
            p = float(db(seg.sum()))
            above = float(db(seg.sum() / self._floor[i0:i1 + 1].sum()))
            lo, hi = self.a.freqs[i0] - df / 2, self.a.freqs[i1] + df / 2
            match = None
            for k, e in enumerate(self.active):
                if lo <= e.f_hi + df and hi >= e.f_lo - df:
                    match = k
                    break
            if match is None:
                self.active.append(Emission(f.t, f.t, lo, hi, p, above, self.a.frame_seconds))
                seen_now.add(len(self.active) - 1)
            else:
                e = self.active[match]
                e.f_lo, e.f_hi = min(e.f_lo, lo), max(e.f_hi, hi)
                e.t_last = f.t
                e.seen += 1
                if p > e.peak_dbfs:
                    e.peak_dbfs, e.above_db = p, above
                seen_now.add(match)
        keep = []
        for k, e in enumerate(self.active):
            if f.t - e.t_last > self.gap_s or (f.t - e.t_start) >= self.max_dur:
                closed.append(e)
            else:
                keep.append(e)
        self.active = keep
        return closed

    def flush(self) -> list[Emission]:
        out, self.active = self.active, []
        return out


def _runs(mask: np.ndarray, merge_gap: int = 0) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > merge_gap + 1:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    return runs
