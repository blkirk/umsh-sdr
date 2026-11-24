import numpy as np
import pytest

from umsh_sdr import config as C
from umsh_sdr.dsp import BurstDetector, EmissionTracker, FrameAnalyzer, StreamFramer, db

SR = C.SAMPLE_RATE
RNG = np.random.default_rng(7)


def noise(n, power_dbfs):
    s = np.sqrt(10 ** (power_dbfs / 10) / 2)
    return (RNG.normal(0, s, n) + 1j * RNG.normal(0, s, n)).astype(np.complex64)


def tone(n, f_off, power_dbfs, t0=0):
    a = np.sqrt(10 ** (power_dbfs / 10))
    t = (np.arange(n) + t0) / SR
    return (a * np.exp(2j * np.pi * f_off * t)).astype(np.complex64)


def test_band_power_is_calibrated_per_500k():
    a = FrameAnalyzer()
    f = a.frame_from(noise(a.frame_len, -30.0), 0.0)
    # white noise over 2 MHz at -30 dBFS total -> -36 dBFS in any 500 kHz
    for band in C.BANDS:
        assert f.band[band] == pytest.approx(-30 - 10 * np.log10(4), abs=0.5)


def test_tone_power_lands_in_its_band():
    a = FrameAnalyzer()
    x = noise(a.frame_len, -60) + tone(a.frame_len, 500e3 + 100e3, -20)  # inside the UMSH channel
    f = a.frame_from(x, 0.0)
    assert f.band["channel"] == pytest.approx(-20, abs=0.5)
    assert f.band["adj_high"] < -50


def test_stream_framer_timestamps():
    a = FrameAnalyzer()
    sf = StreamFramer(a, t0=100.0)
    out = sf.feed(noise(a.frame_len * 2 + 10, -40)) + sf.feed(noise(a.frame_len, -40))
    assert [round(f.t - 100.0, 5) for f in out] == [0, round(a.frame_seconds, 5), round(2 * a.frame_seconds, 5)]


def test_burst_detector_finds_packet_length():
    fs = 0.01024
    d = BurstDetector(fs)
    bursts = []
    for i in range(1000):
        t = i * fs
        p = -45 + RNG.normal(0, 0.3)
        if 5.0 <= t < 5.18:
            p = -20
        b = d.push(t, p)
        if b:
            bursts.append(b)
    assert len(bursts) == 1
    assert bursts[0].duration == pytest.approx(0.18, abs=0.025)
    assert bursts[0].floor_dbfs == pytest.approx(-45, abs=1)
    assert bursts[0].power_dbfs == pytest.approx(-20, abs=0.5)


def test_emission_tracker_quiet_on_pure_noise():
    a = FrameAnalyzer()
    tr = EmissionTracker(a)
    sf = StreamFramer(a, 0.0)
    found = []
    for _ in range(40):  # ~20 s
        for f in sf.feed(noise(SR // 2, -40)):
            found += tr.push(f)
    found += tr.flush()
    assert found == []


def test_emission_tracker_classifies_narrowband_hop():
    a = FrameAnalyzer()
    tr = EmissionTracker(a)
    sf = StreamFramer(a, 0.0)
    found = []
    n = 0
    for k in range(30):  # 15 s
        x = noise(SR // 2, -40)
        if k == 20:  # a 30 ms hop at 916.6 MHz, 20 dB above the per-bin floor
            i0 = 100_000
            x[i0:i0 + 60_000] += tone(60_000, -400e3, -40 + 20 - 10 * np.log10(SR / a.df) + 10, n + i0)
        for f in sf.feed(x):
            found += tr.push(f)
        n += x.size
    found += tr.flush()
    assert len(found) == 1
    e = found[0]
    assert e.cls() == "narrowband"
    assert e.f_center == pytest.approx(C.LO_HZ - 400e3, abs=3e3)
    assert 0.02 <= e.t_end - e.t_start <= 0.07


def test_rtl_power_sweep_straddling_a_second():
    from umsh_sdr.sweep import parse_rtl_power
    lines = []
    for k in range(3):  # three sweeps, each with hops split across two timestamps
        for i, lo in enumerate((900e6, 901e6, 902e6)):
            t = f"12:00:{2 * k + (i >= 2):02d}"
            lines.append(f"2026-11-23, {t}, {lo:.0f}, {lo + 1e6:.0f}, 250000.0, 10, {k}, {k}, {k}, {k}")
    f0, df, S = parse_rtl_power(lines)
    assert S.shape == (3, 12)
    assert f0 == 900e6 + 125000 and df == 250000
    assert list(S[:, 0]) == [0, 1, 2]


def test_desense_from_excess():
    from umsh_sdr.report import desense_from_excess
    assert desense_from_excess(0.0) == pytest.approx(0.0)
    assert desense_from_excess(-1.0) == pytest.approx(0.0)   # below self-noise is measurement scatter
    assert desense_from_excess(3.0) == pytest.approx(3.0, abs=0.05)  # equal NFs: desense ~= excess
    assert desense_from_excess(10.0) == pytest.approx(10.0, abs=0.05)
