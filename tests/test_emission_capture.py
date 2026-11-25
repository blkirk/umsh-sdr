"""Emission capture: plan geometry, the analysis on a synthetic rtl_sdr file, the classifier,
and the scheduler's capture job (pause channel capture, record, resume)."""
import json
import threading
import time

import numpy as np
import pytest

from umsh_sdr import __main__ as M
from umsh_sdr import config as C
from umsh_sdr import emission_capture as EC
from umsh_sdr import iq_analysis as A
from umsh_sdr.dashboard import app as APP

RNG = np.random.default_rng(7)


def _bits(h):
    return "".join(bin(int(c, 16))[2:].zfill(4) for c in h)


def _fsk(bits, fs, baud, dev):
    sps = fs / baud
    idx = (np.arange(int(len(bits) * sps)) / sps).astype(int)
    f = np.where(np.array(list(bits))[idx] == "1", dev, -dev)
    return np.exp(2j * np.pi * np.cumsum(f) / fs)


def _ook(bits, fs, baud):
    sps = fs / baud
    idx = (np.arange(int(len(bits) * sps)) / sps).astype(int)
    return (np.array(list(bits))[idx] == "1").astype(np.complex128)


def _noise(n, p):
    return (RNG.normal(size=n) + 1j * RNG.normal(size=n)) * np.sqrt(p / 2)


def _settings(tmp_path):
    return C.Settings(data_dir=str(tmp_path / "data"), capture_dir=str(tmp_path / "caps"))


def test_plan_keeps_dc_out_of_the_slice(tmp_path):
    s = _settings(tmp_path)
    p = EC.make_plan(s, 917.5e6, 917.6e6, 0, per_hour=1200)
    assert p["decim"] == 8 and p["slice_rate"] == 256_000
    assert p["lo_hz"] < p["f_lo_hz"] - 20e3  # DC spike below the slice
    assert p["seconds"] == 45  # ~15 bursts at 1200/h
    assert p["bytes"] == 45 * 2_048_000 * 2
    wide = EC.make_plan(s, 917.5e6, 917.6e6, 3)
    assert wide["decim"] == 1 and wide["offset_hz"] == 0
    with pytest.raises(ValueError):
        EC.make_plan(s, 917.6e6, 917.5e6, 0)


def test_analyse_synthetic_fsk_capture(tmp_path):
    s = _settings(tmp_path)
    plan = EC.make_plan(s, 917.5e6, 917.6e6, 0, seconds=2)
    d, rec = EC.new_capture(s, plan)
    fs, n = plan["sample_rate"], int(2 * plan["sample_rate"])
    x = _noise(n, 2e-4)
    frame = "AA" * 4 + "2DD4" + "C0FFEE0123456789"
    burst = _fsk(_bits(frame), fs, 4800, 2400) * 0.15
    f_sig = 917.553e6 - plan["lo_hz"]
    for k, t in enumerate((0.3, 0.9, 1.5)):
        i = int(t * fs)
        x[i:i + len(burst)] += burst * np.exp(2j * np.pi * f_sig * np.arange(i, i + len(burst)) / fs)
    x += _noise(n, 1e-8) + 0.01  # small DC offset, like the RTL
    u = np.empty(2 * n, np.uint8)
    u[0::2] = np.clip(np.round(x.real * 127.5 + 127.5), 0, 255)
    u[1::2] = np.clip(np.round(x.imag * 127.5 + 127.5), 0, 255)
    u.tofile(d / f"{rec['id']}.sigmf-data")
    EC.update(d, status="recorded", recording={"started": time.time()})
    out = EC.analyse(d)
    assert out["status"] == "done"
    s_ = out["analysis"]["summary"]
    assert s_["n_target"] == 3
    assert s_["modulation"] == "2-FSK"
    assert abs(s_["baud"] - 4800) < 100
    assert any("2DD4" in k for k in s_["sync_hits"])
    assert list(s_["repeated_frames"]) == ["C0FFEE0123456789"]
    tgt = [b for b in out["analysis"]["bursts"] if b["target"]]
    assert all(abs(b["f_mhz"] - 917.553) < 0.002 for b in tgt)
    names = {f["name"] for f in out["files"]}
    assert {"overview.png", "burst.png", "README.txt", f"{rec['id']}.sigmf-meta"} <= names
    meta = json.loads((d / f"{rec['id']}.sigmf-meta").read_text())
    assert meta["global"]["core:datatype"] == "cu8" and len(meta["annotations"]) >= 3
    assert EC.list_captures(s)[0]["modulation"] == "2-FSK"


@pytest.mark.parametrize("kind", ["ook", "cw"])
def test_classifier(kind):
    fs = 256e3
    if kind == "ook":
        man = "".join("10" if b == "1" else "01" for b in _bits("1F2A60")[-21:] + _bits("ABCD1234"))
        sig = _ook("10" * 16 + man, fs, 8000)
    else:
        sig = np.exp(2j * np.pi * 5e3 * np.arange(int(0.015 * fs)) / fs)
    y = np.concatenate([np.zeros(20000), 0.1 * sig, np.zeros(30000)])
    y = (y + _noise(len(y), 1e-6)).astype(np.complex64)
    S, f, dt = A.spectrogram(y, fs)
    bursts, _ = A.detect(S, f, dt)
    assert len(bursts) == 1
    r = A.analyse_burst(y, fs, bursts[0])
    if kind == "ook":
        assert r["modulation"] == "OOK/ASK" and abs(r["baud"] - 8000) < 100
        assert any("ERT SCM" in x["label"] for x in r["decode"]["sync"])
    else:
        assert r["modulation"].startswith("CW")


def test_capture_job_pauses_channel_and_resumes(tmp_path, monkeypatch):
    s = _settings(tmp_path)
    s.schedule.channel_minutes, s.schedule.sweep_minutes = 60, 0.01
    log, stop = [], threading.Event()

    def fake_channel(settings, db, seconds, publish=None, stop_event=None, on_start=None):
        log.append("channel")
        t0 = time.time()
        while time.time() - t0 < 5 and not stop_event.is_set():
            time.sleep(0.02)

    def fake_record(d, exe="rtl_sdr"):
        log.append(("record", bus.mode.get("mode")))
        assert APP_POST(app)({"f_lo": 917.5e6, "f_hi": 917.6e6, "bb": 0}).status_code == 409  # busy
        return EC.update(d, status="recorded")

    monkeypatch.setattr(M, "channel_session", fake_channel)
    monkeypatch.setattr(EC, "record", fake_record)
    monkeypatch.setattr(EC, "start_analysis", lambda d: log.append("analyse"))
    monkeypatch.setattr(APP, "emission_buckets", lambda db, t0: {"buckets": []})
    db = type("DB", (), {"event": lambda *a: None})()
    bus = APP.Bus()
    app = APP.create_app(s, bus)
    t = threading.Thread(target=M.run_loop, args=(s, db, bus, stop))
    t.start()
    try:
        while bus.request_capture is None or log[:1] != ["channel"]:
            time.sleep(0.01)
        r = APP_POST(app)({"f_lo": 917.5e6, "f_hi": 917.6e6, "bb": 0, "seconds": 5})
        assert r.status_code == 202
        cid = json.loads(r.body)["id"]
        t_end = time.time() + 10
        while log.count("channel") < 2 and time.time() < t_end:
            time.sleep(0.01)
    finally:
        stop.set()
        t.join(timeout=10)
    assert log[:4] == ["channel", ("record", "capture"), "analyse", "channel"]  # no sweep in between
    status = next(r.endpoint for r in app.routes if getattr(r, "path", None) == "/api/capture/{cid}")(cid)
    assert status["status"] == "recorded" and status["plan"]["seconds"] == 5


def APP_POST(app):
    return next(r.endpoint for r in app.routes if getattr(r, "path", None) == "/api/capture"
                and "POST" in getattr(r, "methods", ()))


def test_capture_without_radio_and_bad_ids(tmp_path):
    app = APP.create_app(_settings(tmp_path), None)
    assert APP_POST(app)({"f_lo": 917.5e6, "f_hi": 917.6e6, "bb": 0}).status_code == 409
    status = next(r.endpoint for r in app.routes if getattr(r, "path", None) == "/api/capture/{cid}")
    assert status("../../etc").status_code == 404
    f = next(r.endpoint for r in app.routes if getattr(r, "path", None) == "/api/capture/{cid}/file/{name}")
    assert f("20260101-000000_917.550MHz_bw0", "../../x").status_code == 404
