"""'Sweep now': a dashboard request ends the channel session early, one sweep runs, capture resumes."""
import threading
import time

from umsh_sdr import __main__ as M
from umsh_sdr import config as C
from umsh_sdr.dashboard import app as A


def _endpoint(app, path):
    return next(r.endpoint for r in app.routes if getattr(r, "path", None) == path)


def test_sweep_now_interrupts_channel_and_resumes(tmp_path, monkeypatch):
    s = C.Settings(data_dir=str(tmp_path))
    s.schedule.channel_minutes, s.schedule.sweep_minutes = 60, 0.01
    log, stop = [], threading.Event()

    def fake_channel(settings, db, seconds, publish=None, stop_event=None, on_start=None):
        log.append("channel")
        t0 = time.time()
        while time.time() - t0 < 5 and not stop_event.is_set():
            time.sleep(0.02)
        log.append("ended early" if time.time() - t0 < 5 else "timed out")

    def fake_survey(db, settings, duration_s=None, **kw):
        log.append(("sweep", round(duration_s, 3), bus.mode.get("mode"), bus.mode.get("manual")))
        assert _endpoint(app, "/api/sweep/now")().status_code == 409  # refused while sweeping

    monkeypatch.setattr(M, "channel_session", fake_channel)
    monkeypatch.setattr("umsh_sdr.sweep.survey", fake_survey)
    db = type("DB", (), {"event": lambda *a: None})()
    bus = A.Bus()
    app = A.create_app(s, bus)
    t = threading.Thread(target=M.run_loop, args=(s, db, bus, stop))
    t.start()
    while bus.request_sweep is None or log[:1] != ["channel"]:
        time.sleep(0.01)
    r = _endpoint(app, "/api/sweep/now")()
    assert r.status_code == 202
    assert _endpoint(app, "/api/sweep/now")().status_code == 409  # second press while starting
    while log.count("channel") < 2:
        time.sleep(0.01)
    stop.set()
    t.join(timeout=10)
    assert not t.is_alive()
    assert log[:4] == ["channel", "ended early", ("sweep", 0.6, "sweep", True), "channel"]


def test_sweep_now_without_radio(tmp_path):
    r = _endpoint(A.create_app(C.Settings(data_dir=str(tmp_path)), None), "/api/sweep/now")()
    assert r.status_code == 409


def test_names_endpoint_reads_and_reloads(tmp_path):
    f = tmp_path / "names.txt"
    f.write_text("# comment\n7Abc\tHilltop Relay\r\n5Xyz\tBase Node\n\nbad-line-without-name\n")
    s = C.Settings(data_dir=str(tmp_path), name_hints=str(f))
    names = _endpoint(A.create_app(s, None), "/api/names")
    assert names() == [["7Abc", "Hilltop Relay"], ["5Xyz", "Base Node"]]
    import os
    f.write_text("5Xyz\tBase Node 2\n")
    os.utime(f, (1, 1))
    assert names() == [["5Xyz", "Base Node 2"]]
    f.unlink()
    assert names() == []
