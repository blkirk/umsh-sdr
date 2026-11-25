"""umsh-sdr command line.

  run        scheduler (channel mode <-> sweeps) + live dashboard  [the daemon]
  channel    channel mode only, for --minutes (no sweeps)
  replay     process an rtl_sdr .u8 recording through the same pipeline
  record     record raw IQ (rtl_sdr .u8) at the channel-mode tuning
  sweep      one wideband rtl_power survey
  overload   gain-step test for RTL-generated intermod
  calibrate  ppm | gain | level | selfnoise
  report     build the HTML report for a time range
  dashboard  dashboard over the database only (no radio)
  capture    record + analyse one emission slice into ~/Downloads/umsh-sdr-captures
  analyse    re-run the analysis of a capture folder
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import config as C
from .db import DB


def _monitor(settings, db, publish=None, realtime=True):
    from .dissect import Dissector
    from .monitor import Monitor
    m = Monitor(db, settings, publish=publish, dissector=Dissector(db, publish))
    m.realtime = realtime
    return m


def release(fg) -> None:
    """Drop every reference to the RTL source so librtlsdr closes the device now,
    not whenever the garbage collector gets round to it."""
    import gc
    fg.disconnect_all()
    fg.src = None
    gc.collect()


def open_flowgraph(make, tries: int = 8):
    import gc
    for i in range(tries):
        try:
            return make()
        except RuntimeError as e:
            if "rtlsdr" not in str(e) or i == tries - 1:
                raise
            gc.collect()
            time.sleep(0.5 * (i + 1))


def channel_session(settings, db, seconds: float, publish=None, stop_event: threading.Event | None = None,
                    on_start=None):
    from .flowgraph import ChannelFlowgraph
    m = _monitor(settings, db, publish)
    m.start()
    cal = settings.calibration
    fg = open_flowgraph(lambda: ChannelFlowgraph(m.on_frame, m.on_packet, ppm=cal.ppm, gain_db=cal.gain_db))
    db.event(time.time(), "channel_start", {"ppm": cal.ppm, "gain_db": cal.gain_db, "seconds": seconds})
    fg.start()
    if on_start:
        on_start(fg, m)
    t_end = time.time() + seconds
    try:
        while time.time() < t_end and not (stop_event and stop_event.is_set()):
            time.sleep(0.25)
    finally:
        fg.stop()
        fg.wait()
        release(fg)
        del fg
        m.stop()
        db.event(time.time(), "channel_stop", m.stats)
    return m


def cmd_replay(a, settings):
    from .flowgraph import ChannelFlowgraph
    db = DB(Path(a.db) if a.db else settings.db_path)
    path = Path(a.file)
    t0 = a.t0 if a.t0 is not None else path.stat().st_mtime - path.stat().st_size / 2 / C.SAMPLE_RATE
    m = _monitor(settings, db, realtime=False)
    if a.gain is not None:
        settings.calibration.gain_db = a.gain
    m.start()
    fg = ChannelFlowgraph(m.on_frame, m.on_packet, ppm=0, gain_db=0, source="file", file_path=str(path), file_t0=t0)
    fg.run()
    time.sleep(1.0)  # let the last decodes reach the monitor
    m.drain()
    m.stop()
    print(json.dumps(m.stats))
    for r in db.query("SELECT id, ts, pay_len, crc_ok, round(rssi_dbfs,1) r, round(noise_dbfs,1) n, round(snr_db,1) s,"
                      " round(cfo_hz) c, round(airtime_ms) a FROM packets WHERE ts >= ? ORDER BY ts", (t0 - 1,)):
        print(dict(r))


def cmd_record(a, settings):
    cal = settings.calibration
    n = int(a.seconds * C.SAMPLE_RATE)
    gain = a.gain if a.gain is not None else cal.gain_db
    subprocess.run(["rtl_sdr", "-f", str(C.LO_HZ), "-s", str(C.SAMPLE_RATE), "-g", str(gain), "-p",
                    str(int(round(cal.ppm))), "-n", str(n), a.out], check=True)
    print(f"recorded {a.seconds}s at gain {gain} to {a.out}; replay with: umsh-sdr replay {a.out} --gain {gain}")


def cmd_channel(a, settings):
    db = DB(settings.db_path)
    m = channel_session(settings, db, a.minutes * 60)
    print(json.dumps(m.stats))


def cmd_sweep(a, settings):
    from .sweep import survey
    db = DB(settings.db_path)
    res = survey(db, settings, duration_s=a.seconds)
    if res:
        row, S = res
        print(f"sweep #{row['id']}: {S.shape[0]} intervals x {row['n']} bins, f0 {row['f0_hz']/1e6:.3f} MHz "
              f"df {row['df_hz']:.0f} Hz")


def cmd_overload(a, settings):
    from .sweep import overload_check
    print(json.dumps(overload_check(DB(settings.db_path), settings, duration_s=a.seconds), indent=1))


def cmd_calibrate(a, settings):
    from . import calibrate
    calibrate.main(a, settings)


def cmd_report(a, settings):
    from . import report
    out = report.build(DB(settings.db_path, readonly=True), settings, hours=a.hours, out=a.out)
    print(out)


def cmd_dashboard(a, settings):
    from .dashboard.app import serve
    serve(settings, live=False)


def cmd_run(a, settings):
    from .dashboard.app import Bus, serve_in_thread
    db = DB(settings.db_path)
    bus = Bus()
    serve_in_thread(settings, bus)
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    sc = settings.schedule
    if a.channel_minutes is not None:
        sc.channel_minutes = a.channel_minutes
    if a.sweep_minutes is not None:
        sc.sweep_minutes = a.sweep_minutes
    run_loop(settings, db, bus, stop)
    print("stopped")


def run_loop(settings, db, bus, stop: threading.Event):
    """Channel capture, then a wideband sweep, repeated. A dashboard "Sweep now" ends the
    current capture early, runs one sweep, then starts a fresh capture period. A dashboard
    "Capture" ends it early, records one emission slice with rtl_sdr, starts the analysis in a
    background process, and goes straight back to channel capture."""
    from . import emission_capture as EC
    from .sweep import survey
    sc = settings.schedule
    sweep_s = (sc.sweep_minutes if sc.sweep_minutes > 0 else 5.0) * 60
    job = {"next": None}  # ("sweep", None) or ("capture", folder), set by the dashboard
    lock = threading.Lock()

    class _StopOrJob:  # ends the channel session early on shutdown or a dashboard request
        def is_set(self):
            return stop.is_set() or job["next"] is not None

    def _busy():
        m = bus.mode.get("mode")
        if m == "sweep":
            return "a sweep is already running"
        if m == "capture":
            return "a capture is already recording"
        if job["next"]:
            return f"a {job['next'][0]} is already starting"
        return None

    def request_sweep():
        with lock:
            if (why := _busy()):
                return {"ok": False, "reason": why}
            job["next"] = ("sweep", None)
        return {"ok": True, "seconds": sweep_s}

    def request_capture(plan, bucket=None):
        with lock:
            if (why := _busy()):
                return {"ok": False, "reason": why}
            d, rec = EC.new_capture(settings, plan, bucket)
            job["next"] = ("capture", d)
        return {"ok": True, "id": rec["id"], "seconds": plan["seconds"]}

    bus.request_sweep, bus.sweep_seconds, bus.request_capture = request_sweep, sweep_s, request_capture
    try:
        while not stop.is_set():
            bus.publish("mode", {"mode": "channel", "until": time.time() + sc.channel_minutes * 60})
            channel_session(settings, db, sc.channel_minutes * 60, publish=bus.publish, stop_event=_StopOrJob())
            if stop.is_set():
                break
            nxt = job["next"]
            if nxt and nxt[0] == "capture":
                d = nxt[1]
                plan = EC.load(d)["plan"]
                db.event(time.time(), "capture", {"id": d.name, "f_lo_hz": plan["f_lo_hz"], "seconds": plan["seconds"]})
                bus.publish("mode", {"mode": "capture", "id": d.name, "until": time.time() + plan["seconds"] + 3})
                with lock:
                    job["next"] = None
                try:
                    EC.record(d, settings.rtl_sdr)
                    EC.start_analysis(d)
                except Exception as e:
                    db.event(time.time(), "capture_error", repr(e))
                continue
            manual = nxt is not None
            if not manual and sc.sweep_minutes <= 0:
                break
            if manual:
                db.event(time.time(), "sweep_manual", {"seconds": sweep_s})
            bus.publish("mode", {"mode": "sweep", "manual": manual, "until": time.time() + sweep_s})
            with lock:
                job["next"] = None
            try:
                res = survey(db, settings, duration_s=sweep_s)
                if res:
                    bus.publish("sweep", {"id": res[0]["id"]})
            except Exception as e:
                db.event(time.time(), "sweep_error", repr(e))
    finally:
        bus.request_sweep = bus.request_capture = None
        nxt = job["next"]
        if nxt and nxt[0] == "capture":
            EC.update(nxt[1], status="cancelled", error="the monitor stopped before the capture started")


def cmd_capture(a, settings):
    """Record and analyse one slice directly (the radio must be free: no `run` daemon)."""
    from . import emission_capture as EC
    plan = EC.make_plan(settings, a.f_lo * 1e6, a.f_hi * 1e6, a.bw_bucket, seconds=a.seconds, gain_db=a.gain)
    d, rec = EC.new_capture(settings, plan)
    print(f"recording {plan['seconds']:.0f} s at {plan['lo_hz'] / 1e6:.6f} MHz into {d}", flush=True)
    EC.record(d, settings.rtl_sdr)
    print("analysing...", flush=True)
    rec = EC.analyse(d)
    print(json.dumps(rec["analysis"]["summary"], indent=1))
    print(d)


def cmd_analyse(a, settings):
    from . import emission_capture as EC
    rec = EC.analyse(Path(a.folder))
    print(json.dumps(rec["analysis"]["summary"], indent=1))


def main(argv=None):
    settings = C.load()
    p = argparse.ArgumentParser(prog="umsh-sdr", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("run")
    s.add_argument("--channel-minutes", type=float, help="override schedule.channel_minutes")
    s.add_argument("--sweep-minutes", type=float, help="override schedule.sweep_minutes (0 = no sweeps)")
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser("channel"); s.add_argument("--minutes", type=float, default=10); s.set_defaults(fn=cmd_channel)
    s = sub.add_parser("replay"); s.add_argument("file"); s.add_argument("--t0", type=float)
    s.add_argument("--gain", type=float, help="tuner gain the file was recorded at (stored with rows)")
    s.add_argument("--db", help="database path (default: data/umsh-sdr.sqlite)"); s.set_defaults(fn=cmd_replay)
    s = sub.add_parser("record"); s.add_argument("out"); s.add_argument("--seconds", type=float, default=30)
    s.add_argument("--gain", type=float); s.set_defaults(fn=cmd_record)
    s = sub.add_parser("sweep"); s.add_argument("--seconds", type=float, default=300); s.set_defaults(fn=cmd_sweep)
    s = sub.add_parser("overload"); s.add_argument("--seconds", type=float, default=60); s.set_defaults(fn=cmd_overload)
    s = sub.add_parser("calibrate"); s.add_argument("what", choices=["ppm", "gain", "level", "selfnoise"])
    s.add_argument("--pings", type=int, default=10); s.add_argument("--seconds", type=float, default=60)
    s.add_argument("--gains", default="29.7,33.8,37.2,40.2,43.4,44.5,48.0,49.6")
    s.add_argument("--apply", action="store_true", help="write the result to config.toml"); s.set_defaults(fn=cmd_calibrate)
    s = sub.add_parser("report"); s.add_argument("--hours", type=float, default=24); s.add_argument("--out")
    s.set_defaults(fn=cmd_report)
    sub.add_parser("dashboard").set_defaults(fn=cmd_dashboard)
    s = sub.add_parser("capture", help="record + analyse one emission slice (radio must be free)")
    s.add_argument("--f-lo", type=float, required=True, help="MHz"); s.add_argument("--f-hi", type=float, required=True)
    s.add_argument("--bw-bucket", type=int, default=0, help="0: <25 kHz, 1: 25-200, 2: 200-600, 3: >=600")
    s.add_argument("--seconds", type=float, default=30); s.add_argument("--gain", type=float); s.set_defaults(fn=cmd_capture)
    s = sub.add_parser("analyse", help="(re)analyse a capture folder"); s.add_argument("folder"); s.set_defaults(fn=cmd_analyse)
    a = p.parse_args(argv)
    a.fn(a, settings)


if __name__ == "__main__":
    main()
