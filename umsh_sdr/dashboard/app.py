"""Local dashboard: FastAPI JSON endpoints over SQLite + a websocket live feed."""
from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from .. import config as C
from ..db import DB, unf32
from ..emission_buckets import buckets as emission_buckets
from ..umsh_addr import parse as parse_addr, summarize

STATIC = Path(__file__).parent / "static"


class Bus:
    """Thread-safe fan-out from radio threads to websocket clients."""

    def __init__(self):
        self.loop: asyncio.AbstractEventLoop | None = None
        self.clients: set[asyncio.Queue] = set()
        self.last: dict[str, dict] = {}
        self.mode: dict = {}
        self.request_sweep = None  # set by `umsh-sdr run`: () -> {"ok", "reason"|"seconds"}
        self.sweep_seconds: float | None = None

    def publish(self, kind: str, data: dict):
        if kind == "wf":  # halve the row: max of bin pairs keeps narrow signals visible
            row = np.asarray(data["row"], dtype=np.float32)
            data = {**data, "row": np.round(row.reshape(-1, 2).max(1), 1).tolist()}
        msg = json.dumps({"kind": kind, "data": data}, default=_jsonable)
        if kind in ("mode", "second"):
            self.last[kind] = msg
        if kind == "mode":
            self.mode = data
        loop = self.loop
        if loop is None:
            return
        for q in list(self.clients):
            loop.call_soon_threadsafe(_offer, q, msg)


def _offer(q: asyncio.Queue, msg: str):
    if q.full():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
    q.put_nowait(msg)


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, bytes):
        return o.hex()
    raise TypeError(type(o))


def _rows(rows):
    out = []
    for r in rows:
        d = dict(r)
        for k, v in d.items():
            if isinstance(v, bytes):
                d[k] = v.hex()
        out.append(d)
    return out


def create_app(settings: C.Settings, bus: Bus | None) -> FastAPI:
    app = FastAPI(title="UMSH SDR monitor")
    state = {"db": None}

    def db() -> DB:
        if state["db"] is None:
            state["db"] = DB(settings.db_path, readonly=True)
        return state["db"]

    @app.on_event("startup")
    async def _startup():
        if bus is not None:
            bus.loop = asyncio.get_running_loop()

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/status")
    def status():
        d = db()
        cal = settings.calibration
        last = d.query("SELECT kind, ts, detail FROM events ORDER BY id DESC LIMIT 8")
        n = d.query("SELECT count(*) n, sum(crc_ok) ok FROM packets WHERE ts > ?", (time.time() - 3600,))[0]
        return {"live": bus is not None, "now": time.time(), "packets_1h": n["n"], "crc_ok_1h": n["ok"] or 0,
                "calibration": {"ppm": cal.ppm, "gain_db": cal.gain_db, "dbm_offset": cal.dbm_offset,
                                "self_floor_dbfs": cal.self_floor_dbfs},
                "reference": {"thermal_dbm": C.THERMAL_DBM, "sx1262_floor_dbm": C.SX1262_FLOOR_DBM},
                "mode": bus.mode if bus else None,
                "can_sweep": bool(bus and bus.request_sweep), "sweep_seconds": bus.sweep_seconds if bus else None,
                "lo_hz": C.LO_HZ, "sample_rate": C.SAMPLE_RATE, "channel": C.BANDS["channel"],
                "events": _rows(last)}

    @app.get("/api/packets")
    def packets(limit: int = 100):
        rows = db().query("SELECT id, ts, ts_end, pay_len, crc_ok, matched, rssi_dbfs, noise_dbfs, snr_db, rssi_dbm,"
                          " cfo_hz, clipped, airtime_ms, umsh, payload FROM packets ORDER BY ts DESC LIMIT ?",
                          (min(limit, 1000),))
        out = _rows(rows)
        for r in out:
            r["addr"] = parse_addr(bytes.fromhex(r["payload"])) if r.get("payload") else None
        return out

    @app.get("/api/seconds")
    def seconds(minutes: float = 30):
        t0 = int(time.time() - minutes * 60)
        # dBFS only compares at one gain: show the current channel gain only
        rows = db().query("SELECT ts, band, floor_dbfs, busy_other, busy_umsh FROM band_seconds WHERE ts >= ? "
                          "AND abs(gain_db - ?) < 0.05 ORDER BY ts", (t0, settings.calibration.gain_db))
        out: dict[str, list] = {b: [] for b in C.BANDS}
        for r in rows:
            out.setdefault(r["band"], []).append([r["ts"], r["floor_dbfs"], r["busy_other"], r["busy_umsh"]])
        return out

    @app.get("/api/nodes")
    def nodes(hours: float = 24):
        t0 = 0 if hours <= 0 else time.time() - hours * 3600
        rows = db().query("SELECT ts, payload, crc_ok, matched, clipped, snr_db FROM packets WHERE ts >= ?", (t0,))
        return summarize([dict(r) for r in rows])

    names_cache = {"mtime": None, "rows": []}

    @app.get("/api/names")
    def names():
        """Key-hint -> node name pairs from name-hints.txt, re-read whenever the file changes."""
        p = Path(settings.name_hints)
        try:
            mtime = p.stat().st_mtime
        except OSError:
            return []
        if mtime != names_cache["mtime"]:
            rows = []
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t", 1) if "\t" in line else line.split(None, 1)
                if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                    rows.append([parts[0].strip(), parts[1].strip()])
            names_cache.update(mtime=mtime, rows=rows)
        return names_cache["rows"]

    @app.get("/api/emissions/buckets")
    def emissions_buckets(hours: float = 24, step_khz: float = 100):
        t0 = 0 if hours <= 0 else time.time() - hours * 3600
        return emission_buckets(db(), t0, max(1.0, step_khz) * 1e3)

    @app.get("/api/emissions")
    def emissions(limit: int = 100):
        return _rows(db().query("SELECT * FROM emissions ORDER BY ts DESC LIMIT ?", (min(limit, 1000),)))

    @app.get("/api/sweep/latest")
    def sweep_latest():
        r = db().query("SELECT * FROM sweeps WHERE kind='survey' ORDER BY ts DESC LIMIT 1")
        if not r:
            return None
        r = r[0]
        return {"ts": r["ts"], "f0_hz": r["f0_hz"], "df_hz": r["df_hz"], "gain_db": r["gain_db"],
                "p50": np.round(unf32(r["p50"]), 1).tolist(), "maxhold": np.round(unf32(r["maxhold"]), 1).tolist()}

    @app.post("/api/sweep/now")
    def sweep_now():
        """Stop the channel capture and run one wideband sweep now; capture resumes after it."""
        if bus is None or bus.request_sweep is None:
            return JSONResponse({"ok": False, "reason": "dashboard is not attached to the radio (run `umsh-sdr run`)"},
                                status_code=409)
        r = bus.request_sweep()
        return JSONResponse(r, status_code=202 if r["ok"] else 409)

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        if bus is None:
            await sock.close()
            return
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        bus.clients.add(q)
        try:
            for msg in bus.last.values():
                await sock.send_text(msg)
            while True:
                await sock.send_text(await q.get())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            bus.clients.discard(q)

    return app


def serve(settings: C.Settings, live: bool = False, bus: Bus | None = None):
    import uvicorn
    uvicorn.run(create_app(settings, bus if live else None), host=settings.dashboard_host,
                port=settings.dashboard_port, log_level="warning")


def serve_in_thread(settings: C.Settings, bus: Bus) -> threading.Thread:
    import uvicorn
    server = uvicorn.Server(uvicorn.Config(create_app(settings, bus), host=settings.dashboard_host,
                                           port=settings.dashboard_port, log_level="warning"))
    t = threading.Thread(target=server.run, name="dashboard", daemon=True)
    t.start()
    return t
