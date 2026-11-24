"""Decode UMSH fields by running tshark with the UMSH Lua dissector.

Each CRC-valid frame is written to a one-record LoRaTAP pcap and dissected
in a background thread; the ``umsh.*`` fields are stored as JSON on the
packet row. Without tshark this is a no-op (raw payload bytes are still kept).
"""
from __future__ import annotations

import os
import json
import queue
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from . import config as C
from .loratap import PcapWriter

UMSH_DISSECTOR = Path(os.environ.get("UMSH_REPO", Path.home() / "umsh")) / "dissectors/umsh"


def tshark_path() -> str | None:
    return shutil.which("tshark")


def tshark_args() -> list[str]:
    """Load the dissector from the UMSH checkout unless it's already installed
    as a personal plugin (``make install-dissector``)."""
    personal = Path.home() / ".local/lib/wireshark/plugins/umsh"
    if personal.exists() or not UMSH_DISSECTOR.exists():
        return []
    return ["-X", f"lua_script:{UMSH_DISSECTOR / 'umsh.lua'}"]


def _flatten_layers(obj, out: dict, want=("umsh", "loratap")):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if any(k.startswith(w) for w in want) and not isinstance(v, (dict, list)):
                out.setdefault(k, v)
            _flatten_layers(v, out, want)
    elif isinstance(obj, list):
        for v in obj:
            _flatten_layers(v, out, want)


def dissect_pcap(path: Path) -> list[dict]:
    exe = tshark_path()
    if not exe:
        return []
    env = {"LUA_PATH": f"{UMSH_DISSECTOR}/?.lua;;"}
    import os
    proc = subprocess.run([exe, *tshark_args(), "-r", str(path), "-T", "json"], capture_output=True,
                          text=True, timeout=60, env={**os.environ, **env})
    if proc.returncode != 0 or not proc.stdout.strip():
        return []
    out = []
    for pkt in json.loads(proc.stdout):
        fields: dict = {}
        _flatten_layers(pkt.get("_source", {}).get("layers", {}), fields)
        protos = pkt.get("_source", {}).get("layers", {}).get("frame", {}).get("frame.protocols", "")
        fields["_protocols"] = protos
        out.append(fields)
    return out


class Dissector:
    def __init__(self, db_, publish=None):
        self.db = db_
        self.publish = publish or (lambda kind, data: None)
        self.enabled = tshark_path() is not None
        self.q: queue.Queue = queue.Queue(maxsize=1000)
        self.tmp = Path(tempfile.mkdtemp(prefix="umsh-sdr-dissect-"))
        self.thread = threading.Thread(target=self._run, name="dissector", daemon=True)
        if self.enabled:
            self.thread.start()

    def submit(self, pid, ts, payload, rssi_dbm, snr_db):
        if self.enabled:
            try:
                self.q.put_nowait((pid, ts, payload, rssi_dbm, snr_db))
            except queue.Full:
                pass

    def _run(self):
        while True:
            pid, ts, payload, rssi, snr = self.q.get()
            path = self.tmp / f"{pid}.pcap"
            try:
                w = PcapWriter(path)
                w.write_lora(ts, payload, freq_hz=C.CHANNEL_HZ, bw_hz=C.BW_HZ, sf=C.SF, rssi_dbm=rssi,
                             snr_db=snr, sync_word=C.SYNC_WORD)
                w.close()
                res = dissect_pcap(path)
                if res:
                    self.db.execute("UPDATE packets SET umsh=? WHERE id=?", (json.dumps(res[0]), pid))
                    self.publish("dissected", {"id": pid, "umsh": res[0]})
            except Exception as e:
                self.db.event(ts, "dissect_error", repr(e))
            finally:
                path.unlink(missing_ok=True)
