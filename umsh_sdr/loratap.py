"""pcap files with LoRaTAP v0 (DLT 270) framing.

Byte layout mirrors umshctl's writer (~/umsh/tools/umshctl/src/command/capture/pcap.rs)
so the UMSH Wireshark dissector treats SDR captures exactly like node captures.
"""
from __future__ import annotations

import struct
import threading
from dataclasses import dataclass
from pathlib import Path

LINKTYPE_LORATAP = 270
LINKTYPE_ETHERNET = 1
LORATAP_V0_LEN = 15
RSSI_BIAS = 139
_BW_CODE = {125_000: 1, 250_000: 2, 500_000: 4}
_BW_HZ = {v: k for k, v in _BW_CODE.items()}


def loratap_header(freq_hz: int, bw_hz: int, sf: int, rssi_dbm: float | None,
                   snr_db: float | None, sync_word: int) -> bytes:
    rssi = 0 if rssi_dbm is None else max(0, min(255, round(rssi_dbm) + RSSI_BIAS))
    snr = -128 if snr_db is None else max(-128, min(127, round(snr_db * 4)))
    return struct.pack(">BBHIBBBBBbB", 0, 0, LORATAP_V0_LEN, int(freq_hz), _BW_CODE.get(bw_hz, 0),
                       sf, rssi, rssi, rssi, snr, sync_word & 0xFF)


class PcapWriter:
    """Append-safe pcap writer (creates the global header on first use)."""

    def __init__(self, path: Path, linktype: int = LINKTYPE_LORATAP):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        new = not self.path.exists() or self.path.stat().st_size == 0
        self._f = open(self.path, "ab")
        if new:
            self._f.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, linktype))
            self._f.flush()

    def write(self, ts: float, data: bytes) -> None:
        sec = int(ts)
        usec = int(round((ts - sec) * 1e6))
        if usec >= 1_000_000:
            sec, usec = sec + 1, usec - 1_000_000
        with self._lock:
            self._f.write(struct.pack("<IIII", sec, usec, len(data), len(data)) + data)
            self._f.flush()

    def write_lora(self, ts: float, payload: bytes, *, freq_hz: int, bw_hz: int, sf: int,
                   rssi_dbm: float | None, snr_db: float | None, sync_word: int) -> None:
        self.write(ts, loratap_header(freq_hz, bw_hz, sf, rssi_dbm, snr_db, sync_word) + payload)

    def close(self) -> None:
        self._f.close()


@dataclass
class Record:
    ts: float
    linktype: int
    data: bytes
    payload: bytes  # LoRa PHY payload when recoverable
    freq_hz: int | None = None
    bw_hz: int | None = None
    sf: int | None = None
    rssi_dbm: float | None = None
    snr_db: float | None = None
    sync_word: int | None = None


def _strip(linktype: int, data: bytes) -> Record:
    rec = Record(0.0, linktype, data, data)
    if linktype == LINKTYPE_LORATAP and len(data) >= 4:
        hlen = struct.unpack(">H", data[2:4])[0]
        if data[0] == 0 and hlen >= LORATAP_V0_LEN:
            _, _, _, f, bw, sf, rssi, _, _, snr, sw = struct.unpack(">BBHIBBBBBbB", data[:LORATAP_V0_LEN])
            rec.payload = data[hlen:]
            rec.freq_hz, rec.bw_hz, rec.sf, rec.sync_word = f, _BW_HZ.get(bw), sf, sw
            rec.rssi_dbm = None if rssi == 0 else rssi - RSSI_BIAS
            rec.snr_db = None if snr == -128 else snr / 4
    elif linktype == LINKTYPE_ETHERNET and len(data) > 42 and data[12:14] == b"\x08\x00":
        ihl = (data[14] & 0x0F) * 4
        if data[14 + 9] == 17:  # UDP
            rec.payload = data[14 + ihl + 8:]
    return rec


def read_pcap(path: Path) -> list[Record]:
    raw = Path(path).read_bytes()
    magic = struct.unpack("<I", raw[:4])[0]
    endian = "<" if magic in (0xA1B2C3D4, 0xA1B23C4D) else ">"
    nano = magic in (0xA1B23C4D, 0x4D3CB2A1)
    linktype = struct.unpack(endian + "I", raw[20:24])[0]
    out, off = [], 24
    while off + 16 <= len(raw):
        sec, frac, incl, _ = struct.unpack(endian + "IIII", raw[off:off + 16])
        off += 16
        rec = _strip(linktype, raw[off:off + incl])
        rec.ts = sec + frac / (1e9 if nano else 1e6)
        out.append(rec)
        off += incl
    return out
