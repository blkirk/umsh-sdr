"""SQLite storage. One writer thread per process; WAL so the dashboard can read."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS packets (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,            -- burst start (unix s); decode time if unmatched
    ts_end REAL,
    decoded_at REAL NOT NULL,
    pay_len INTEGER NOT NULL,
    crc_ok INTEGER NOT NULL,
    cr INTEGER,
    payload BLOB NOT NULL,
    airtime_ms REAL,             -- measured burst length (NULL if unmatched)
    rssi_dbfs REAL,              -- in-channel power during the burst
    noise_dbfs REAL,             -- in-channel floor just before the burst
    snr_db REAL,                 -- preamble SNR (in-band, like SX1262) if available, else power-based
    snr_power_db REAL,           -- (burst power - floor) / floor; NULL for packets below the floor
    rssi_dbm REAL,               -- via calibration offset, if known
    cfo_hz REAL,                 -- transmitter carrier minus SDR tuning (+ = above)
    cfo_ppm REAL,                -- cfo_hz relative to 917.5 MHz (TX vs SDR crystal)
    clipped REAL,                -- fraction of raw samples at the ADC rails during burst
    gain_db REAL,
    matched INTEGER,             -- 1: own energy burst found; 0: power measured over computed window
    umsh TEXT                    -- JSON of dissector fields (tshark), if available
);
CREATE INDEX IF NOT EXISTS packets_ts ON packets(ts);

CREATE TABLE IF NOT EXISTS band_seconds (
    ts INTEGER NOT NULL,         -- unix second
    band TEXT NOT NULL,
    floor_dbfs REAL,             -- dBFS per 500 kHz: p10 of 10 ms frames, UMSH packets excluded
    median_dbfs REAL,
    max_dbfs REAL,
    mean_dbfs REAL,              -- linear mean, UMSH packets excluded
    busy_other REAL,             -- fraction of frames > ref floor + 6 dB, not UMSH
    busy_umsh REAL,              -- fraction of frames inside decoded UMSH bursts
    clip REAL,                   -- fraction of raw samples at ADC rails
    gain_db REAL,
    PRIMARY KEY (ts, band)
);

CREATE TABLE IF NOT EXISTS emissions (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    ts_end REAL NOT NULL,
    f_center_hz REAL NOT NULL,
    bw_hz REAL NOT NULL,
    peak_dbfs REAL NOT NULL,     -- strongest 10 ms frame, power within the emission
    above_floor_db REAL NOT NULL,
    cls TEXT NOT NULL,           -- narrowband | medium | wideband | continuous
    in_channel INTEGER NOT NULL,
    umsh INTEGER NOT NULL        -- 1 if it coincides with a decoded UMSH packet
);
CREATE INDEX IF NOT EXISTS emissions_ts ON emissions(ts);

CREATE TABLE IF NOT EXISTS spectra (    -- per-minute channel-mode spectrum
    ts INTEGER PRIMARY KEY,
    f0_hz REAL, df_hz REAL,
    mean BLOB, maxhold BLOB, p10 BLOB, -- float32 dBFS per bin
    gain_db REAL
);

CREATE TABLE IF NOT EXISTS sweeps (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL, ts_end REAL NOT NULL,
    f0_hz REAL, df_hz REAL, n INTEGER,
    p50 BLOB, p90 BLOB, maxhold BLOB,   -- float32 rtl_power dB per bin
    gain_db REAL, kind TEXT DEFAULT 'survey'
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY, ts REAL NOT NULL, kind TEXT NOT NULL, detail TEXT
);
"""


def f32(a) -> bytes:
    return np.asarray(a, dtype=np.float32).tobytes()


def unf32(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


class DB:
    def __init__(self, path: Path, readonly: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        uri = f"file:{self.path}?mode=ro" if readonly else f"file:{self.path}"
        self.conn = sqlite3.connect(uri, uri=True, check_same_thread=False, timeout=10)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        if not readonly:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    def execute(self, sql: str, args=()):
        with self.lock:
            cur = self.conn.execute(sql, args)
            self.conn.commit()
            return cur

    def executemany(self, sql: str, rows):
        with self.lock:
            self.conn.executemany(sql, rows)
            self.conn.commit()

    def query(self, sql: str, args=()) -> list[sqlite3.Row]:
        with self.lock:
            return self.conn.execute(sql, args).fetchall()

    def insert(self, table: str, row: dict) -> int:
        cols = ",".join(row)
        qs = ",".join("?" * len(row))
        return self.execute(f"INSERT INTO {table} ({cols}) VALUES ({qs})", tuple(row.values())).lastrowid

    def event(self, ts: float, kind: str, detail=None) -> None:
        self.insert("events", {"ts": ts, "kind": kind, "detail": json.dumps(detail) if detail is not None else None})
