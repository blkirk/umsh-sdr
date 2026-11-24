"""Radio constants and per-device calibration.

UMSH PHY values come from ~/umsh/crates/umsh-ulcp/src/profiles.rs (UMSH_US_CA)
and crates/umsh-radio-loraphy (explicit header, CRC on, IQ normal).
Calibration lives in ``config.toml`` at the project root and is rewritten by
``umsh-sdr calibrate ... --apply``.
"""
from __future__ import annotations

import dataclasses
import shutil
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.toml"

# --- UMSH PHY (fixed by the protocol profile) ---
CHANNEL_HZ = 917_500_000
BW_HZ = 500_000
SF = 10
CR = 1  # gr-lora_sdr encoding: 1 => 4/5
HAS_CRC = True
IMPLICIT_HEADER = False
SYNC_WORD = 0x12  # LoRa 8-bit form of the SX126x 0x1424 private sync word
TX_PREAMBLE = 32  # UMSH transmit preamble; other mesh stacks may use 8/16

# --- Capture geometry ---
SAMPLE_RATE = 2_000_000
LO_HZ = CHANNEL_HZ - 500_000  # keeps the RTL DC spike out of the channel
FFT_SIZE = 1024
FRAME_FFTS = 20  # FFTs averaged per analysis frame -> 10.24 ms frames

# Bands measured continuously in channel mode (Hz, absolute).
BANDS = {
    "channel": (CHANNEL_HZ - BW_HZ // 2, CHANNEL_HZ + BW_HZ // 2),
    "adj_low": (LO_HZ - 920_000, CHANNEL_HZ - BW_HZ // 2),
    "adj_high": (CHANNEL_HZ + BW_HZ // 2, LO_HZ + 920_000),
}

# Thermal noise in the 500 kHz channel and the SX1262 reference floor.
THERMAL_DBM = -174.0 + 10 * __import__("math").log10(BW_HZ)
SX1262_NF_DB = 6.0
SX1262_FLOOR_DBM = THERMAL_DBM + SX1262_NF_DB
SF10_DEMOD_SNR_DB = -15.0


@dataclass
class Calibration:
    ppm: float = 0.0  # per dongle: `umsh-sdr calibrate ppm --apply` (a generic R820T measured +58)
    gain_db: float = 43.4  # channel-mode tuner gain
    sweep_gain_db: float = 43.4
    dbm_offset: float | None = None  # dBm = dBFS + offset at gain_db
    self_floor_dbfs: dict[str, float] = field(default_factory=dict)  # terminated-input floor per band


@dataclass
class Schedule:
    channel_minutes: float = 55.0
    sweep_minutes: float = 5.0
    sweep_start_hz: int = 900_000_000
    sweep_stop_hz: int = 935_000_000
    sweep_bin_hz: int = 10_000
    sweep_interval_s: int = 10


@dataclass
class Settings:
    calibration: Calibration = field(default_factory=Calibration)
    schedule: Schedule = field(default_factory=Schedule)
    data_dir: str = str(ROOT / "data")
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8917
    umshctl: str = field(default_factory=lambda: shutil.which("umshctl")
                         or str(Path.home() / ".cargo/bin/umshctl"))
    node_port: str = "/dev/ttyACM0"
    name_hints: str = str(ROOT / "name-hints.txt")  # "<key or hint>\t<node name>" per line
    ping_target: str = ""  # base58 key of a node that answers pings; set in config.toml

    @property
    def db_path(self) -> Path:
        return Path(self.data_dir) / "umsh-sdr.sqlite"


def _merge(dc, values: dict):
    for k, v in values.items():
        if not hasattr(dc, k):
            continue
        cur = getattr(dc, k)
        if dataclasses.is_dataclass(cur) and isinstance(v, dict):
            _merge(cur, v)
        else:
            setattr(dc, k, v)
    return dc


def load(path: Path = CONFIG_PATH) -> Settings:
    s = Settings()
    if path.exists():
        _merge(s, tomllib.loads(path.read_text()))
    if not Path(s.data_dir).is_absolute():
        s.data_dir = str(ROOT / s.data_dir)
    return s


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    raise TypeError(v)


def save(s: Settings, path: Path = CONFIG_PATH) -> None:
    lines = ["# umsh-sdr settings (calibration is rewritten by `umsh-sdr calibrate --apply`)"]
    top = {k: v for k, v in dataclasses.asdict(s).items() if not isinstance(v, dict)}
    lines += [f"{k} = {_toml_value(v)}" for k, v in top.items() if v is not None]
    for section in ("calibration", "schedule"):
        d = dataclasses.asdict(getattr(s, section))
        lines.append(f"\n[{section}]")
        nested = {}
        for k, v in d.items():
            if isinstance(v, dict):
                nested[k] = v
            elif v is not None:
                lines.append(f"{k} = {_toml_value(v)}")
        for k, v in nested.items():
            lines.append(f"\n[{section}.{k}]")
            lines += [f"{kk} = {_toml_value(vv)}" for kk, vv in v.items()]
    path.write_text("\n".join(lines) + "\n")
