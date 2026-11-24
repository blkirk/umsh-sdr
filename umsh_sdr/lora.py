"""LoRa timing and offset helpers."""
from __future__ import annotations

import math

from . import config as C


def symbol_time(sf: int = C.SF, bw: int = C.BW_HZ) -> float:
    return (1 << sf) / bw


def airtime(pay_len: int, preamble: int = C.TX_PREAMBLE, sf: int = C.SF, bw: int = C.BW_HZ,
            cr: int = C.CR, crc: bool = C.HAS_CRC, implicit: bool = C.IMPLICIT_HEADER) -> float:
    """Semtech AN1200.13 time-on-air in seconds. ``cr`` is 1..4 for 4/5..4/8."""
    ts = symbol_time(sf, bw)
    de = 1 if ts > 0.016 else 0
    num = 8 * pay_len - 4 * sf + 28 + 16 * int(crc) - 20 * int(implicit)
    n_payload = 8 + max(math.ceil(num / (4 * (sf - 2 * de))) * (cr + 4), 0)
    return (preamble + 4.25 + n_payload) * ts


def cfo_hz(cfo_int: int, cfo_frac: float, sf: int = C.SF, bw: int = C.BW_HZ) -> float:
    """frame_sync's CFO estimate (in FFT bins) to Hz."""
    return (cfo_int + cfo_frac) * bw / (1 << sf)
