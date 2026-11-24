"""GNU Radio flowgraph for channel mode.

    source (RTL-SDR @ LO 917.0 MHz, 2 MS/s, ppm-corrected; or a recorded .u8 file)
      ├── SpectrumTap  -> dsp.Frame every 10.24 ms -> on_frame()
      └── freq_xlating_fir (+500 kHz, 280 kHz LPF)
            └── frame_sync ─(CfoTap: CFO + header sample position)
                  └── fft_demod → gray_mapping → deinterleaver → hamming_dec
                        → header_decoder → dewhitening → crc_verif → FrameSink -> on_packet()

The LoRa chain mirrors gr-lora_sdr's lora_rx hier block, built out so
frame_sync's tags (CFO) and input position (timing) can be tapped.
"""
from __future__ import annotations

import time
import threading
from collections import deque
from dataclasses import dataclass

import numpy as np
import pmt
from gnuradio import blocks, filter, gr, lora_sdr
from gnuradio.filter import firdes

from . import config as C
from . import lora
from .dsp import FrameAnalyzer, StreamFramer


@dataclass
class DecodedFrame:
    payload: bytes
    crc_ok: bool
    cr: int
    decoded_at: float          # wall clock
    t_header: float | None     # stream time when the header was decoded (approx.)
    cfo_hz: float | None
    snr_preamble_db: float | None = None  # gr-lora_sdr preamble estimate (patched frame_sync)


class SpectrumTap(gr.sync_block):
    def __init__(self, on_frame, t0_fn):
        gr.sync_block.__init__(self, "umsh_spectrum_tap", [np.complex64], [])
        self.on_frame = on_frame
        self.t0_fn = t0_fn
        self.framer: StreamFramer | None = None
        self.analyzer = FrameAnalyzer()

    def work(self, input_items, output_items):
        x = input_items[0]
        if self.framer is None:
            self.framer = StreamFramer(self.analyzer, self.t0_fn())
        for f in self.framer.feed(x):
            self.on_frame(f)
        return len(x)


class CfoTap(gr.sync_block):
    """Reads frame_sync output tags: CFO at each header, confirmed by the
    payload tag that follows only when the header decoded."""

    def __init__(self, frame_sync, sample_time):
        gr.sync_block.__init__(self, "umsh_cfo_tap", [np.complex64], [])
        self.fs = frame_sync
        self.sample_time = sample_time
        self.pending = None
        self.confirmed: deque = deque(maxlen=64)
        self.lock = threading.Lock()

    def work(self, input_items, output_items):
        n = len(input_items[0])
        for tag in self.get_tags_in_window(0, 0, n, pmt.intern("frame_info")):
            d = tag.value
            if pmt.to_bool(pmt.dict_ref(d, pmt.intern("is_header"), pmt.PMT_F)):
                ci = pmt.to_long(pmt.dict_ref(d, pmt.intern("cfo_int"), pmt.from_long(0)))
                cf = pmt.to_float(pmt.dict_ref(d, pmt.intern("cfo_frac"), pmt.from_float(0.0)))
                snr = pmt.dict_ref(d, pmt.intern("snr"), pmt.PMT_NIL)
                snr = None if pmt.is_null(snr) else pmt.to_float(snr)
                self.pending = (lora.cfo_hz(ci, cf), self.sample_time(self.fs.nitems_read(0)), snr, time.time())
            elif self.pending is not None:
                with self.lock:
                    self.confirmed.append(self.pending)
                self.pending = None
        return n

    def take(self):
        now = time.time()
        with self.lock:
            while self.confirmed:
                cfo, t_hdr, snr, wall = self.confirmed.popleft()
                if now - wall < 10.0:
                    return cfo, t_hdr, snr
        return None, None, None


class FrameSink(gr.sync_block):
    def __init__(self, on_packet, cfo_tap: CfoTap):
        gr.sync_block.__init__(self, "umsh_frame_sink", [np.uint8], [])
        self.on_packet = on_packet
        self.cfo_tap = cfo_tap
        self.cur = None  # [info, bytearray]

    def _start(self, d):
        info = {k: pmt.to_python(pmt.dict_ref(d, pmt.intern(k), pmt.PMT_NIL))
                for k in ("pay_len", "crc_valid", "cr")}
        if info["pay_len"] is None:
            return
        self.cur = [info, bytearray()]

    def _emit(self):
        info, buf = self.cur
        self.cur = None
        cfo, t_hdr, snr = self.cfo_tap.take()
        self.on_packet(DecodedFrame(bytes(buf), bool(info.get("crc_valid")), int(info.get("cr") or 0),
                                    time.time(), t_hdr, cfo, snr))

    def work(self, input_items, output_items):
        x = input_items[0]
        n = len(x)
        base = self.nitems_read(0)
        tags = sorted(self.get_tags_in_window(0, 0, n, pmt.intern("frame_info")), key=lambda t: t.offset)
        pos = 0
        for tag in tags:
            rel = tag.offset - base
            if not pmt.dict_has_key(tag.value, pmt.intern("crc_valid")) and not pmt.dict_has_key(tag.value, pmt.intern("pay_len")):
                continue
            self._consume(x[pos:rel])
            pos = rel
            if self.cur is not None and self.cur[1]:
                self._emit()  # previous frame came up short; flush what we have
            self._start(tag.value)
        self._consume(x[pos:])
        return n

    def _consume(self, chunk):
        if self.cur is None or len(chunk) == 0:
            return
        info, buf = self.cur
        need = int(info["pay_len"]) - len(buf)
        buf.extend(bytes(chunk[:need]))
        if len(buf) >= int(info["pay_len"]):
            self._emit()


class ChannelFlowgraph(gr.top_block):
    def __init__(self, on_frame, on_packet, *, ppm: float, gain_db: float,
                 source: str = "rtl", file_path: str | None = None, file_t0: float | None = None,
                 device_args: str = "numchan=1 rtl=0"):
        gr.top_block.__init__(self, "umsh_channel")
        sr = C.SAMPLE_RATE
        self._t0 = None
        self._t0_lock = threading.Lock()
        self.file_t0 = file_t0

        if source == "rtl":
            import osmosdr
            self.src = osmosdr.source(args=device_args)
            self.src.set_sample_rate(sr)
            self.src.set_freq_corr(ppm)
            self.src.set_center_freq(C.LO_HZ)
            self.src.set_gain_mode(False)
            self.src.set_gain(gain_db)
            self.src.set_bandwidth(0)
            head = self.src
        else:  # rtl_sdr-style interleaved uint8 file
            fsrc = blocks.file_source(gr.sizeof_char, file_path, False)
            u2f = blocks.uchar_to_float()
            sub = blocks.add_const_ff(-127.4)
            scl = blocks.multiply_const_ff(1.0 / 128)
            dei = blocks.deinterleave(gr.sizeof_float)
            f2c = blocks.float_to_complex()
            self.connect(fsrc, u2f, sub, scl, dei)
            self.connect((dei, 0), (f2c, 0))
            self.connect((dei, 1), (f2c, 1))
            head = f2c
            self.src = fsrc

        self.spec = SpectrumTap(on_frame, self.t0)
        self.connect(head, self.spec)

        taps = firdes.low_pass(1.0, sr, 280e3, 80e3)
        self.xlate = filter.freq_xlating_fir_filter_ccc(1, taps, C.CHANNEL_HZ - C.LO_HZ, sr)
        soft = True
        os_factor = sr // C.BW_HZ
        self.fs = lora_sdr.frame_sync(C.CHANNEL_HZ, C.BW_HZ, C.SF, C.IMPLICIT_HEADER, [C.SYNC_WORD], os_factor, 8)
        fft = lora_sdr.fft_demod(soft, True)
        gray = lora_sdr.gray_mapping(soft)
        deint = lora_sdr.deinterleaver(soft)
        ham = lora_sdr.hamming_dec(soft)
        hdr = lora_sdr.header_decoder(C.IMPLICIT_HEADER, C.CR, 255, C.HAS_CRC, 2, False)
        dew = lora_sdr.dewhitening()
        crc = lora_sdr.crc_verif(0, False)
        self.cfo = CfoTap(self.fs, self.sample_time)
        self.sink = FrameSink(on_packet, self.cfo)
        self.connect(head, self.xlate, self.fs, fft, gray, deint, ham, hdr, dew, crc, self.sink)
        self.connect(self.fs, self.cfo)
        self.msg_connect((hdr, "frame_info"), (self.fs, "frame_info"))
        self._blocks = (fft, gray, deint, ham, hdr, dew, crc)

    def t0(self) -> float:
        """Stream time of sample 0: wall clock at first data (RTL) or the file's start time."""
        with self._t0_lock:
            if self._t0 is None:
                self._t0 = self.file_t0 if self.file_t0 is not None else time.time()
            return self._t0

    def sample_time(self, n: int) -> float:
        return self.t0() + n / C.SAMPLE_RATE

    def set_gain(self, gain_db: float) -> None:
        if hasattr(self.src, "set_gain"):
            self.src.set_gain(gain_db)
