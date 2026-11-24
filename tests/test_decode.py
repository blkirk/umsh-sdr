"""End-to-end: synthetic UMSH-PHY frames -> .u8 file -> ChannelFlowgraph + Monitor."""
import time

import numpy as np
import pmt
import pytest

from umsh_sdr import config as C

gr = pytest.importorskip("gnuradio.gr")
lora_sdr = pytest.importorskip("gnuradio.lora_sdr")
from gnuradio import blocks, channels  # noqa: E402

SR = C.SAMPLE_RATE
# lora_tx UTF-8-encodes its string input, so synthetic payloads stay ASCII.
PAYLOADS = [bytes((0x21 + (j * 7 + i) % 90) for j in range(30 + 7 * i)) for i in range(4)]


def synth(payloads, ppm, amp, gap_s=0.25):
    tb = gr.top_block()
    tx = lora_sdr.lora_sdr_lora_tx(bw=C.BW_HZ, cr=C.CR, has_crc=True, impl_head=False, samp_rate=SR, sf=C.SF,
                                   ldro_mode=2, frame_zero_padd=int(gap_s * C.BW_HZ), sync_word=[C.SYNC_WORD])
    # a crystal error moves carrier and sample clock together
    ch = channels.channel_model(noise_voltage=0.0, frequency_offset=ppm * 1e-6 * C.CHANNEL_HZ / SR,
                                epsilon=1.0 + ppm * 1e-6, taps=[1.0], noise_seed=3)
    sink = blocks.vector_sink_c()
    tb.connect(tx, ch, sink)
    tb.start()
    for p in payloads:
        # lora_tx takes the payload as a string message; latin-1 keeps bytes 1:1
        tx.lora_sdr_whitening_0.to_basic_block()._post(pmt.intern("msg"), pmt.intern(p.decode("ascii")))
    n_prev, stable = -1, 0
    while stable < 5:
        time.sleep(0.2)
        n = len(sink.data())
        stable = stable + 1 if n == n_prev and n > 0 else 0
        n_prev = n
    tb.stop()
    tb.wait()
    return np.array(sink.data(), dtype=np.complex64) * amp


def to_u8_file(path, x, noise_power, seed=5):
    rng = np.random.default_rng(seed)
    lead = np.zeros(int(5.5 * SR), np.complex64)  # past the monitor warm-up, floors settled
    tail = np.zeros(int(9 * SR), np.complex64)    # let the monitor's lag window flush
    x = np.concatenate([lead, x, tail])
    t = np.arange(x.size) / SR
    x = x * np.exp(2j * np.pi * (C.CHANNEL_HZ - C.LO_HZ) * t)  # place at the UMSH channel
    s = np.sqrt(noise_power / 2)
    x = x + rng.normal(0, s, x.size) + 1j * rng.normal(0, s, x.size)
    iq = np.empty(2 * x.size, np.float32)
    iq[0::2], iq[1::2] = x.real, x.imag
    np.clip(np.round(iq * 128 + 127.4), 0, 255).astype(np.uint8).tofile(path)


def run_pipeline(tmp_path, u8):
    from umsh_sdr.db import DB
    from umsh_sdr.flowgraph import ChannelFlowgraph
    from umsh_sdr.monitor import Monitor
    s = C.Settings(data_dir=str(tmp_path))
    db = DB(s.db_path)
    m = Monitor(db, s)
    m.realtime = False
    m.start()
    fg = ChannelFlowgraph(m.on_frame, m.on_packet, ppm=0, gain_db=0, source="file", file_path=str(u8), file_t0=1000.0)
    fg.run()
    time.sleep(1.0)
    m.drain()
    m.stop()
    return db


@pytest.mark.parametrize("snr_db", [20.0, 0.0])
def test_decode_bit_exact_with_metadata(tmp_path, snr_db):
    amp = 0.1                       # chirp power -20 dBFS
    inband_noise = amp ** 2 / 10 ** (snr_db / 10)
    x = synth(PAYLOADS, ppm=5.0, amp=amp)
    u8 = tmp_path / "s.u8"
    to_u8_file(u8, x, noise_power=4 * inband_noise)  # 2 MHz of noise = 4x the 500 kHz in-band
    db = run_pipeline(tmp_path, u8)
    rows = db.query("SELECT * FROM packets ORDER BY ts")
    assert [bytes(r["payload"]) for r in rows] == PAYLOADS
    assert all(r["crc_ok"] for r in rows)
    want_cfo = 5.0 * 1e-6 * C.CHANNEL_HZ  # carrier 5 ppm above the receiver's tuning reads as +CFO
    for r in rows:
        assert r["cfo_hz"] == pytest.approx(want_cfo, abs=50)
        assert r["snr_db"] == pytest.approx(snr_db, abs=2.0)
    # timing: packets separated by airtime + gap, in order
    ts = [r["ts"] for r in rows]
    assert all(b > a for a, b in zip(ts, ts[1:]))
    if snr_db >= 10:
        assert all(r["matched"] == 1 for r in rows)
    sec = db.query("SELECT avg(floor_dbfs) f FROM band_seconds WHERE band='channel'")[0]["f"]
    assert sec == pytest.approx(10 * np.log10(inband_noise), abs=1.5)
