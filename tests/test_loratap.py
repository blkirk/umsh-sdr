import shutil
import subprocess

import pytest

from umsh_sdr import config as C
from umsh_sdr.loratap import LINKTYPE_LORATAP, PcapWriter, loratap_header, read_pcap


def test_header_layout_matches_umshctl():
    h = loratap_header(917_500_000, 500_000, 10, -48, 12.25, 0x12)
    assert len(h) == 15
    assert h[:4] == bytes([0, 0, 0, 15])
    assert int.from_bytes(h[4:8], "big") == 917_500_000
    assert h[8] == 4 and h[9] == 10
    assert h[10] == h[11] == h[12] == 139 - 48
    assert h[13] == 49 and h[14] == 0x12


def test_round_trip(tmp_path):
    p = tmp_path / "x.pcap"
    w = PcapWriter(p)
    w.write_lora(1_700_000_000.25, b"\xc0\x01\x02", freq_hz=C.CHANNEL_HZ, bw_hz=C.BW_HZ, sf=C.SF,
                 rssi_dbm=None, snr_db=-7.5, sync_word=C.SYNC_WORD)
    w.close()
    w = PcapWriter(p)  # append keeps a single global header
    w.write_lora(1_700_000_001.0, b"\xaa", freq_hz=C.CHANNEL_HZ, bw_hz=C.BW_HZ, sf=C.SF,
                 rssi_dbm=-90, snr_db=None, sync_word=C.SYNC_WORD)
    w.close()
    recs = read_pcap(p)
    assert [r.linktype for r in recs] == [LINKTYPE_LORATAP] * 2
    assert recs[0].payload == b"\xc0\x01\x02" and recs[0].rssi_dbm is None and recs[0].snr_db == -7.5
    assert recs[1].payload == b"\xaa" and recs[1].rssi_dbm == -90 and recs[1].snr_db is None
    assert recs[0].ts == pytest.approx(1_700_000_000.25)


@pytest.mark.skipif(shutil.which("tshark") is None, reason="tshark not installed")
def test_umsh_dissector_claims_sdr_frames(tmp_path):
    from umsh_sdr.dissect import UMSH_DISSECTOR, dissect_pcap
    vectors = UMSH_DISSECTOR.parent / "test_vectors.pcap"
    if not vectors.exists():
        pytest.skip("UMSH test vectors not found")
    src = read_pcap(vectors)
    p = tmp_path / "v.pcap"
    w = PcapWriter(p)
    for r in src[:5]:
        w.write_lora(r.ts, r.payload, freq_hz=C.CHANNEL_HZ, bw_hz=C.BW_HZ, sf=C.SF, rssi_dbm=-80, snr_db=5,
                     sync_word=C.SYNC_WORD)
    w.close()
    res = dissect_pcap(p)
    assert res and all("umsh" in r["_protocols"] for r in res)
