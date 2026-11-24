from umsh_sdr.umsh_addr import b58_32, hint, parse, summarize


def test_full_key_and_hint():
    key = bytes(range(1, 33))
    assert hint(key) == b58_32(key)[:4]
    h = hint(key[:3])
    assert len(h) == 4 and b58_32(key).startswith(h.rstrip("*"))


def test_unicast_ack_req():
    dst, src = bytes([0x11, 0x22, 0x33]), bytes([0x44, 0x55, 0x66])
    fcf = (3 << 6) | (3 << 3)  # v3, UNAR, hint SRC, no FHOPS
    r = parse(bytes([fcf]) + dst + src + bytes(5) + b"\xff" + bytes(8))
    assert r == {"type": "UNAR", "src": hint(src), "dst": hint(dst), "src_id": "445566", "dst_id": "112233"}


def test_broadcast_with_fhops_full_src_and_ack():
    key = bytes(range(32))
    r = parse(bytes([(3 << 6) | 4 | 1, 0x50]) + key + b"\xff")
    assert (r["type"], r["src"], r["dst"], r["src_id"]) == ("BCST", b58_32(key)[:4], "all", "000102")
    assert parse(bytes([(3 << 6) | (1 << 3)]) + bytes(8))["src"] is None


def test_multicast_clear_and_encrypted():
    base = bytes([(3 << 6) | (4 << 3), 0xAB, 0xCD])
    src = bytes([9, 8, 7])
    clear = parse(base + bytes([0x00]) + bytes(4) + b"\xff" + src + bytes(6))
    assert (clear["src"], clear["dst"], clear["src_id"]) == (hint(src), "ch ABCD", "090807")
    assert parse(base + bytes([0x80]) + bytes(4) + b"\xff" + bytes(10))["src"] == "enc"


def test_not_umsh():
    assert parse(b"") is None and parse(b"\x00\x01") is None


def _unar(dst, src, full=False):
    fcf = (3 << 6) | (3 << 3) | (4 if full else 0)
    return bytes([fcf]) + dst + src + bytes(5) + b"\xff" + bytes(8)


def test_summarize_groups_hint_and_full_key():
    a, b = bytes(range(1, 33)), bytes([0x77] * 32)
    rows = [
        dict(ts=1, payload=_unar(b[:3], a[:3]), crc_ok=1, matched=1, clipped=0, snr_db=10),
        dict(ts=2, payload=_unar(b[:3], a, full=True), crc_ok=1, matched=0, clipped=0, snr_db=-12),
        dict(ts=3, payload=_unar(a[:3], b[:3]), crc_ok=1, matched=1, clipped=0.2, snr_db=30),
        dict(ts=4, payload=b"\x00junk", crc_ok=0, matched=0, clipped=0, snr_db=None),
    ]
    s = summarize(rows)
    na, nb = (next(n for n in s["nodes"] if n["id"] == k[:3].hex()) for k in (a, b))
    assert len(s["nodes"]) == 2 and na["label"] == b58_32(a)[:4]
    assert (na["sent"], na["received"], na["good"], na["weak"], na["clipped"]) == (2, 1, 1, 1, 0)
    assert (nb["sent"], nb["received"], nb["clipped"], nb["last"]) == (1, 2, 1, 3)
    t = s["totals"]
    assert (t["decoded"], t["crc_ok"], t["crc_bad"], t["good"], t["weak"], t["clipped"]) == (4, 3, 1, 1, 1, 1)
    assert (t["src_keys"], t["dst_keys"]) == (2, 2)
