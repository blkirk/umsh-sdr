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
    assert r == {"type": "UNAR", "src": hint(src), "dst": hint(dst), "src_id": "445566", "dst_id": "112233",
                 "tx": {"kind": "sender", "rh": None}, "rep": None}


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


def _unar(dst, src, full=False, trace=None, fhops=None):
    fcf = (3 << 6) | (3 << 3) | (4 if full else 0) | (1 if fhops is not None else 0)
    head = bytes([fcf]) + (bytes([fhops]) if fhops is not None else b"")
    opts = (bytes([(7 << 4) | len(trace)]) + trace if trace is not None else b"") + b"\xff"
    return head + dst + src + bytes(5) + opts + bytes(8)


def test_transmitter_from_trace_and_fhops():
    a, b = bytes([1, 2, 3]), bytes([4, 5, 6])
    assert parse(_unar(a, b))["tx"] == {"kind": "sender", "rh": None}
    assert parse(_unar(a, b, trace=b""))["tx"]["kind"] == "sender"
    assert parse(_unar(a, b, trace=bytes([0xAA, 0xBB, 0x50, 0x10])))["tx"] == {"kind": "repeater", "rh": "aabb"}
    assert parse(_unar(a, b, trace=bytes([0xAA, 0xBB, 0x50, 0x10])))["rep"] == hint(bytes([0xAA, 0xBB]), 3)
    assert parse(_unar(a, b, trace=bytes([0xFE, 0xCC, 0xDD])))["tx"] == {"kind": "repeater", "rh": "ccdd"}
    assert parse(_unar(a, b, trace=b"\xff"))["tx"]["kind"] == "forwarded"
    assert parse(_unar(a, b, fhops=0x31))["tx"]["kind"] == "forwarded"
    assert parse(_unar(a, b, fhops=0x40))["tx"]["kind"] == "sender"


def test_summarize_charges_the_transmitter():
    a, b = bytes(range(1, 33)), bytes([0x77] * 32)
    rows = [
        # a -> b, sent directly by a: a's radio is measured
        dict(ts=1, payload=_unar(b[:3], a[:3]), crc_ok=1, matched=1, clipped=0, snr_db=10),
        dict(ts=2, payload=_unar(b[:3], a, full=True), crc_ok=1, matched=0, clipped=0, snr_db=-12),
        # b -> a, sent directly by b
        dict(ts=3, payload=_unar(a[:3], b[:3]), crc_ok=1, matched=1, clipped=0.2, snr_db=30),
        # b -> a relayed by repeater a (hint = a's first 2 bytes): a's radio is measured
        dict(ts=4, payload=_unar(a[:3], b[:3], trace=a[:2] + bytes([60, 8])), crc_ok=1, matched=1, clipped=0, snr_db=5),
        # a -> b relayed by an unknown repeater 0xEEEE
        dict(ts=5, payload=_unar(b[:3], a[:3], trace=bytes([0xEE, 0xEE, 60, 8])), crc_ok=1, matched=0, clipped=0, snr_db=-14),
        # a -> b relayed by an unnamed hop: nobody's radio
        dict(ts=6, payload=_unar(b[:3], a[:3], fhops=0x21), crc_ok=1, matched=0, clipped=0, snr_db=-15),
        dict(ts=7, payload=b"\x00junk", crc_ok=0, matched=0, clipped=0, snr_db=None),
    ]
    s = summarize(rows)
    by = {n["id"]: n for n in s["nodes"]}
    na, nb, nr = by[a[:3].hex()], by[b[:3].hex()], by["eeee"]
    assert na["label"] == b58_32(a)[:4] and nr["is_repeater_hint"]
    # created / received / last follow SRC and DST only
    assert (na["created"], na["received"], na["last"]) == (4, 2, 6)
    assert (nb["created"], nb["received"], nb["last"]) == (2, 4, 4)
    assert (nr["created"], nr["received"], nr["last"]) == (0, 0, None)
    # quality and SNR follow the radio that transmitted
    assert (na["tx"], na["repeated"], na["good"], na["weak"], na["clipped"]) == (3, 1, 2, 1, 0)
    assert sorted([10, -12, 5])[1] == na["snr_median"]
    assert (nb["tx"], nb["repeated"], nb["good"], nb["weak"], nb["clipped"], nb["snr_median"]) == (1, 0, 0, 0, 1, 30)
    assert (nr["tx"], nr["repeated"], nr["weak"], nr["snr_median"]) == (1, 1, 1, -14)
    t = s["totals"]
    assert (t["decoded"], t["crc_ok"], t["crc_bad"], t["attributed"], t["unattributed"]) == (7, 6, 1, 5, 1)
    assert (t["src_keys"], t["dst_keys"]) == (2, 2)
