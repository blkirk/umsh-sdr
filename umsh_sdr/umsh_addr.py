"""Source/destination addresses from a raw UMSH frame, without tshark.

Mirrors the framing in ~/umsh/dissectors/umsh/umsh.lua:
  FCF (ver:2 type:3 S:1 R:1 H:1) [FHOPS] then per type
  BCST       SRC(3|32) OPTIONS payload                      -> dst "all"
  UACK       OPTIONS ack_mic ack_tag                       -> no addresses
  UNIC/UNAR  DST(3) SRC(3|32) SECINFO ...
  MCST       CHANNEL(2) SECINFO OPTIONS [SRC if not encrypted] body MIC
  BUNI/BUAR  CHANNEL(2) SECINFO OPTIONS encrypted(DST, SRC) ...
Addresses render in UMSH's canonical base58 form, cut to 4 characters
(the node-hint budget), with "*" where a 3-byte hint stops being certain.
src_id/dst_id are the key's first 3 bytes (hex): the same node whether it
sent a 3-byte hint or its full key, so they are what tallies group by.
"""
from __future__ import annotations

ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
TYPES = ["BCST", "UACK", "UNIC", "UNAR", "MCST", "RSVD", "BUNI", "BUAR"]
ENC = "enc"  # address is inside the encrypted body


def b58_32(b: bytes) -> str:
    n, out = int.from_bytes(b, "big"), []
    for _ in range(44):
        n, r = divmod(n, 58)
        out.append(ALPHABET[r])
    return "".join(reversed(out))


def hint(b: bytes, budget: int = 4) -> str:
    """Star-truncated base58 of a key prefix (or the first `budget` digits of a full key)."""
    if len(b) == 32:
        return b58_32(b)[:budget]
    lo, hi = b58_32(b + b"\x00" * (32 - len(b))), b58_32(b + b"\xff" * (32 - len(b)))
    out = ""
    for i in range(budget):
        if lo[i] != hi[i]:
            return out + "*"
        out += lo[i]
    return out


OPT_SOURCE_ROUTE, OPT_TRACE = 3, 7


def _nibble(n: int, p: bytes, off: int) -> tuple[int, int]:
    if n == 13:
        return p[off] + 13, off + 1
    if n == 14:
        return p[off] * 256 + p[off + 1] + 269, off + 2
    return n, off


def _options(p: bytes, off: int, bound: int | None = None) -> tuple[dict[int, bytes], int | None]:
    """CoAP-style options block -> ({number: first value}, offset after it or None if malformed)."""
    bound = len(p) if bound is None else bound
    opts, num = {}, 0
    while off < bound:
        b = p[off]
        if b == 0xFF:
            return opts, off + 1
        if b >> 4 == 15 or b & 0x0F == 15:
            return opts, None
        delta, off = _nibble(b >> 4, p, off + 1)
        ln, off = _nibble(b & 0x0F, p, off)
        if off + ln > bound:
            return opts, None
        num += delta
        opts.setdefault(num, p[off:off + ln])
        off += ln
    return opts, off


def _options_end(p: bytes, off: int) -> int | None:
    return _options(p, off)[1]


def transmitter(opts: dict[int, bytes], fhops: int | None) -> dict:
    """Who put the frame on the air, per the frame itself (options.lua `transmitted_by`).

    kind: "sender" (not forwarded: SRC transmitted it), "repeater" (forwarded; `rh` is the
    2-byte repeater hint of the last hop, from the Trace option), "forwarded" (forwarded by a
    hop the frame doesn't name) or "unknown" (source-route hints remain, so either)."""
    tr = opts.get(OPT_TRACE)
    if tr is not None:
        n = len(tr)
        if n % 4 == 0:
            first = tr[0:2] if n else None
        elif n % 2 == 1 and tr[0] in (0xFE, 0xFF):
            first = tr[1:3] if n >= 3 else None
            if first is None and tr[0] == 0xFF:
                return {"kind": "forwarded", "rh": None}
        else:
            first, tr = None, None  # malformed trace: ignored, like a receiver does
        if tr is not None:
            return {"kind": "repeater", "rh": first.hex()} if first else {"kind": "sender", "rh": None}
    if fhops is not None and fhops & 0x0F:
        return {"kind": "forwarded", "rh": None}
    sr = opts.get(OPT_SOURCE_ROUTE)
    if sr is not None:
        return {"kind": "forwarded" if len(sr) == 0 else "unknown", "rh": None}
    return {"kind": "sender", "rh": None}


def parse(p: bytes) -> dict | None:
    """{"type", "src", "dst", "src_id", "dst_id", "tx", "rep"} or None if the frame isn't UMSH v3 / is too short."""
    try:
        if not p or p[0] >> 6 != 3:
            return None
        t, full, off = (p[0] >> 3) & 7, bool(p[0] & 4), 1 + (p[0] & 1)
        fhops = p[1] if p[0] & 1 else None
        slen = 32 if full else 3
        r = {"type": TYPES[t], "src": None, "dst": None, "src_id": None, "dst_id": None,
             "tx": {"kind": "unknown", "rh": None}, "rep": None}
        opts = None
        if t == 0:
            r.update(src=hint(p[off:off + slen]), dst="all", src_id=p[off:off + 3].hex())
            opts, _ = _options(p, off + slen)
        elif t == 1:
            opts, _ = _options(p, off, len(p) - 8)
        elif t in (2, 3):
            if len(p) < off + 3 + slen:
                return r
            r.update(dst=hint(p[off:off + 3]), src=hint(p[off + 3:off + 3 + slen]),
                     dst_id=p[off:off + 3].hex(), src_id=p[off + 3:off + 6].hex())
            off += 3 + slen
            off += 7 if p[off] & 0x10 else 5
            opts, _ = _options(p, off)
        elif t in (4, 6, 7):
            ch, off = p[off:off + 2].hex().upper(), off + 2
            scf = p[off]
            enc = bool(scf & 0x80)
            off += 7 if scf & 0x10 else 5
            opts, o = _options(p, off)
            if t == 4:
                r["dst"] = f"ch {ch}"
                if enc:
                    r["src"] = ENC
                elif o is not None and o + slen <= len(p):
                    r.update(src=hint(p[o:o + slen]), src_id=p[o:o + 3].hex())
            else:
                r.update(src=ENC, dst=f"{ENC} (ch {ch})")
        if opts is not None:
            r["tx"] = transmitter(opts, fhops)
        rh = r["tx"]["rh"]
        r["rep"] = hint(bytes.fromhex(rh), 3) if rh else None  # last hop's repeater hint, base58
        return r
    except (IndexError, ValueError):
        return None


def summarize(rows, clip_thresh: float = 1e-3) -> dict:
    """Per-radio tallies over packet rows (dicts with payload, crc_ok, matched, clipped, snr_db, ts).

    created/received: packets naming the node as SRC / DST, whoever put them on the air.
    last: newest packet the node created (SRC).
    tx: packets this radio itself transmitted, which is what the SDR's signal measurements
    describe: frames it sent unrelayed (it is SRC and the frame shows no forwarding hop) plus
    frames it relayed as the hop named first in the Trace option (repeated counts those).
    tx splits into good (clean), weak (below the floor) and clipped (ADC saturated), and the
    median SNR is over tx. A repeater hint is the first 2 bytes of the repeater's key, so it is
    folded into the one known node whose key starts with it; otherwise it gets its own row.
    Frames with a bad CRC, or relayed by a hop the frame doesn't name, are in the totals only."""
    nodes: dict[str, dict] = {}
    tot = {"decoded": 0, "crc_ok": 0, "crc_bad": 0, "good": 0, "weak": 0, "clipped": 0,
           "attributed": 0, "unattributed": 0}

    def node(nid, label):
        n = nodes.setdefault(nid, {"id": nid, "label": label, "created": 0, "received": 0, "tx": 0, "repeated": 0,
                                   "good": 0, "weak": 0, "clipped": 0, "snr": [], "last": None})
        if not label.endswith("*"):
            n["label"] = label  # a full key gives the 4th character a hint may lack
        return n

    relayed = []  # (rh, quality, snr) resolved once every node key is known
    for r in rows:
        tot["decoded"] += 1
        if not r["crc_ok"]:
            tot["crc_bad"] += 1
            continue
        tot["crc_ok"] += 1
        q = "clipped" if (r.get("clipped") or 0) > clip_thresh else "weak" if not r.get("matched") else "good"
        tot[q] += 1
        a = parse(bytes(r["payload"])) if r.get("payload") else None
        if a and a["src_id"]:
            n = node(a["src_id"], a["src"])
            n["created"] += 1
            n["last"] = max(n["last"] or 0, r["ts"])
        if a and a["dst_id"]:
            node(a["dst_id"], a["dst"])["received"] += 1
        tx = a["tx"] if a else None
        if tx and tx["kind"] == "sender" and a["src_id"]:
            _charge(nodes[a["src_id"]], q, r.get("snr_db"))
            tot["attributed"] += 1
        elif tx and tx["kind"] == "repeater":
            relayed.append((tx["rh"], q, r.get("snr_db")))
            tot["attributed"] += 1
        else:
            tot["unattributed"] += 1
    for rh, q, snr in relayed:
        match = [n for nid, n in nodes.items() if len(nid) == 6 and nid.startswith(rh)]
        n = match[0] if len(match) == 1 else node(rh, hint(bytes.fromhex(rh), 3))
        n["repeated"] += 1
        _charge(n, q, snr)
    out = []
    for n in nodes.values():
        s = sorted(n.pop("snr"))
        n["snr_median"] = s[len(s) // 2] if s else None
        n["is_repeater_hint"] = len(n["id"]) == 4
        out.append(n)
    out.sort(key=lambda n: (-n["tx"], -(n["created"] + n["received"]), n["label"]))
    tot["src_keys"] = sum(1 for n in out if n["created"])
    tot["dst_keys"] = sum(1 for n in out if n["received"])
    return {"nodes": out, "totals": tot}


def _charge(n: dict, q: str, snr):
    n["tx"] += 1
    n[q] += 1
    if snr is not None:
        n["snr"].append(snr)
