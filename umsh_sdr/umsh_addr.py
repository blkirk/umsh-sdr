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


def _options_end(p: bytes, off: int) -> int | None:
    while off < len(p):
        b = p[off]
        if b == 0xFF:
            return off + 1
        dn, ln = b >> 4, b & 0x0F
        off += 1 + (1 if dn == 13 else 2 if dn == 14 else 0)
        if ln == 13:
            ln, off = p[off] + 13, off + 1
        elif ln == 14:
            ln, off = p[off] * 256 + p[off + 1] + 269, off + 2
        off += ln
    return None


def parse(p: bytes) -> dict | None:
    """{"type", "src", "dst", "src_id", "dst_id"} or None if the frame isn't UMSH v3 / is too short."""
    try:
        if not p or p[0] >> 6 != 3:
            return None
        t, full, off = (p[0] >> 3) & 7, bool(p[0] & 4), 1 + (p[0] & 1)
        slen = 32 if full else 3
        r = {"type": TYPES[t], "src": None, "dst": None, "src_id": None, "dst_id": None}
        if t == 0:
            r.update(src=hint(p[off:off + slen]), dst="all", src_id=p[off:off + 3].hex())
        elif t in (2, 3):
            if len(p) < off + 3 + slen:
                return r
            r.update(dst=hint(p[off:off + 3]), src=hint(p[off + 3:off + 3 + slen]),
                     dst_id=p[off:off + 3].hex(), src_id=p[off + 3:off + 6].hex())
        elif t in (4, 6, 7):
            ch, off = p[off:off + 2].hex().upper(), off + 2
            scf = p[off]
            enc = bool(scf & 0x80)
            off += 7 if scf & 0x10 else 5
            if t == 4:
                r["dst"] = f"ch {ch}"
                if enc:
                    r["src"] = ENC
                else:
                    o = _options_end(p, off)
                    if o is not None and o + slen <= len(p):
                        r.update(src=hint(p[o:o + slen]), src_id=p[o:o + 3].hex())
            else:
                r.update(src=ENC, dst=f"{ENC} (ch {ch})")
        return r
    except (IndexError, ValueError):
        return None


def summarize(rows, clip_thresh: float = 1e-3) -> dict:
    """Per-node tallies over packet rows (dicts with payload, crc_ok, matched, clipped, snr_db, ts).

    A node's sent packets are split into good (clean), weak (below the floor, matched=0) and
    clipped (ADC saturated); good + weak + clipped = sent. CRC failures can't be attributed
    to a node, so they only appear in the totals."""
    nodes: dict[str, dict] = {}
    tot = {"decoded": 0, "crc_ok": 0, "crc_bad": 0, "good": 0, "weak": 0, "clipped": 0, "no_addr": 0}

    def node(nid, label):
        n = nodes.setdefault(nid, {"id": nid, "label": label, "sent": 0, "received": 0, "good": 0, "weak": 0,
                                   "clipped": 0, "snr": [], "last": 0.0})
        if not label.endswith("*"):
            n["label"] = label  # a full key gives the 4th character a hint may lack
        return n

    for r in rows:
        tot["decoded"] += 1
        if not r["crc_ok"]:
            tot["crc_bad"] += 1
            continue
        tot["crc_ok"] += 1
        q = "clipped" if (r.get("clipped") or 0) > clip_thresh else "weak" if not r.get("matched") else "good"
        tot[q] += 1
        a = parse(bytes(r["payload"])) if r.get("payload") else None
        if not a or not (a["src_id"] or a["dst_id"]):
            tot["no_addr"] += 1
        if a and a["src_id"]:
            n = node(a["src_id"], a["src"])
            n["sent"] += 1
            n[q] += 1
            n["last"] = max(n["last"], r["ts"])
            if r.get("snr_db") is not None:
                n["snr"].append(r["snr_db"])
        if a and a["dst_id"]:
            n = node(a["dst_id"], a["dst"])
            n["received"] += 1
            n["last"] = max(n["last"], r["ts"])
    out = []
    for n in nodes.values():
        s = sorted(n.pop("snr"))
        n["snr_median"] = s[len(s) // 2] if s else None
        out.append(n)
    out.sort(key=lambda n: (-(n["sent"] + n["received"]), n["label"]))
    tot["src_keys"] = sum(1 for n in out if n["sent"])
    tot["dst_keys"] = sum(1 for n in out if n["received"])
    return {"nodes": out, "totals": tot}
