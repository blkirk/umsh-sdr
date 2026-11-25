"""Find bursts in a short IQ recording and characterise each one.

For every burst: time, centre frequency, occupied bandwidth, duration and SNR, then a
modulation guess (OOK/ASK, 2-FSK/4-FSK, unmodulated carrier, chirp/LoRa, BPSK/QPSK) with
symbol rate and FSK deviation, then a bit-slicing pass that looks for a preamble, known sync
words and Manchester coding. These are heuristics: they say what a burst is consistent with,
not what it is.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

# ---------------------------------------------------------------- channelizing


def channelize(path, fs: float, shift_hz: float, decim: int, half_bw_hz: float, chunk: int = 1 << 21):
    """rtl_sdr .cu8 file -> complex64 baseband centred on `shift_hz` (relative to the LO),
    low-pass filtered to +-half_bw_hz and decimated by `decim`. Also returns the share of
    raw samples at the ADC rails (0 or 255), i.e. clipping."""
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    n = raw.size // 2
    n -= n % decim
    out = np.empty(n // decim, np.complex64)
    h = None
    if decim > 1:
        edge = min(fs / decim / 2, half_bw_hz + 0.25 * (fs / decim / 2 - half_bw_hz) + 20e3)
        trans = max(edge - half_bw_hz, 10e3)
        taps = int(4 * fs / trans) | 1
        h = signal.firwin(min(taps, 1023), (half_bw_hz + edge) / 2, fs=fs).astype(np.float32)
        hist = ((len(h) - 1 + decim - 1) // decim) * decim
        prev = np.zeros(hist, np.complex64)
    chunk -= chunk % max(decim, 1)
    rails = 0
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        u = np.asarray(raw[2 * s:2 * e])
        rails += int(np.count_nonzero((u == 0) | (u == 255)))
        x = (u.astype(np.float32) - 127.5) / 127.5
        x = (x[0::2] + 1j * x[1::2]).astype(np.complex64)
        x -= x.mean()  # the RTL DC spike
        if shift_hz:
            ph = (shift_hz / fs * np.arange(s, e, dtype=np.float64)) % 1.0
            x *= np.exp(-2j * np.pi * ph).astype(np.complex64)
        if h is None:
            out[s:e] = x
            continue
        seg = np.concatenate([prev, x])
        y = signal.upfirdn(h, seg, down=decim)
        k0 = hist // decim
        out[s // decim:e // decim] = y[k0:k0 + (e - s) // decim]
        prev = seg[-hist:]
    return out, (rails / max(1, 2 * n))


# ---------------------------------------------------------------- spectrogram + bursts


def spectrogram(y: np.ndarray, fs: float, nfft: int = 256, frame_s: float = 0.5e-3):
    """Non-overlapping Hann periodograms, `avg` of them averaged per row (~frame_s).
    Returns (power[rows, nfft] linear, bin centre freqs relative to centre, row seconds)."""
    avg = max(1, int(round(frame_s * fs / nfft)))
    step = nfft * avg
    rows = len(y) // step
    w = signal.windows.hann(nfft, sym=False).astype(np.float32)
    norm = float(np.sum(w) ** 2)
    S = np.empty((rows, nfft), np.float32)
    blk = max(1, 8192 // avg)
    for r0 in range(0, rows, blk):
        r1 = min(rows, r0 + blk)
        seg = y[r0 * step:r1 * step].reshape(-1, nfft) * w
        P = np.abs(np.fft.fftshift(np.fft.fft(seg, axis=1), axes=1)) ** 2 / norm
        S[r0:r1] = P.reshape(r1 - r0, avg, nfft).mean(1)
    f = (np.arange(nfft) - nfft // 2) * fs / nfft
    return S, f, step / fs


def detect(S: np.ndarray, f: np.ndarray, dt: float, thresh_db: float = 8.0, max_bursts: int = 2000):
    """Bursts = connected time-frequency regions over the per-bin floor (median over time).
    Regions overlapping in time and within 30 kHz of each other merge (FSK tones, OOK gaps)."""
    Sm = ndimage.uniform_filter(S, size=(2, 3), mode="nearest")
    floor = np.median(Sm, axis=0)
    floor = np.maximum(floor, 1e-20)
    mask = Sm > floor * 10 ** (thresh_db / 10)
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3), bool))
    lab, nlab = ndimage.label(mask, structure=np.ones((3, 3), bool))
    df = f[1] - f[0]
    boxes = []
    for i, sl in enumerate(ndimage.find_objects(lab)):
        if sl is None:
            continue
        t0, t1 = sl[0].start, sl[0].stop
        if t1 - t0 < 2 or np.count_nonzero(lab[sl] == i + 1) < 4:
            continue
        boxes.append([t0, t1, sl[1].start, sl[1].stop])
    boxes.sort()
    merged = []
    gap = max(1, int(round(30e3 / df)))
    for b in boxes:
        for m in merged:
            ov = min(b[1], m[1]) - max(b[0], m[0])
            if ov >= 0.5 * min(b[1] - b[0], m[1] - m[0]) and b[2] <= m[3] + gap and m[2] <= b[3] + gap:
                m[:] = [min(m[0], b[0]), max(m[1], b[1]), min(m[2], b[2]), max(m[3], b[3])]
                break
        else:
            merged.append(list(b))
    out = []
    for t0, t1, k0, k1 in merged[:max_bursts]:
        sub = S[t0:t1, k0:k1].mean(0) - floor[k0:k1]
        sub = np.maximum(sub, 0)
        tot = float(sub.sum())
        if tot <= 0:
            continue
        c = np.cumsum(sub) / tot
        lo, hi = int(np.searchsorted(c, 0.005)), int(np.searchsorted(c, 0.995))
        fc = float(np.sum(sub * f[k0:k1]) / tot)
        noise = float(floor[k0:k1].sum())
        out.append({
            "t0": t0 * dt, "t1": t1 * dt, "dur_ms": (t1 - t0) * dt * 1e3,
            "f_lo": float(f[k0] - df / 2), "f_hi": float(f[k1 - 1] + df / 2), "fc": fc,
            "bw": float(max(df, (hi - lo + 1) * df)),
            "snr_db": float(10 * np.log10(tot / noise)) if noise > 0 else None,
            "peak_db": float(10 * np.log10(S[t0:t1, k0:k1].sum(1).max() + 1e-20)),
            "edge": bool(k0 == 0 or k1 == len(f)),
        })
    return out, floor


# ---------------------------------------------------------------- per-burst analysis


def _mavg(x, n):
    if n <= 1:
        return x
    k = np.ones(n, np.float32) / n
    return np.convolve(x, k, mode="same")


def _kmeans1d(x, k, iters=30):
    c = np.percentile(x, np.linspace(100 / (2 * k), 100 - 100 / (2 * k), k))
    for _ in range(iters):
        lab = np.argmin(np.abs(x[:, None] - c[None, :]), axis=1)
        new = np.array([x[lab == j].mean() if np.any(lab == j) else c[j] for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    lab = np.argmin(np.abs(x[:, None] - c[None, :]), axis=1)
    resid = x - c[lab]
    share = np.bincount(lab, minlength=k) / len(x)
    return c, float(np.sqrt(np.mean(resid ** 2))), share


def _hysteresis(x, h):
    out = np.empty(len(x), np.int8)
    st = x[0] > 0
    for i, v in enumerate(x.tolist()):
        if st and v < -h:
            st = False
        elif not st and v > h:
            st = True
        out[i] = st
    return out


def estimate_baud(b: np.ndarray, fs: float, rmin: float, rmax: float):
    """Symbol rate from a sliced 0/1 stream: candidates from the transition train's spectral
    lines and from the shortest runs, refined so every run is a whole number of symbols; the
    longest symbol period that explains >= 75% of runs wins (shorter ones are subharmonics)."""
    tr = np.flatnonzero(b[1:] != b[:-1]) + 1
    if len(tr) < 6:
        return None, 0.0
    runs = np.diff(tr).astype(np.float64)
    x = np.zeros(len(b), np.float32)
    x[tr] = 1
    nf = 1 << int(np.ceil(np.log2(len(x) * 4)))
    X = np.abs(np.fft.rfft(x - x.mean(), nf))
    fr = np.fft.rfftfreq(nf, 1 / fs)
    cands = []
    m = np.flatnonzero((fr >= rmin) & (fr <= rmax))
    if m.size:
        cands += [fs / fr[i] for i in m[np.argsort(X[m])[-6:]]]
    srt = np.sort(runs)
    cands.append(float(np.median(srt[:max(1, len(srt) // 5)])))
    best = (None, 0.0)
    for T in cands:
        if not (fs / rmax * 0.8 <= T <= fs / rmin):
            continue
        for _ in range(5):
            n = np.maximum(1, np.round(runs / T))
            T = runs.sum() / n.sum()
        r = runs / T
        fit = float(np.mean((np.abs(r - np.round(r)) < 0.25) & (np.round(r) >= 1)))
        if fit >= 0.75 and (best[0] is None or best[1] < 0.75 or T > best[0] * 1.05):
            best = (T, fit)
        elif best[1] < 0.75 and fit > best[1]:
            best = (T, fit)
    if best[0] is None or best[1] < 0.6:
        return None, best[1]
    return fs / best[0], best[1]


def slice_bits(b: np.ndarray, fs: float, baud: float) -> str:
    T = fs / baud
    edges = np.concatenate([[0], np.flatnonzero(b[1:] != b[:-1]) + 1, [len(b)]])
    out = []
    for a, e in zip(edges[:-1], edges[1:]):
        k = int(round((e - a) / T))
        if k:
            out.append(("1" if b[a] else "0") * k)
    return "".join(out)


def _line(z, power):
    """Strength (dB over median) and frequency of the strongest line in z**power."""
    w = z ** power
    nf = 1 << int(np.ceil(np.log2(max(len(w), 64))))
    P = np.abs(np.fft.fft(w * signal.windows.hann(len(w)), nf)) ** 2
    i = int(np.argmax(P))
    return float(10 * np.log10(P[i] / (np.median(P) + 1e-30))), float(np.fft.fftfreq(nf)[i])


def classify(z: np.ndarray, fs: float, bw: float, snr_db: float | None) -> dict:
    """z: one burst at baseband, centred and filtered to its occupied bandwidth."""
    r = {"modulation": "unknown", "reasons": [], "baud": None, "baud_fit": None, "deviation_hz": None,
         "levels_hz": None, "bits": None, "chirp_rate_hz_s": None}
    n = len(z)
    if n < 64:
        r["modulation"], r["reasons"] = "too short", [f"{n} samples"]
        return r
    if snr_db is not None and snr_db < 10:
        r["modulation"], r["reasons"] = "too weak", [f"SNR {snr_db:.1f} dB"]
        return r
    L = max(1, int(fs / (4 * max(bw, 2e3))))
    a = np.abs(z)
    k = max(1, n // 10)
    body = slice(k, n - k)
    sm = _mavg(a, L)
    p90, p10 = np.percentile(sm[body], 90), np.percentile(sm[body], 10)
    off_frac = float(np.mean(sm[body] < 0.35 * p90))
    cv = float(np.std(a[body]) / (np.mean(a[body]) + 1e-12))
    d = np.angle(z[1:] * np.conj(z[:-1])) * fs / (2 * np.pi)
    ds = _mavg(d.astype(np.float32), L)
    on = sm[:-1] > 0.5 * p90
    bmask = np.zeros(n - 1, bool)
    bmask[k:n - 1 - k] = True
    dv = ds[on & bmask]
    r.update(envelope_cv=round(cv, 3), off_fraction=round(off_frac, 3))
    if dv.size < 32:
        r["reasons"].append("too few strong samples")
        return r
    rmax = min(fs / 4, 1.5 * bw + 1e3)

    # OOK / ASK: the envelope drops to the noise inside the burst
    if off_frac > 0.15 and cv > 0.35:
        r["modulation"] = "OOK/ASK"
        r["reasons"].append(f"envelope off {100 * off_frac:.0f}% of the burst (CV {cv:.2f})")
        b = _hysteresis(sm - (p90 + p10) / 2, 0.1 * (p90 - p10))
        r["baud"], r["baud_fit"] = estimate_baud(b, fs, 100, rmax)
        if r["baud"]:
            r["bits"] = slice_bits(b, fs, r["baud"])
        return r

    # chirp: the instantaneous frequency ramps linearly through most of the burst
    W = max(16, len(ds) // 48)
    good = slopes = 0
    sl = []
    t = np.arange(W) / fs
    for i in range(1, len(ds) // W - 1):
        seg = ds[i * W:(i + 1) * W]
        if not on[i * W:(i + 1) * W].all():
            continue
        slopes += 1
        p = np.polyfit(t, seg, 1)
        res = seg - np.polyval(p, t)
        r2 = 1 - res.var() / (seg.var() + 1e-9)
        if r2 > 0.9 and abs(p[0]) * W / fs > 0.1 * bw:
            good += 1
            sl.append(p[0])
    same_sign = max(np.mean(np.array(sl) > 0), np.mean(np.array(sl) < 0)) if sl else 0
    if slopes >= 8 and good / slopes > 0.4 and same_sign > 0.7:
        rate = float(np.median(np.abs(sl)))
        r["modulation"], r["chirp_rate_hz_s"] = "chirp (LoRa-like)", rate
        r["reasons"].append(f"frequency ramps linearly in {good}/{slopes} windows")
        sf = np.log2(bw ** 2 / rate) if rate > 0 else None
        if sf is not None and 5.5 <= sf <= 12.5:
            r["lora_sf_estimate"] = int(round(sf))
            r["reasons"].append(f"sweep rate fits LoRa SF{int(round(sf))} at BW {bw / 1e3:.0f} kHz")
        return r

    # FSK: instantaneous frequency clusters into 2 or 4 levels
    c2, s2, sh2 = _kmeans1d(dv, 2)
    sep = abs(c2[1] - c2[0])
    if sep / (s2 + 1e-9) > 3.0 and sh2.min() > 0.1 and sep > 500:
        c4, s4, sh4 = _kmeans1d(dv, 4)
        gaps = np.diff(np.sort(c4))
        four = s4 < 0.35 * s2 and sh4.min() > 0.1 and gaps.max() / max(gaps.min(), 1) < 1.5
        levels = np.sort(c4) if four else np.sort(c2)
        r["modulation"] = "4-FSK" if four else "2-FSK"
        r["levels_hz"] = [round(float(v)) for v in levels]
        r["deviation_hz"] = round(float(levels[-1] - levels[0]) / 2)  # outer deviation
        r["reasons"].append(f"frequency settles at {len(levels)} levels {sep / 1e3:.1f} kHz apart"
                            f" (separation {sep / (s2 + 1e-9):.1f}x the spread)")
        if not four:
            mid = (c2[0] + c2[1]) / 2
            seg = np.flatnonzero(on)
            s = ds[seg[0]:seg[-1] + 1] - mid
            b = _hysteresis(s, 0.15 * sep / 2)
            r["baud"], r["baud_fit"] = estimate_baud(b, fs, 100, rmax)
            if r["baud"]:
                r["bits"] = slice_bits(b, fs, r["baud"])
                r["mod_index"] = round(sep / r["baud"], 2)
        return r

    # unmodulated carrier
    fstd = float(np.std(dv))
    if bw <= 3 * max(fs / 256, 1e3) and fstd < max(0.3 * bw, 300):
        r["modulation"] = "CW (unmodulated carrier)"
        r["reasons"].append(f"narrow ({bw / 1e3:.1f} kHz), steady frequency (sd {fstd:.0f} Hz), constant envelope")
        return r

    # PSK: constant envelope, and z^2 / z^4 shows a line
    if cv < 0.35:
        l2, f2 = _line(z, 2)
        l4, f4 = _line(z, 4)
        if l2 > 18:
            r["modulation"] = "BPSK"
            r["reasons"].append(f"constant envelope; z^2 has a {l2:.0f} dB spectral line")
            w = z * np.exp(-1j * np.pi * f2 * np.arange(n))
            w = w * np.exp(-1j * np.angle(np.sum(w ** 2)) / 2)
            b = _hysteresis(_mavg(w.real.astype(np.float32), L), 0.1 * np.std(w.real))
            r["baud"], r["baud_fit"] = estimate_baud(b, fs, 100, rmax)
            if r["baud"]:
                r["bits"] = slice_bits(b, fs, r["baud"])
            return r
        if l4 > 18:
            r["modulation"] = "QPSK"
            r["reasons"].append(f"constant envelope; z^4 has a {l4:.0f} dB spectral line")
            return r
        r["modulation"] = "constant envelope, unresolved"
        r["reasons"].append("constant envelope, but frequency doesn't cluster and no PSK line: "
                            "possibly GFSK/MSK with heavy filtering, analog FM, or too little SNR")
        return r
    r["modulation"] = "noise-like / multicarrier"
    r["reasons"].append(f"envelope varies (CV {cv:.2f}) without on/off keying: OFDM, DSSS or noise")
    return r


# ---------------------------------------------------------------- bit-level decoding

def _hexbits(h: str, nbits: int | None = None) -> str:
    b = bin(int(h, 16))[2:].zfill(4 * len(h))
    return b[-nbits:] if nbits else b


SYNC_WORDS = [  # (label, bit pattern, domain)
    ("Semtech/HopeRF FSK sync 0x2DD4 (SX12xx/RFM default)", _hexbits("2DD4"), "nrz"),
    ("TI CC1101/CC1310 sync 0xD391 (default)", _hexbits("D391"), "nrz"),
    ("IEEE 802.15.4g SUN-FSK SFD 0x904E (Wi-SUN, uncoded)", _hexbits("904E"), "nrz"),
    ("IEEE 802.15.4g SUN-FSK SFD 0x6F4E (Wi-SUN, FEC)", _hexbits("6F4E"), "nrz"),
    ("IEEE 802.15.4g SUN-FSK SFD 0x7A0E (Wi-SUN, uncoded, alt)", _hexbits("7A0E"), "nrz"),
    ("IEEE 802.15.4g SUN-FSK SFD 0x632D (Wi-SUN, FEC, alt)", _hexbits("632D"), "nrz"),
    ("Itron ERT SCM+/IDM sync 0x16A3", _hexbits("16A3"), "nrz"),
    ("Itron ERT SCM preamble 0x1F2A60 (21 bits)", _hexbits("1F2A60", 21), "manchester"),
    ("Itron ERT SCM+/IDM sync 0x16A3", _hexbits("16A3"), "manchester"),
    ("Z-Wave preamble end + SOF 0x55F0", _hexbits("55F0"), "nrz"),
]


def _alternating_run(bits: str) -> tuple[int, int]:
    """(start, length) of the longest 0101... run."""
    best, start, run = (0, 0), 0, 1
    for i in range(1, len(bits)):
        if bits[i] != bits[i - 1]:
            run += 1
        else:
            if run > best[1]:
                best = (start, run)
            start, run = i, 1
    if run > best[1]:
        best = (start, run)
    return best


def manchester(bits: str):
    """Best of the two pair alignments: (offset, error rate, decoded bits with 10->1, 01->0)."""
    best = None
    for off in (0, 1):
        pairs = [bits[i:i + 2] for i in range(off, len(bits) - 1, 2)]
        if len(pairs) < 16:
            continue
        err = sum(p in ("00", "11") for p in pairs) / len(pairs)
        dec = "".join("1" if p == "10" else "0" if p == "01" else "?" for p in pairs)
        if best is None or err < best[1]:
            best = (off, err, dec)
    return best


def to_hex(bits: str) -> str:
    bits = bits[:len(bits) - len(bits) % 8]
    return "".join(f"{int(bits[i:i + 8], 2):02X}" for i in range(0, len(bits), 8))


def decode_bits(bits: str) -> dict:
    inv = bits.translate(str.maketrans("01", "10"))
    pstart, plen = _alternating_run(bits)
    out = {"n_bits": len(bits), "preamble_bits": plen if plen >= 8 else 0, "sync": [], "manchester": None,
           "frame_hex": None, "frame_from": None}
    man = manchester(bits)
    if man and man[1] <= 0.12:
        out["manchester"] = {"offset": man[0], "error_rate": round(man[1], 3), "bits": man[2][:512]}
    for label, pat, dom in SYNC_WORDS:
        if dom == "manchester":
            if not out["manchester"]:
                continue
            spaces = [("manchester", out["manchester"]["bits"]),
                      ("manchester, inverted", out["manchester"]["bits"].translate(str.maketrans("01", "10")))]
        else:
            spaces = [("nrz", bits), ("nrz, inverted", inv)]
        for name, sp in spaces:
            i = sp.find(pat)
            if i >= 0:
                out["sync"].append({"label": label, "where": name, "bit": i})
                if out["frame_hex"] is None:
                    out["frame_hex"], out["frame_from"] = to_hex(sp[i + len(pat):])[:128], f"after {label} ({name})"
    if out["frame_hex"] is None:
        if out["manchester"]:
            out["frame_hex"], out["frame_from"] = to_hex(out["manchester"]["bits"])[:128], "Manchester-decoded, after preamble"
        elif plen >= 8:
            out["frame_hex"], out["frame_from"] = to_hex(bits[pstart + plen:])[:128], "after preamble (NRZ, 1 = high)"
        else:
            out["frame_hex"], out["frame_from"] = to_hex(bits)[:128], "whole burst (NRZ, 1 = high)"
    return out


# ---------------------------------------------------------------- one burst, end to end


def analyse_burst(y: np.ndarray, fs: float, b: dict) -> dict:
    if b.get("edge"):
        return {"modulation": "wider than passband", "reasons": [
            "the burst reaches the edge of the analysed passband, so it is part of a wider signal; "
            "capture a wider bandwidth bucket to see all of it"]}
    pad = int(0.2e-3 * fs)
    n0, n1 = max(0, int(b["t0"] * fs) - pad), min(len(y), int(b["t1"] * fs) + pad)
    z = y[n0:n1].astype(np.complex64)
    z = z * np.exp(-2j * np.pi * b["fc"] / fs * np.arange(len(z))).astype(np.complex64)
    cut = max(b["bw"] * 0.6, 1.5e3)
    if cut < fs / 2 * 0.9:
        h = signal.firwin(min(int(6 * fs / cut) | 1, 1023), cut, fs=fs)
        z = np.convolve(z, h, mode="same").astype(np.complex64)
    r = classify(z, fs, b["bw"], b.get("snr_db"))
    if r.get("bits"):
        r["decode"] = decode_bits(r["bits"])
        r["bits"] = r["bits"][:1024]
    return r
