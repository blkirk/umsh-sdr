"""Record one "other emissions" bucket to disk for later research, then analyse it.

A capture is a folder under settings.capture_dir (default ~/Downloads/umsh-sdr-captures):

  <id>.sigmf-data      raw rtl_sdr samples: interleaved unsigned 8-bit I/Q ("cu8"), 2.048 MS/s
  <id>.sigmf-meta      SigMF metadata: LO, rate, gain, ppm, and one annotation per burst found
  <id>_slice_<k>k.cs16 the bucket's slice, mixed to 0 Hz and decimated (int16 I/Q), when narrower
  capture.json         plan, status and analysis (what the dashboard's capture page shows)
  overview.png, burst.png, README.txt, analysis.log, rtl_433.txt (if rtl_433 is installed)

The LO sits beside the slice so the RTL's DC spike and IQ image stay out of it.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RATE = 2_048_000
BW_HI_HZ = [25e3, 200e3, 600e3, 1.2e6]  # upper edge used for each bandwidth bucket's passband
ID_RE = re.compile(r"^\d{8}-\d{6}_\d+\.\d{3}MHz_bw\d$")


def root(settings) -> Path:
    return Path(settings.capture_dir).expanduser()


def capture_dir(settings, cid: str) -> Path | None:
    if not ID_RE.match(cid or ""):
        return None
    d = root(settings) / cid
    return d if (d / "capture.json").exists() else None


def make_plan(settings, f_lo_hz: float, f_hi_hz: float, bw_bucket: int, seconds: float | None = None,
              gain_db: float | None = None, per_hour: float | None = None) -> dict:
    f_lo_hz, f_hi_hz, bw_bucket = float(f_lo_hz), float(f_hi_hz), int(bw_bucket)
    if not (24e6 <= f_lo_hz < f_hi_hz <= 1.766e9) or f_hi_hz - f_lo_hz > 1e6 or not 0 <= bw_bucket <= 3:
        raise ValueError("bad slice: need 24 MHz <= f_lo < f_hi <= 1766 MHz, at most 1 MHz wide, bucket 0-3")
    fc = (f_lo_hz + f_hi_hz) / 2
    half = (f_hi_hz - f_lo_hz) / 2 + BW_HI_HZ[bw_bucket] / 2 + 10e3
    offset = half + 50e3 if 2 * half + 50e3 <= 0.9e6 else 0.0
    decim = next((d for d in (8, 4, 2) if RATE / d * 0.4 >= half), 1)
    if seconds is None:
        seconds = min(60, max(10, math.ceil(15 * 3600 / per_hour))) if per_hour else 30
    seconds = float(min(300, max(2, seconds)))
    gain = settings.calibration.gain_db if gain_db is None else float(gain_db)
    return {
        "f_lo_hz": f_lo_hz, "f_hi_hz": f_hi_hz, "bw_bucket": bw_bucket, "fc_hz": fc,
        "lo_hz": int(round(fc - offset)), "offset_hz": offset, "sample_rate": RATE, "decim": decim,
        "slice_rate": RATE // decim, "half_bw_hz": half, "seconds": seconds, "gain_db": gain,
        "ppm": settings.calibration.ppm, "bytes": int(seconds * RATE * 2),
        "expected_bursts": round(per_hour * seconds / 3600, 1) if per_hour else None,
        "pause_s": seconds + 6, "dir": str(root(settings)),
    }


def _write(d: Path, rec: dict):
    tmp = d / "capture.json.tmp"
    tmp.write_text(json.dumps(rec, indent=1, default=_js))
    os.replace(tmp, d / "capture.json")


def _js(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def load(d: Path) -> dict:
    return json.loads((d / "capture.json").read_text())


def update(d: Path, **kw) -> dict:
    rec = load(d)
    rec.update(kw)
    _write(d, rec)
    return rec


def new_capture(settings, plan: dict, bucket: dict | None = None) -> tuple[Path, dict]:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    cid = f"{stamp}_{plan['fc_hz'] / 1e6:.3f}MHz_bw{plan['bw_bucket']}"
    d = root(settings) / cid
    d.mkdir(parents=True, exist_ok=False)
    rec = {"id": cid, "created": time.time(), "status": "queued", "plan": plan, "bucket": bucket,
           "files": [], "error": None}
    _write(d, rec)
    return d, rec


def list_captures(settings, limit: int = 50) -> list[dict]:
    out = []
    r = root(settings)
    if not r.is_dir():
        return out
    for d in sorted(r.iterdir(), reverse=True):
        if not ID_RE.match(d.name) or not (d / "capture.json").exists():
            continue
        try:
            rec = load(d)
        except (OSError, ValueError):
            continue
        s = (rec.get("analysis") or {}).get("summary") or {}
        out.append({"id": rec["id"], "created": rec["created"], "status": rec["status"],
                    "f_lo_hz": rec["plan"]["f_lo_hz"], "f_hi_hz": rec["plan"]["f_hi_hz"],
                    "bw_bucket": rec["plan"]["bw_bucket"], "seconds": rec["plan"]["seconds"],
                    "modulation": s.get("modulation"), "n_target": s.get("n_target")})
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------- recording (needs the radio)


def record(d: Path, rtl_sdr: str = "rtl_sdr", tries: int = 4) -> dict:
    """Run rtl_sdr for the planned length. The caller must have released the dongle."""
    rec = load(d)
    p = rec["plan"]
    data = d / f"{rec['id']}.sigmf-data"
    n = int(p["seconds"] * p["sample_rate"])
    cmd = [rtl_sdr, "-f", str(p["lo_hz"]), "-s", str(p["sample_rate"]), "-g", str(p["gain_db"]),
           "-p", str(int(round(p["ppm"]))), "-n", str(n), str(data)]
    err = ""
    for i in range(tries):
        t0 = time.time()
        rec = update(d, status="recording", recording={"started": t0, "until": t0 + p["seconds"] + 2})
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=p["seconds"] + 60)
        err = (proc.stderr or "").strip()
        size = data.stat().st_size if data.exists() else 0
        if proc.returncode == 0 and size >= 0.95 * 2 * n:
            info = {"started": t0, "ended": time.time(), "bytes": size, "samples": size // 2,
                    "command": " ".join(cmd), "lost_samples": "samples lost" in err.lower()}
            write_sigmf(d, rec, t0)
            return update(d, status="recorded", recording=info)
        if "usb_claim_interface" not in err and "Failed to open" not in err and "No supported devices" not in err:
            break
        time.sleep(1.0 + i)
    update(d, status="error", error=f"rtl_sdr failed: {err[-400:] or 'no output'}")
    raise RuntimeError(f"rtl_sdr failed: {err[-400:]}")


def write_sigmf(d: Path, rec: dict, t0: float, annotations: list | None = None):
    p = rec["plan"]
    meta = {
        "global": {"core:datatype": "cu8", "core:sample_rate": p["sample_rate"], "core:version": "1.0.0",
                   "core:hw": "RTL-SDR (RTL2832U)", "core:recorder": "umsh-sdr rtl_sdr",
                   "core:description": f"Emission bucket {p['f_lo_hz'] / 1e6:.3f}-{p['f_hi_hz'] / 1e6:.3f} MHz, "
                                       f"bandwidth bucket {p['bw_bucket']}; tuner gain {p['gain_db']} dB, "
                                       f"ppm {p['ppm']} applied by rtl_sdr"},
        "captures": [{"core:sample_start": 0, "core:frequency": p["lo_hz"],
                      "core:datetime": datetime.fromtimestamp(t0, timezone.utc).isoformat().replace("+00:00", "Z")}],
        "annotations": annotations or [],
    }
    (d / f"{rec['id']}.sigmf-meta").write_text(json.dumps(meta, indent=1))


def start_analysis(d: Path) -> subprocess.Popen:
    """Analyse in a separate, niced process so the live decoder keeps its CPU and the GIL."""
    log = open(d / "analysis.log", "ab")
    return subprocess.Popen([sys.executable, "-m", "umsh_sdr", "analyse", str(d)], stdout=log, stderr=log,
                            stdin=subprocess.DEVNULL, preexec_fn=lambda: os.nice(10), cwd=str(Path(__file__).parent.parent))


# ---------------------------------------------------------------- analysis (offline)

BUCKET_EDGES = [0, 25e3, 200e3, 600e3, float("inf")]


def analyse(d: Path) -> dict:
    from . import iq_analysis as A
    rec = update(d, status="analysing", error=None)
    try:
        p = rec["plan"]
        data = d / f"{rec['id']}.sigmf-data"
        t_start = time.time()
        y, clip = A.channelize(data, p["sample_rate"], p["offset_hz"], p["decim"], p["half_bw_hz"])
        fs = p["sample_rate"] / p["decim"]
        files = [{"name": data.name, "bytes": data.stat().st_size,
                  "what": f"raw IQ, unsigned 8-bit interleaved (cu8), {p['sample_rate'] / 1e6:.3f} MS/s, "
                          f"centre {p['lo_hz'] / 1e6:.6f} MHz"},
                 {"name": f"{rec['id']}.sigmf-meta", "what": "SigMF metadata with one annotation per burst"}]
        if p["decim"] > 1:
            sl = d / f"{rec['id']}_slice_{int(fs / 1e3)}k.cs16"
            q = np.empty(2 * len(y), np.int16)
            peak = float(np.max(np.abs(y))) or 1.0
            scale = 30000 / max(peak, 1e-6)
            q[0::2] = np.clip(np.round(y.real * scale), -32767, 32767)
            q[1::2] = np.clip(np.round(y.imag * scale), -32767, 32767)
            q.tofile(sl)
            files.append({"name": sl.name, "bytes": sl.stat().st_size,
                          "what": f"the slice mixed to 0 Hz (= {p['fc_hz'] / 1e6:.6f} MHz) and decimated: "
                                  f"signed 16-bit interleaved (cs16), {fs / 1e3:.0f} kS/s"})
        else:
            sl = None
        S, f, dt = A.spectrogram(y, fs)
        bursts, floor = A.detect(S, f, dt)
        f_abs = p["fc_hz"]
        lo_t, hi_t = p["f_lo_hz"] - f_abs, p["f_hi_hz"] - f_abs
        for b in bursts:
            b["f_mhz"] = (f_abs + b["fc"]) / 1e6
            b["bw_bucket"] = int(np.searchsorted(BUCKET_EDGES, b["bw"], side="right") - 1)
            b["target"] = bool(lo_t <= b["fc"] < hi_t)
            b["matches_bw"] = b["bw_bucket"] == p["bw_bucket"]
            h = p["half_bw_hz"]  # a wide burst straddling the analysed passband's edge is cut off by the filter
            b["edge"] = b["edge"] or (b["bw"] > 0.5 * h and (b["f_lo"] < -h < b["f_hi"] or b["f_lo"] < h < b["f_hi"]))
        order = sorted(range(len(bursts)), key=lambda i: (not bursts[i]["target"], not bursts[i]["matches_bw"],
                                                          -(bursts[i]["snr_db"] or -99)))
        for rank, i in enumerate(order[:40]):
            b = bursts[i]
            b.update(A.analyse_burst(y, fs, b))
        summary = summarize(bursts, p)
        _plots(d, y, fs, S, f, dt, floor, bursts, p)
        files += [{"name": "overview.png", "what": "spectrogram of the whole capture; target bursts boxed"}]
        if (d / "burst.png").exists():
            files.append({"name": "burst.png", "what": "the strongest target burst: spectrogram, frequency, envelope"})
        rtl433 = _rtl_433(d, sl, fs) if sl is not None else None
        if rtl433 and rtl433.get("available"):
            files.append({"name": "rtl_433.txt", "what": "rtl_433 decode attempt and pulse analysis"})
        ann = [{"core:sample_start": int(b["t0"] * p["sample_rate"]), "core:sample_count": int((b["t1"] - b["t0"]) * p["sample_rate"]),
                "core:freq_lower_edge": f_abs + b["fc"] - b["bw"] / 2, "core:freq_upper_edge": f_abs + b["fc"] + b["bw"] / 2,
                "core:label": (b.get("modulation") or "burst") + (" *" if b["target"] else "")}
               for b in bursts[:2000]]
        write_sigmf(d, rec, (rec.get("recording") or {}).get("started", rec["created"]), ann)
        _readme(d, rec, files, fs)
        files.append({"name": "README.txt", "what": "how to open these files"})
        keep = ("t0", "t1", "dur_ms", "f_mhz", "fc", "bw", "bw_bucket", "snr_db", "peak_db", "target", "matches_bw", "edge",
                "modulation", "reasons", "baud", "baud_fit", "deviation_hz", "levels_hz", "mod_index", "chirp_rate_hz_s",
                "lora_sf_estimate", "envelope_cv", "off_fraction", "decode", "bits")
        analysis = {"seconds_taken": round(time.time() - t_start, 1), "clip_fraction": clip, "fs": fs,
                    "bin_hz": float(f[1] - f[0]), "frame_s": dt, "n_bursts": len(bursts), "summary": summary,
                    "bursts": [{k: b.get(k) for k in keep} for b in bursts[:300]], "rtl_433": rtl433}
        for fl in files:
            fp = d / fl["name"]
            if fp.exists():
                fl["bytes"], fl["path"] = fp.stat().st_size, str(fp)
        return update(d, status="done", analysis=analysis, files=files)
    except Exception as e:
        traceback.print_exc()
        update(d, status="error", error=f"analysis failed: {e!r}")
        raise


def summarize(bursts: list[dict], p: dict) -> dict:
    tgt = [b for b in bursts if b["target"]]
    skip = ("too weak", "too short", "unknown", "wider than passband")
    use = [b for b in tgt if b.get("modulation") and b["modulation"] not in skip]
    mods: dict[str, int] = {}
    for b in use:
        mods[b["modulation"]] = mods.get(b["modulation"], 0) + 1
    mod = max(mods, key=mods.get) if mods else None
    same = [b for b in use if b["modulation"] == mod]

    def med(key, rows):
        v = [b[key] for b in rows if b.get(key) is not None]
        return float(np.median(v)) if v else None

    bauds = [b for b in same if b.get("baud") and (b.get("baud_fit") or 0) >= 0.75]
    sync: dict[str, int] = {}
    frames: dict[str, int] = {}
    man = 0
    for b in same:
        dec = b.get("decode") or {}
        for s in {x["label"] for x in dec.get("sync", [])}:
            sync[s] = sync.get(s, 0) + 1
        if dec.get("manchester"):
            man += 1
        if dec.get("frame_hex") and dec.get("sync"):
            k = dec["frame_hex"][:16]
            frames[k] = frames.get(k, 0) + 1
    starts = sorted(b["t0"] for b in tgt)
    freqs = sorted({round(b["f_mhz"], 3) for b in (same or tgt)})
    others = [b for b in bursts if not b["target"]]
    s = {"n_wide": sum(b.get("modulation") == "wider than passband" for b in tgt), "n_target": len(tgt), "n_classified": len(use), "n_other": len(others), "modulation": mod, "modulation_votes": mods,
         "baud": med("baud", bauds), "n_baud": len(bauds), "deviation_hz": med("deviation_hz", same),
         "bw_hz": med("bw", same or tgt), "dur_ms": med("dur_ms", same or tgt), "snr_db": med("snr_db", same or tgt),
         "interval_s": float(np.median(np.diff(starts))) if len(starts) > 2 else None,
         "distinct_freqs_mhz": freqs[:40], "n_distinct_freqs": len(freqs), "sync_hits": sync, "manchester_bursts": man,
         "repeated_frames": {k: v for k, v in frames.items() if v > 1},
         "other_freqs_mhz": sorted({round(b["f_mhz"], 2) for b in others})[:40]}
    s["hints"] = hints(s, p)
    return s


def hints(s: dict, p: dict) -> list[str]:
    h = []
    mod, bw, dur = s["modulation"], s["bw_hz"] or 0, s["dur_ms"] or 0
    syncs = " ".join(s["sync_hits"])
    if not s["n_target"]:
        h.append("Nothing was detected in this slice during the capture. These emissions are intermittent: "
                 "try a longer capture, or a busier bucket.")
        return h
    if "ERT" in syncs:
        h.append("An Itron ERT preamble/sync was found: an ERT utility meter (gas/water/electric). "
                 "rtlamr (github.com/bemasher/rtlamr) decodes SCM, SCM+ and IDM messages and meter IDs.")
    if "802.15.4g" in syncs:
        h.append("An IEEE 802.15.4g SUN-FSK frame delimiter was found: Wi-SUN or similar smart-grid mesh "
                 "(electric meters, street lights, distribution automation). Payloads are normally encrypted.")
    if "2DD4" in syncs or "D391" in syncs:
        h.append("A default sync word of a common sub-GHz FSK transceiver (Semtech SX12xx/HopeRF or TI CC1101) "
                 "was found: a generic sensor or telemetry device. rtl_433 knows many of these.")
    if "Z-Wave" in syncs:
        h.append("Z-Wave preamble + start-of-frame pattern found (US Z-Wave uses 908.4 and 916 MHz).")
    if mod and mod.startswith("chirp"):
        h.append("Chirps: another LoRa network (LoRaWAN, Meshtastic, MeshCore, private links). It shares the band "
                 "with UMSH; a different sync word or spreading factor keeps it from decoding as UMSH.")
    if mod in ("2-FSK", "4-FSK") and bw < 60e3 and dur < 60 and s["n_distinct_freqs"] > 1:
        h.append("Short narrowband FSK bursts on several frequencies: consistent with frequency-hopping utility "
                 "meters (AMI) or industrial telemetry, which hop across 902-928 MHz.")
    elif mod in ("2-FSK", "4-FSK"):
        h.append("FSK bursts: typical of sensors, utility meters, alarm panels and data radios in this band.")
    if mod == "OOK/ASK":
        h.append("On-off keying: typical of ERT meters, simple remotes/sensors, and RFID readers' downlink.")
    if mod and mod.startswith("CW"):
        h.append("Unmodulated carrier: an RFID (EPC Gen2) reader's carrier between commands, a test transmitter, "
                 "or a spur. RFID readers hop across 902-928 MHz with dwell times up to 0.4 s.")
    if s["n_wide"]:
        h.append(f"{s['n_wide']} bursts in the slice are parts of wider signals (they fill the analysed passband). "
                 "Capture this slice again with the 200-600 or >= 600 kHz bucket to see them whole.")
    if s["repeated_frames"]:
        h.append("Some bursts carry identical frame starts: a device repeating its ID or the same message.")
    if s["manchester_bursts"]:
        h.append(f"{s['manchester_bursts']} bursts decode cleanly as Manchester.")
    if not h:
        h.append("No known pattern recognised. Open the raw file in inspectrum or Universal Radio Hacker "
                 "(see README.txt) to look closer.")
    if s["n_baud"] == 0 and mod in ("2-FSK", "OOK/ASK", "BPSK"):
        h.append("The symbol rate could not be pinned down; the bursts may be too weak or too short.")
    return h


def _rtl_433(d: Path, sl: Path, fs: float) -> dict | None:
    exe = shutil.which("rtl_433")
    if not exe:
        return {"available": False}
    out = {"available": True, "decoded": [], "analysis": ""}
    try:
        p = subprocess.run([exe, "-r", f"cs16:{sl}", "-s", str(int(fs)), "-F", "json", "-M", "level"],
                           capture_output=True, text=True, timeout=180)
        for line in p.stdout.splitlines():
            try:
                out["decoded"].append(json.loads(line))
            except ValueError:
                pass
        a = subprocess.run([exe, "-r", f"cs16:{sl}", "-s", str(int(fs)), "-A"], capture_output=True, text=True, timeout=180)
        out["analysis"] = (a.stdout + a.stderr)[-6000:]
    except (OSError, subprocess.SubprocessError) as e:
        out["error"] = repr(e)
    (d / "rtl_433.txt").write_text(json.dumps(out["decoded"], indent=1) + "\n\n" + out["analysis"])
    out["decoded"] = out["decoded"][:50]
    return out


def _readme(d: Path, rec: dict, files: list, fs: float):
    p = rec["plan"]
    lines = [f"umsh-sdr emission capture {rec['id']}",
             f"Slice {p['f_lo_hz'] / 1e6:.3f}-{p['f_hi_hz'] / 1e6:.3f} MHz, bandwidth bucket {p['bw_bucket']}, "
             f"{p['seconds']:.0f} s at tuner gain {p['gain_db']} dB (ppm {p['ppm']} corrected by rtl_sdr).",
             f"The tuner was centred at {p['lo_hz'] / 1e6:.6f} MHz; the slice sits {p['offset_hz'] / 1e3:.0f} kHz above it.", "",
             "Files:"] + [f"  {f['name']}: {f['what']}" for f in files] + [
             "", "Opening them:",
             f"  inspectrum {rec['id']}.sigmf-data      (reads the SigMF metadata; or set rate {p['sample_rate']})",
             f"  Universal Radio Hacker: copy/rename the .sigmf-data to .complex16u, sample rate {p['sample_rate']}",
             f"  rtl_433 -r cu8:{rec['id']}.sigmf-data -s {p['sample_rate']} -f {p['lo_hz']}   (raw)"]
    if p["decim"] > 1:
        lines.append(f"  rtl_433 -r cs16:{rec['id']}_slice_{int(fs / 1e3)}k.cs16 -s {int(fs)} -A   (slice; -A = pulse analyser)")
    lines += [f"  numpy: x = np.fromfile(f, np.uint8).astype(np.float32) - 127.5; iq = x[0::2] + 1j*x[1::2]",
              "", "Re-run the analysis: umsh-sdr analyse <this folder>"]
    (d / "README.txt").write_text("\n".join(lines) + "\n")


def _plots(d: Path, y, fs, S, f, dt, floor, bursts, p):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from . import iq_analysis as A
    fc = p["fc_hz"]
    rows = S.shape[0]
    k = max(1, rows // 1500)
    Sd = S[:rows - rows % k].reshape(-1, k, S.shape[1]).max(1)
    db = 10 * np.log10(Sd / floor + 1e-12)
    fig, ax = plt.subplots(figsize=(11, 6), dpi=110)
    ext = [(fc + f[0]) / 1e6, (fc + f[-1]) / 1e6, Sd.shape[0] * k * dt, 0]
    im = ax.imshow(db, aspect="auto", extent=ext, cmap="magma", vmin=0, vmax=max(20, np.percentile(db, 99.9)),
                   interpolation="nearest")
    for x in (p["f_lo_hz"], p["f_hi_hz"]):
        ax.axvline(x / 1e6, color="#3987e5", lw=1, ls="--")
    for b in bursts[:2000]:
        ax.add_patch(Rectangle(((fc + b["fc"] - b["bw"] / 2) / 1e6, b["t0"]), b["bw"] / 1e6, b["t1"] - b["t0"],
                               fill=False, lw=0.8, ec="#1baf7a" if b["target"] else "#9ec5f4"))
    ax.set_xlabel("MHz")
    ax.set_ylabel("seconds")
    ax.set_title(f"{p['f_lo_hz'] / 1e6:.3f}-{p['f_hi_hz'] / 1e6:.3f} MHz slice (dashed); "
                 f"green = bursts in the slice, blue = others")
    fig.colorbar(im, ax=ax, label="dB over per-bin floor")
    fig.tight_layout()
    fig.savefig(d / "overview.png")
    plt.close(fig)

    tg = [b for b in bursts if b["target"] and b.get("modulation") not in (None, "too weak", "too short", "wider than passband")]
    tg = tg or [b for b in bursts if b["target"] and b.get("modulation")]
    if not tg:
        return
    b = max(tg, key=lambda x: x["snr_db"] or -99)
    pad = int(1e-3 * fs)
    n0, n1 = max(0, int(b["t0"] * fs) - pad), min(len(y), int(b["t1"] * fs) + pad)
    z = y[n0:n1] * np.exp(-2j * np.pi * b["fc"] / fs * np.arange(n1 - n0))
    from scipy import signal
    cut = max(b["bw"] * 0.6, 1.5e3)
    if cut < fs / 2 * 0.9:
        z = np.convolve(z, signal.firwin(min(int(6 * fs / cut) | 1, 1023), cut, fs=fs), mode="same")
    t = (n0 + np.arange(len(z))) / fs * 1e3
    fig, axs = plt.subplots(3, 1, figsize=(11, 8), dpi=110, sharex=True)
    nper = 128 if len(z) > 512 else 32
    ff, tt, Z = signal.spectrogram(y[n0:n1], fs=fs, nperseg=nper, noverlap=nper * 3 // 4, return_onesided=False)
    Z = np.fft.fftshift(Z, axes=0)
    ff = np.fft.fftshift(ff)
    axs[0].pcolormesh(n0 / fs * 1e3 + tt * 1e3, (fc + ff) / 1e6, 10 * np.log10(Z + 1e-20), shading="auto", cmap="magma")
    axs[0].set_ylim((fc + b["fc"] - 2 * b["bw"]) / 1e6, (fc + b["fc"] + 2 * b["bw"]) / 1e6)
    axs[0].set_ylabel("MHz")
    axs[0].set_title(f"Burst at {b['t0']:.3f} s, {b['f_mhz']:.4f} MHz: {b.get('modulation')}"
                     + (f", {b['baud']:.0f} Bd" if b.get("baud") else ""))
    inst = np.angle(z[1:] * np.conj(z[:-1])) * fs / (2 * np.pi)
    L = max(1, int(fs / (4 * max(b["bw"], 2e3))))
    axs[1].plot(t[1:], A._mavg(inst.astype(np.float32), L) / 1e3, lw=0.7, color="#2a78d6")
    for lv in b.get("levels_hz") or []:
        axs[1].axhline(lv / 1e3, color="#eb6834", lw=0.8, ls="--")
    axs[1].set_ylabel("inst. freq kHz")
    axs[1].set_ylim(-1.5 * b["bw"] / 1e3, 1.5 * b["bw"] / 1e3)
    axs[2].plot(t, np.abs(z), lw=0.7, color="#1baf7a")
    axs[2].set_ylabel("envelope")
    axs[2].set_xlabel("ms from capture start")
    fig.tight_layout()
    fig.savefig(d / "burst.png")
    plt.close(fig)
