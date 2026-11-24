"""Self-contained HTML report (embedded PNG charts) for a time window."""
from __future__ import annotations

import base64
import html
import io
import time
from pathlib import Path

import numpy as np

from . import config as C
from .db import DB, unf32
from .sweep import KNOWN_SERVICES

INK, INK2, GRID = "#0b0b0b", "#52514e", "#ebeae6"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3 (validated order)
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BAND_STYLE = {"channel": ("UMSH channel 917.25–917.75", S1), "adj_low": ("Below 916.08–917.25", S2),
              "adj_high": ("Above 917.75–917.92", S3)}


def desense_from_excess(excess_db: float) -> float:
    """Desense of an SX1262 given how far the antenna floor sits above the SDR's own
    (terminated-input) floor. External noise = N_sdr*(10^(x/10) - 1); with the RTL's
    noise figure taken as equal to the SX1262's (both ~4-6 dB), that external noise
    relative to the SX1262 floor is the same ratio."""
    ext = max(10 ** (excess_db / 10) - 1, 0.0) * 10 ** ((RTL_NF_DB - C.SX1262_NF_DB) / 10)
    return float(10 * np.log10(1 + ext))


RTL_NF_DB = 6.0  # R820T at high gain is typically 3.5-6 dB; equal to SX1262_NF_DB makes desense ~= excess


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#b9b8b2", "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                         "lines.linewidth": 2})
    return plt


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _label_ends(ax, ends, min_gap_frac=0.06):
    """Direct labels at line ends, nudged apart vertically so they never collide."""
    if not ends:
        return
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    ends = sorted(ends, key=lambda e: e[1])
    ys = [ends[0][1]]
    for _, y, _ in ends[1:]:
        ys.append(max(y, ys[-1] + gap))
    for (x, _, text), y in zip(ends, ys):
        ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", color=INK2, fontsize=8,
                    annotation_clip=False)


def _break_gaps(t, v, max_gap):
    """Insert NaN where samples are further apart than max_gap so lines don't bridge outages."""
    t = np.asarray(t, float); v = np.asarray(v, float)
    if t.size < 2:
        return t, v
    cut = np.flatnonzero(np.diff(t) > max_gap) + 1
    return np.insert(t, cut, t[cut - 1] + 1), np.insert(v, cut, np.nan)


def build(db: DB, s: C.Settings, hours: float = 24, out: str | None = None) -> str:
    plt = _plt()
    import matplotlib.dates as mdates
    from matplotlib.colors import LinearSegmentedColormap
    t1 = time.time()
    t0 = t1 - hours * 3600
    cal = s.calibration
    sec_all = db.query("SELECT * FROM band_seconds WHERE ts BETWEEN ? AND ? ORDER BY ts", (t0, t1))
    # dBFS only compares at one tuner gain: keep the current channel gain, report what was dropped.
    sec = [r for r in sec_all if r["gain_db"] is not None and abs(r["gain_db"] - cal.gain_db) < 0.05]
    pk = db.query("SELECT * FROM packets WHERE ts BETWEEN ? AND ? ORDER BY ts", (t0, t1))
    em = db.query("SELECT * FROM emissions WHERE ts BETWEEN ? AND ? ORDER BY ts", (t0, t1))
    sp = db.query("SELECT * FROM spectra WHERE ts BETWEEN ? AND ? AND abs(gain_db - ?) < 0.05 ORDER BY ts",
                  (t0, t1, cal.gain_db))
    sw = db.query("SELECT * FROM sweeps WHERE ts BETWEEN ? AND ? AND kind='survey' ORDER BY ts", (t0, t1))
    charts, notes = [], []
    dropped = len(sec_all) - len(sec)
    if dropped:
        other = sorted({r["gain_db"] for r in sec_all if r not in sec and r["gain_db"] is not None})
        notes.append(f"{dropped // len(C.BANDS)} seconds recorded at other gains ({', '.join(f'{g:g}' for g in other)} dB, "
                     f"e.g. calibration runs) are left out of the noise figures; current gain is {cal.gain_db} dB.")

    # ---- headline numbers ----
    by_band = {b: np.array([r["floor_dbfs"] for r in sec if r["band"] == b and r["floor_dbfs"] is not None])
               for b in C.BANDS}
    ch = by_band["channel"]
    summary = []
    if ch.size:
        med = float(np.median(ch))
        summary.append(("Channel floor, median", f"{med:.1f} dBFS/500k" + (
            f"  ({med + cal.dbm_offset:.1f} dBm)" if cal.dbm_offset is not None else "")))
        summary.append(("Channel floor, p90 of seconds", f"{np.percentile(ch, 90):.1f} dBFS/500k"))
        self_n = cal.self_floor_dbfs.get("channel")
        if self_n is not None:
            excess = med - self_n
            des = desense_from_excess(excess)
            summary.append(("Channel noise above SDR self-noise", f"{excess:+.1f} dB (terminated-input floor {self_n:.1f} dBFS/500k)"))
            summary.append(("Estimated SX1262 desense (median)", f"{des:.1f} dB"))
            summary.append(("Estimated desense, worst 10% of seconds", f"{desense_from_excess(np.percentile(ch, 90) - self_n):.1f} dB"))
            summary.append(("Noise-limited SF10 sensitivity", f"{C.SX1262_FLOOR_DBM + des + C.SF10_DEMOD_SNR_DB:.1f} dBm"
                            f" (vs {C.SX1262_FLOOR_DBM + C.SF10_DEMOD_SNR_DB:.1f} dBm in quiet RF)"))
        else:
            notes.append("Desense not computed: measure the SDR's own floor once with a 50-ohm terminator "
                         "(`umsh-sdr calibrate selfnoise --apply`).")
        if cal.dbm_offset is not None:
            summary.append(("Channel floor, node-referenced", f"{med + cal.dbm_offset:.1f} dBm/500k (approximate)"))
    busy = [r["busy_other"] for r in sec if r["band"] == "channel" and r["busy_other"] is not None]
    busy_u = [r["busy_umsh"] for r in sec if r["band"] == "channel" and r["busy_umsh"] is not None]
    if busy:
        summary.append(("Channel occupied by non-UMSH energy", f"{100 * np.mean(busy):.2f} % of time"))
        summary.append(("Channel occupied by UMSH packets", f"{100 * np.mean(busy_u):.2f} % of time"))
    summary.append(("Packets decoded (CRC ok)", f"{sum(r['crc_ok'] for r in pk)} of {len(pk)}"))
    summary.append(("Non-UMSH emissions logged", f"{len(em)}"))
    clip = [r["clip"] for r in sec if r["band"] == "channel" and r["clip"] is not None]
    if clip and np.mean(np.array(clip) > 1e-3) > 0.01:
        notes.append(f"ADC clipping in {100 * np.mean(np.array(clip) > 1e-3):.1f}% of seconds (usually a nearby "
                     "transmitter); clipped frames are excluded from noise statistics.")

    # ---- floor over time (per-minute median of per-second floors) ----
    if sec:
        fig, ax = plt.subplots(figsize=(9, 3.2))
        ends = []
        for b, (name, col) in BAND_STYLE.items():
            rows = [(r["ts"], r["floor_dbfs"]) for r in sec if r["band"] == b and r["floor_dbfs"] is not None]
            if not rows:
                continue
            t = np.array([r[0] for r in rows]); v = np.array([r[1] for r in rows])
            mins = (t // 60).astype(int)
            um, idx = np.unique(mins, return_inverse=True)
            mv = np.array([np.median(v[idx == i]) for i in range(len(um))])
            tm, mv = _break_gaps(um * 60.0, mv, 180)
            from datetime import datetime
            x = [datetime.fromtimestamp(t) if np.isfinite(v) else datetime.fromtimestamp(t) for t, v in zip(tm, mv)]
            few = np.isfinite(mv).sum() < 15
            ax.plot(x, mv, color=col, label=name, marker="o" if few else None, ms=4)
            ends.append((x[-1], mv[-1], name.split(" ")[0]))
        ax.set_ylabel("dBFS per 500 kHz")
        _label_ends(ax, ends)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8)
        charts.append(("Noise floor over time", "Per-minute median of per-second p10 floors; UMSH packets and clipped frames excluded.", _png(fig)))

        # ---- occupancy heatmap by hour ----
        hrs = np.arange(24)
        mat = np.full((3, 24), np.nan)
        for i, b in enumerate(C.BANDS):
            for h in hrs:
                vals = [r["busy_other"] for r in sec if r["band"] == b and r["busy_other"] is not None
                        and time.localtime(r["ts"]).tm_hour == h]
                if vals:
                    mat[i, h] = 100 * np.mean(vals)
        fig, ax = plt.subplots(figsize=(9, 1.9))
        cmap = LinearSegmentedColormap.from_list("seq", SEQ)
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=max(1.0, np.nanmax(mat) if np.isfinite(mat).any() else 1))
        ax.set_yticks(range(3), [BAND_STYLE[b][0].split(" 9")[0] for b in C.BANDS])
        ax.set_xticks(hrs[::2], [f"{h:02d}" for h in hrs[::2]])
        ax.set_xlabel("hour of day (local)")
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="% time busy", pad=0.01)
        charts.append(("Non-UMSH occupancy by hour", "Share of 10 ms frames more than 6 dB over the rolling floor, UMSH packets excluded. Blank: no data.", _png(fig)))

    # ---- channel-mode spectrum ----
    if sp:
        mean = np.stack([unf32(r["mean"]) for r in sp]); mx = np.stack([unf32(r["maxhold"]) for r in sp])
        p10 = np.stack([unf32(r["p10"]) for r in sp])
        f = (sp[0]["f0_hz"] + np.arange(mean.shape[1]) * sp[0]["df_hz"]) / 1e6
        keep = np.abs(f - C.LO_HZ / 1e6) < 0.92
        fig, ax = plt.subplots(figsize=(9, 3.2))
        lo, hi = C.BANDS["channel"]
        ax.axvspan(lo / 1e6, hi / 1e6, color=S1, alpha=0.08, lw=0)
        ax.text(lo / 1e6 + 0.01, 0.97, "UMSH", transform=ax.get_xaxis_transform(), va="top", color=INK2, fontsize=8)
        for y, name, col in ((mx.max(0), "Max hold", S2), (np.median(mean, 0), "Median of minute means", S1),
                             (np.median(p10, 0), "Quiet floor (p10)", S3)):
            ax.plot(f[keep], y[keep], color=col, label=name, lw=1.4)
        ax.set_xlabel("MHz"); ax.set_ylabel("dBFS per bin (1.95 kHz)")
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8)
        charts.append(("Spectrum around the channel", "2 MHz channel-mode window; DC spike at 917.000 MHz is the SDR's own.", _png(fig)))

    # ---- wideband sweep ----
    if sw:
        p50 = np.stack([unf32(r["p50"]) for r in sw if r["n"] == sw[0]["n"]])
        mxh = np.stack([unf32(r["maxhold"]) for r in sw if r["n"] == sw[0]["n"]])
        f = (sw[0]["f0_hz"] + np.arange(sw[0]["n"]) * sw[0]["df_hz"]) / 1e6
        fig, ax = plt.subplots(figsize=(9, 3.4))
        for a, b, label in KNOWN_SERVICES:
            if b < f[0] or a > f[-1]:
                continue
            ax.axvspan(max(a, f[0]), min(b, f[-1]), color=S1, alpha=0.06 if b - a > 5 else 0.18, lw=0)
            ax.text(max(a, f[0]) + 0.1, 0.98 - 0.07 * (KNOWN_SERVICES.index((a, b, label)) % 3), label,
                    transform=ax.get_xaxis_transform(), va="top", color=INK2, fontsize=7)
        ax.plot(f, mxh.max(0), color=S2, label="Max hold", lw=1.2)
        ax.plot(f, np.median(p50, 0), color=S1, label="Median", lw=1.2)
        ax.set_xlabel("MHz"); ax.set_ylabel("rtl_power dB (relative)")
        ax.legend(frameon=False, loc="lower right", fontsize=8)
        charts.append((f"Wideband survey ({len(sw)} sweeps)", "Same gain every sweep, so sweeps compare with each other, but the levels are not dBm.", _png(fig)))

    # ---- packets ----
    if pk:
        fig, axs = plt.subplots(1, 2, figsize=(9, 2.8))
        snr = [r["snr_db"] for r in pk if r["snr_db"] is not None and r["crc_ok"]]
        if snr:
            axs[0].hist(snr, bins=np.arange(-20, 50, 2), color=S1, rwidth=0.85)
            axs[0].axvline(C.SF10_DEMOD_SNR_DB, color=INK2, lw=1, ls="--")
            axs[0].text(C.SF10_DEMOD_SNR_DB + 0.5, 0.95, "SF10 limit", transform=axs[0].get_xaxis_transform(), va="top", fontsize=7, color=INK2)
        axs[0].set_xlabel("SNR dB (CRC-valid packets)"); axs[0].set_ylabel("packets")
        hours_ = np.array([time.localtime(r["ts"]).tm_hour for r in pk])
        axs[1].bar(range(24), [int((hours_ == h).sum()) for h in range(24)], color=S1, width=0.8)
        axs[1].set_xlabel("hour of day"); axs[1].set_ylabel("packets")
        charts.append(("Decoded UMSH packets", "SNR is the preamble estimate for weak frames and the power ratio for strong ones; strong frames from a nearby node may be clipped.", _png(fig)))

    # ---- emission table ----
    em_rows = ""
    if em:
        cls = {}
        for r in em:
            k = (r["cls"], "in channel" if r["in_channel"] else "adjacent")
            d = cls.setdefault(k, {"n": 0, "above": [], "freqs": {}})
            d["n"] += 1; d["above"].append(r["above_floor_db"])
            fk = round(r["f_center_hz"] / 1e6, 2); d["freqs"][fk] = d["freqs"].get(fk, 0) + 1
        for (c, where), d in sorted(cls.items(), key=lambda kv: -kv[1]["n"]):
            top = ", ".join(f"{f:.2f}" for f, _ in sorted(d["freqs"].items(), key=lambda kv: -kv[1])[:5])
            em_rows += (f"<tr><td>{c}</td><td>{where}</td><td class=num>{d['n']}</td><td class=num>"
                        f"{d['n'] / hours:.1f}</td><td class=num>{np.median(d['above']):.1f}</td><td>{top}</td></tr>")

    # ---- assemble ----
    rng = f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(t0))} – {time.strftime('%Y-%m-%d %H:%M', time.localtime(t1))}"
    body = "".join(f"<section><h2>{html.escape(t)}</h2><p class=sub>{html.escape(sub)}</p><img alt='{html.escape(t)}' src='{img}'></section>"
                   for t, sub, img in charts)
    summ = "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>" for k, v in summary)
    note_html = "".join(f"<li>{html.escape(n)}</li>" for n in notes)
    page = f"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width, initial-scale=1">
<title>UMSH RF Report</title><style>
:root {{ color-scheme: light; }} body {{ margin:0; background:#f4f4f2; color:{INK}; font:14px/1.5 system-ui,sans-serif; }}
main {{ max-width: 980px; margin: 0 auto; padding: 24px 16px 48px; }} h1 {{ font-size: 20px; margin: 0 0 4px; }}
h2 {{ font-size: 15px; margin: 0 0 2px; }} .sub {{ color:{INK2}; font-size: 12px; margin: 0 0 8px; }}
section {{ background:#fcfcfb; border:1px solid #e2e1dc; border-radius:10px; padding:14px 16px; margin: 14px 0; }}
img {{ max-width: 100%; height: auto; }} table {{ border-collapse: collapse; font-size: 13px; }}
th, td {{ text-align:left; padding: 4px 12px 4px 0; border-bottom:1px solid {GRID}; }} th {{ color:{INK2}; font-weight:500; }}
td.num {{ text-align:right; font-variant-numeric: tabular-nums; }}
</style></head><body><main>
<h1>UMSH RF environment report</h1><p class=sub>{rng} · 917.5 MHz, SF10, 500 kHz · RTL-SDR ppm {cal.ppm}, gain {cal.gain_db} dB</p>
<section><h2>Summary</h2><table>{summ}</table>{f"<ul class=sub>{note_html}</ul>" if note_html else ""}
<p class=sub>Desense is how far external noise raises an SX1262's own floor (thermal {C.THERMAL_DBM:.0f} dBm + {C.SX1262_NF_DB:.0f} dB NF = {C.SX1262_FLOOR_DBM:.0f} dBm in 500 kHz). It comes from the antenna floor's excess over the SDR's terminated-input floor, assuming the SDR and SX1262 have similar noise figures. Every dB of desense is a dB of link margin lost. Node-referenced dBm figures come from the node's RSSI through a different antenna, so treat them as approximate (±6 dB).</p></section>
{body}
{f"<section><h2>Non-UMSH emissions by class</h2><table><tr><th>Class</th><th>Where</th><th>Count</th><th>Per hour</th><th>Median dB over floor</th><th>Most common MHz</th></tr>{em_rows}</table></section>" if em_rows else ""}
</main></body></html>"""
    path = Path(out) if out else Path(s.data_dir) / "reports" / f"report-{time.strftime('%Y%m%d-%H%M', time.localtime(t1))}.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page)
    return str(path)
