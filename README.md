# umsh-sdr

Passive RTL-SDR monitor for the UMSH LoRa mesh (917.5 MHz, BW 500 kHz, SF10, CR 4/5,
sync 0x12 / SX126x 0x1424). It does two jobs:

1. **Decodes UMSH frames.** It records CRC status, timing, in-channel power, SNR and
   carrier offset (CFO), and writes LoRaTAP pcaps that the UMSH Wireshark dissector reads.
2. **Quantifies RF noise in and around the channel.** It keeps per-second noise floors and
   occupancy, logs and classifies every non-UMSH emission, runs periodic 900–935 MHz sweeps,
   and estimates how much an SX1262 receiver is desensitised.

![status](https://img.shields.io/badge/status-experimental-orange)

## Requirements

- Linux (tested on Ubuntu 24.04) with an RTL2832U dongle (R820T/R828D tuner).
- GNU Radio 3.10 with gr-osmosdr, librtlsdr and `rtl_power`, plus numpy, scipy and matplotlib,
  all from apt. `scripts/setup_host.sh` lists the packages.
- [gr-lora_sdr](https://github.com/tapparelj/gr-lora_sdr). The setup script clones a tested
  revision, applies a one-line patch and builds it into `.prefix/`.
- Optional:
  - `tshark` together with the UMSH repository's Wireshark dissector (set `UMSH_REPO`;
    the default is `~/umsh`). This decodes UMSH fields in the dashboard.
  - `umshctl` and a UMSH node on USB, for calibration.

## Quick start

```sh
git clone https://github.com/blkirk/umsh-sdr && cd umsh-sdr
scripts/setup_host.sh          # prints missing apt packages; --sudo installs them
cp config.example.toml config.toml   # then set ping_target etc.
source env.sh                  # system numpy/GNU Radio + local gr-lora_sdr + venv
python -m pytest -q            # the dissector test is skipped without tshark
umsh-sdr calibrate ppm --apply # crystal offset; needs a node that answers pings
umsh-sdr run                   # daemon: 55 min channel / 5 min sweep, dashboard on :8917
```

Open http://127.0.0.1:8917/ for:

- the live waterfall (centred on the channel, or the full 2 MHz)
- the noise floor
- the latest sweep
- decoded packets, with SRC and DST keys
- per-node tallies
- other emissions

`umsh-sdr report --hours 24` writes a self-contained HTML report to `data/reports/`.
[`docs/sample-report.html`](docs/sample-report.html) is an example from the first day.

| Command | What it does |
|---|---|
| `run [--channel-minutes N --sweep-minutes M]` | Scheduler and live dashboard |
| `channel --minutes N` | Channel mode only |
| `record out.u8 --seconds N` / `replay out.u8 --gain G` | Raw IQ capture and offline reprocessing |
| `sweep`, `overload` | One wideband survey; the gain-step intermod test |
| `calibrate ppm\|gain\|level\|selfnoise [--apply]` | Calibration (uses `umshctl ping` on the test node) |
| `report --hours H`, `dashboard` | HTML report; dashboard over the database only |

## Calibration (config.toml)

Every dongle needs its own calibration. Below are the figures from the development setup.

- **ppm**:
  - A generic R820T dongle with no TCXO measured +58 ppm, reading about 53.5 kHz low.
  - Without correction, most frames fail CRC.
  - It also drifts about 2–3 ppm while warming up.
  - Decoding tolerates that, but re-run `calibrate ppm` after warm-up if CFO figures matter.
- **gain 49.6 dB** (channel mode), chosen by `calibrate gain`:
  - It is the highest gain with no clipping between packets.
  - It decoded 100% of frames.
  - Below about 44 dB the floor is mostly the dongle's own noise.
- **dbm_offset −78.8 dB** is *node-referenced and approximate (±6 dB)*.
  - It comes from the node's RSSI for ping replies at 8.7 dB gain, carried up to 49.6 dB using the nominal tuner step.
  - It puts the floor at about thermal noise, which is impossible, so absolute dBm is not trusted for desense.
- **Desense** uses the floor's excess over the dongle's *own* floor instead, which needs no absolute calibration.
  - Run once: `umsh-sdr calibrate selfnoise --apply`, with a 50 Ω terminator on the SDR input.

## How it works

- `flowgraph.py`: the GNU Radio pipeline.
  - RTL at LO 917.0 MHz, 2 MS/s, so the DC spike falls outside the channel.
  - Spectrum tap, producing 10.24 ms frames.
  - gr-lora_sdr chain, built out block by block so CFO and timing can be tapped.
- `monitor.py`:
  - **Packet matching:**
    - Each packet is matched to its energy burst (±40 ms).
    - Weak packets below the floor get power measured over their computed time window.
    - SNR: preamble estimate for weak frames, power ratio for strong ones.
  - **Exclusions:** noise statistics leave out UMSH packets, clipped frames and a 4 s warm-up.
    Emissions that coincide with strong local packets are dropped as dongle artifacts.
- `umsh_addr.py`:
  - Reads SRC and DST keys straight from the frame bytes, with no tshark needed.
  - Shows them in UMSH's base58 format, first 4 characters.
  - Tallies packets per node.
- gr-lora_sdr is built into `.prefix` with a one-line patch
  (`third_party/gr-lora_sdr-snr-tag.patch`) that publishes its preamble SNR estimate as a tag.
- `PYTHONNOUSERSITE=1` is required. A user-site numpy 2.x breaks Ubuntu's GNU Radio
  (`ImportError: initialization failed`), and `env.sh` handles it.

## Known limits

- The SDR is an 8-bit dongle sitting next to a 30 dBm node:
  - Local packets clip, so their RSSI is a lower bound (flagged "clipped").
  - Weak distant packets still decode, down to about −17 dB SNR.
- One dongle can't sweep and decode at once, so sweeps are time-shared.
- `rtl_power` levels are relative dB.
- Emission classes are by bandwidth and duration only. Narrowband hits are consistent with FHSS meters or RFID but are not identified.

## License

GPL-3.0-or-later (see `LICENSE`). This matches GNU Radio and gr-lora_sdr, which it links against.
