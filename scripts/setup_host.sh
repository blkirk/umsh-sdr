#!/usr/bin/env bash
# One-time host setup for umsh-sdr. Re-runnable. Steps needing root are printed, not run,
# unless --sudo is given.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUDO=0; [[ "${1:-}" == "--sudo" ]] && SUDO=1

echo "== system packages"
need=()
command -v tshark >/dev/null || need+=(tshark)
dpkg -s gnuradio-dev >/dev/null 2>&1 || need+=(gnuradio-dev)
dpkg -s gr-osmosdr >/dev/null 2>&1 || need+=(gr-osmosdr)
dpkg -s rtl-sdr >/dev/null 2>&1 || need+=(rtl-sdr)
for p in cmake pybind11-dev libvolk-dev libspdlog-dev libgmp-dev python3-matplotlib; do dpkg -s "$p" >/dev/null 2>&1 || need+=("$p"); done
if ((${#need[@]})); then
  if ((SUDO)); then sudo apt-get install -y "${need[@]}"; else echo "  missing: ${need[*]}  ->  sudo apt-get install ${need[*]}"; fi
else echo "  ok"; fi

echo "== gr-lora_sdr (project-local build in .prefix, with the SNR-tag patch)"
export PYTHONNOUSERSITE=1  # a user-site numpy 2.x breaks Ubuntu's GNU Radio ABI
SRC="$ROOT/third_party/gr-lora_sdr"
GR_LORA_REV=862746d  # tested revision; the patch applies to it
if [[ ! -d "$SRC" ]]; then git clone https://github.com/tapparelj/gr-lora_sdr.git "$SRC" && git -C "$SRC" checkout -q "$GR_LORA_REV"; fi
if ! grep -q "umsh-sdr patch" "$SRC/lib/frame_sync_impl.cc"; then (cd "$SRC" && git apply "$ROOT/third_party/gr-lora_sdr-snr-tag.patch"); fi
cmake -S "$SRC" -B "$SRC/build-prefix" -DCMAKE_INSTALL_PREFIX="$ROOT/.prefix" -DCMAKE_BUILD_TYPE=Release >/dev/null
make -C "$SRC/build-prefix" -j"$(nproc)" >/dev/null && make -C "$SRC/build-prefix" install >/dev/null && echo "  ok"

echo "== python venv (system site-packages for numpy/GNU Radio)"
[[ -d "$ROOT/.venv" ]] || uv venv --system-site-packages --python /usr/bin/python3 "$ROOT/.venv"
VIRTUAL_ENV="$ROOT/.venv" uv pip install --python "$ROOT/.venv/bin/python" -q "fastapi>=0.115" "uvicorn[standard]>=0.32" "pytest>=8"
echo "  ok"

echo "== UMSH Wireshark dissector"
UMSH_REPO="${UMSH_REPO:-$HOME/umsh}"
if [[ -d "$UMSH_REPO" ]]; then (cd "$UMSH_REPO" && make install-dissector >/dev/null 2>&1 && echo "  installed") || echo "  run 'make install-dissector' in $UMSH_REPO"; else echo "  UMSH repo not found (set UMSH_REPO); skipping dissector"; fi

echo "== RTL-SDR"
# librtlsdr detaches the DVB driver on open; blacklisting is optional but avoids surprises.
if lsmod | grep -q dvb_usb_rtl28xxu && [[ ! -f /etc/modprobe.d/blacklist-rtl-sdr.conf ]]; then
  echo "  optional: echo 'blacklist dvb_usb_rtl28xxu' | sudo tee /etc/modprobe.d/blacklist-rtl-sdr.conf"
fi
timeout 5 rtl_test -t 2>&1 | grep -E "Found|tuner" || echo "  rtl_test found no device"
echo "done. source env.sh, then: python -m pytest -q && umsh-sdr --help"
