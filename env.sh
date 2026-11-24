# Source this before running umsh-sdr tools.
# Uses the system numpy/GNU Radio (the user-site numpy 2.x is ABI-incompatible
# with Ubuntu's GNU Radio), the project-local gr-lora_sdr build in .prefix,
# and the project venv (FastAPI, pytest) layered on system site-packages.
_UMSH_SDR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$_UMSH_SDR_ROOT/.prefix/lib/python3.12/site-packages:$_UMSH_SDR_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$_UMSH_SDR_ROOT/.prefix/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$_UMSH_SDR_ROOT/.venv/bin:$PATH"
umsh-sdr() { python -m umsh_sdr "$@"; }
