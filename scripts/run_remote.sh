#!/usr/bin/env bash
# Run the cpu / gpu / model benchmarks on a remote SSH host that has CUDA
# but no Docker (typical "ML container" with ssh + nvidia-smi).
#
# Requires on the remote: ssh, tar, python3 (>=3.10 recommended). No rsync,
# no nvcc, no docker needed. The gpu-config bench (pycuda) auto-skips with a
# warning if nvcc is missing — only the inference numbers really need GPU.
#
# The script:
#   1. tars relevant project files and streams them over ssh (no rsync needed)
#   2. creates a venv on the remote (cached between runs)
#   3. installs the per-bench requirements.txt
#   4. runs the selected bench scripts directly (no docker)
#   5. tars the resulting CSVs back into ./tmp/data/<host-tag>/
#
# Usage:
#   scripts/run_remote.sh user@host [options]
#
# Options:
#   --only cpu,gpu,model     subset of benches (default: cpu,gpu,model)
#   --remote-dir PATH        remote workdir          (default: ~/yolo-hw-bench)
#   --local-out PATH         local dir for CSVs      (default: ./tmp/data/<host>)
#   --tag NAME               suffix for local-out    (default: ssh host)
#   --runner-family NAME     RUNNER_FAMILY for model bench (e.g. rtdetr, yolo, timm)
#   --models LIST            MODELS override (comma-sep)
#   --img-sizes LIST         IMG_SIZES (default: spec defaults)
#   --batches LIST           BATCHES   (default: spec defaults)
#   --devices LIST           DEVICES   (default: cpu,cuda)
#   --skip-install           reuse existing venv, don't reinstall
#   --python BIN             remote python (default: python3)

set -euo pipefail

# ───── arg parsing ────────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    sed -n '2,32p' "$0"
    exit 1
fi

SSH_TARGET="$1"; shift

ONLY="cpu,gpu,model"
REMOTE_DIR='$HOME/yolo-hw-bench'
LOCAL_OUT=""
TAG=""
RUNNER_FAMILY=""
MODELS=""
IMG_SIZES=""
BATCHES=""
DEVICES=""
SKIP_INSTALL=0
REMOTE_PYTHON="python3"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)           ONLY="$2"; shift 2 ;;
        --remote-dir)     REMOTE_DIR="$2"; shift 2 ;;
        --local-out)      LOCAL_OUT="$2"; shift 2 ;;
        --tag)            TAG="$2"; shift 2 ;;
        --runner-family)  RUNNER_FAMILY="$2"; shift 2 ;;
        --models)         MODELS="$2"; shift 2 ;;
        --img-sizes)      IMG_SIZES="$2"; shift 2 ;;
        --batches)        BATCHES="$2"; shift 2 ;;
        --devices)        DEVICES="$2"; shift 2 ;;
        --skip-install)   SKIP_INSTALL=1; shift ;;
        --python)         REMOTE_PYTHON="$2"; shift 2 ;;
        -h|--help)        sed -n '2,32p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$TAG" ]]; then
    TAG="${SSH_TARGET##*@}"; TAG="${TAG//[^A-Za-z0-9_-]/_}"
fi
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[[ -z "$LOCAL_OUT" ]] && LOCAL_OUT="$PROJECT_ROOT/tmp/data/$TAG"
mkdir -p "$LOCAL_OUT"

IFS=',' read -ra SELECTED <<< "$ONLY"
declare -A WANT=()
for n in "${SELECTED[@]}"; do
    case "$n" in
        cpu|gpu|model) WANT[$n]=1 ;;
        *) echo "unknown bench: $n (want cpu|gpu|model)" >&2; exit 2 ;;
    esac
done

echo "▶ remote      : $SSH_TARGET"
echo "▶ remote dir  : $REMOTE_DIR"
echo "▶ benches     : ${SELECTED[*]}"
echo "▶ local out   : $LOCAL_OUT"

# ───── 1. push code via tar-over-ssh (no rsync needed) ────────────────────────
echo "▶ syncing code via tar…"
# REMOTE_DIR may use $HOME → resolve once on the remote.
REMOTE_DIR_RESOLVED="$(ssh "$SSH_TARGET" "echo $REMOTE_DIR")"
ssh "$SSH_TARGET" "mkdir -p '$REMOTE_DIR_RESOLVED'"

# Stream the bits we actually need: src/, scripts/, requirements.txt.
# --exclude trims __pycache__ / *.pyc so the tarball stays small.
tar -C "$PROJECT_ROOT" -cz \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    src scripts requirements.txt \
  | ssh "$SSH_TARGET" "tar -C '$REMOTE_DIR_RESOLVED' -xz"

# ───── 2. build the remote bash script ────────────────────────────────────────
# Unquoted heredoc → ${VAR} placeholders are interpolated *locally*; tokens
# that should be expanded on the remote are escaped with \$.
REMOTE_SCRIPT=$(cat <<EOF
set -euo pipefail
cd "$REMOTE_DIR_RESOLVED"

echo "── host: \$(hostname)  cuda: \$(command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || echo none)"

if [[ ! -d .venv ]]; then
    $REMOTE_PYTHON -m venv .venv
fi
. .venv/bin/activate
python -m pip install --upgrade pip wheel >/dev/null

mkdir -p data weights

SKIP_INSTALL=$SKIP_INSTALL
pip_install() {  # \$1 = requirements file
    if [[ \$SKIP_INSTALL -eq 1 ]]; then
        echo "── skip pip install (\$1)"
    else
        echo "── pip install -r \$1"
        pip install -q -r "\$1"
    fi
}

run_cpu() {
    pip_install src/check_cpu_config/requirements.txt
    echo "── running cpu bench…"
    DATA_DIR="\$PWD/data" python src/check_cpu_config/check_cpu_config.py
}

run_gpu() {
    if ! command -v nvcc >/dev/null; then
        echo "!! nvcc not found — skipping gpu-config (pycuda needs cuda-devel)." >&2
        echo "!! the model bench still runs on GPU via torch.cuda — that's the important one." >&2
        return 0
    fi
    pip_install src/check_gpu_config/requirements.txt
    echo "── running gpu bench…"
    DATA_DIR="\$PWD/data" python src/check_gpu_config/check_gpu_config.py
}

run_model() {
    pip_install src/check_model_predict/requirements.txt
    echo "── running model bench…"
    # check_model_predict.py does sys.path.insert(0, "/app") for the container
    # layout; on the remote we put src/ on PYTHONPATH so runners/ still resolves.
    DATA_DIR="\$PWD/data" \\
    IMAGES_DIR="\$PWD/src/check_model_predict/images" \\
    WEIGHTS_DIR="\$PWD/weights" \\
    YOLO_CONFIG_DIR="\$PWD/weights" \\
    HF_HOME="\$PWD/weights/hf" \\
    TORCH_HOME="\$PWD/weights/torch" \\
    MPLBACKEND=Agg \\
    ${RUNNER_FAMILY:+RUNNER_FAMILY=$RUNNER_FAMILY} \\
    ${MODELS:+MODELS=$MODELS} \\
    ${IMG_SIZES:+IMG_SIZES=$IMG_SIZES} \\
    ${BATCHES:+BATCHES=$BATCHES} \\
    ${DEVICES:+DEVICES=$DEVICES} \\
    PYTHONPATH="\$PWD/src" \\
    python src/check_model_predict/check_model_predict.py
}

${WANT[cpu]:+run_cpu}
${WANT[gpu]:+run_gpu}
${WANT[model]:+run_model}

echo "── results in \$PWD/data:"
ls -la data
EOF
)

# ───── 3. execute on remote ───────────────────────────────────────────────────
echo "▶ executing on remote…"
ssh "$SSH_TARGET" "bash -s" <<< "$REMOTE_SCRIPT"

# ───── 4. pull CSVs back via tar-over-ssh ─────────────────────────────────────
echo "▶ pulling CSVs to $LOCAL_OUT"
ssh "$SSH_TARGET" "cd '$REMOTE_DIR_RESOLVED/data' && tar -cz *.csv 2>/dev/null || true" \
    | tar -C "$LOCAL_OUT" -xz

echo "✓ done — files in $LOCAL_OUT"
ls -la "$LOCAL_OUT"
