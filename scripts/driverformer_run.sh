#!/usr/bin/env bash
set -euo pipefail
exec > >(tee -a stdout.txt) 2> >(tee -a stderr.txt >&2)

# --------- 입력/파라미터(환경변수) ----------
CLS="${NXF_CLS}"
FEAT="${NXF_FEAT}"
MUTS="${NXF_MUTS}"
REQS="${NXF_REQS}"
WHEELS_DIR="${NXF_WHEELS:-}"
OUT_DIR="${NXF_OUT_DIR:-results/run}"

LR="${NXF_LR:-2e-4}"
BATCH="${NXF_BATCH:-128}"
EPOCHS="${NXF_EPOCHS:-20}"
SEED="${NXF_SEED:-42}"
DMODEL="${NXF_DMODEL:-768}"
NHEAD="${NXF_NHEAD:-8}"
NLAYERS="${NXF_NLAYERS:-6}"
DFF="${NXF_DFF:-3072}"
DROPOUT="${NXF_DROPOUT:-0.2}"
MAXLEN="${NXF_MAXLEN:-1024}"
SEGLEN="${NXF_SEGLEN:-}"       # "10 50 100" or empty
OVERLAP="${NXF_OVERLAP:-0.3}"
USE_MAD="${NXF_USE_MAD:-true}"
HUBER="${NXF_HUBER:-3.0}"
CUTMIX="${NXF_CUTMIX:-0.2}"
NWORKERS="${NXF_WORKERS:-8}"
TTHREADS="${NXF_TTHREADS:-8}"
LEN_ALPHA="${NXF_LEN_ALPHA:-0.5}"
RES_BETA="${NXF_RES_BETA:-0.5}"
LABEL_ROLL="${NXF_LABEL_ROLL:-true}"
LABEL_W="${NXF_LABEL_W:-2}"

RUN_PIPE="${NXF_RUN_PIPE:-true}"
P_OUT="${NXF_P_OUT:-${OUT_DIR}/postproc_k_auto}"
P_CHUNK="${NXF_P_CHUNK:-1000000}"
P_OVERLAP="${NXF_P_OVERLAP:-100000}"
P_MINDIST="${NXF_P_MINDIST:-0}"
P_MAXDIST="${NXF_P_MAXDIST:-100000}"
P_FRAC="${NXF_P_FRAC:-0.01}"
P_GMMK="${NXF_P_GMMK:-2}"
P_BETA="${NXF_P_BETA:-1.0}"
P_GAMMA="${NXF_P_GAMMA:-0.0}"
P_SEED="${NXF_P_SEED:-$SEED}"
P_GAP="${NXF_P_GAP:-0}"

FDR_METHOD="${NXF_FDR_METHOD:-storey}"
BOOT="${NXF_BOOT:-400}"
LAMBDA_S="${NXF_LAMBDA_S:-0.20}"
LAMBDA_E="${NXF_LAMBDA_E:-0.95}"
LAMBDA_T="${NXF_LAMBDA_T:-0.01}"
PI0_F="${NXF_PI0_F:-0.01}"
PI0_C="${NXF_PI0_C:-1.0}"

echo "[INFO] PWD=$(pwd)"; ls -al | sed 's/^/  /' || true

# ---- ENV & PYTHONPATH ----
export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${TTHREADS}"
export MKL_NUM_THREADS="${TTHREADS}"
export OPENBLAS_NUM_THREADS="${TTHREADS}"
export NUMEXPR_NUM_THREADS="${TTHREADS}"
export PYTHONPATH="$PWD:$PWD/driverformer${PYTHONPATH:+:$PYTHONPATH}"

# ---- wheels offline fallback ----
if [ -d "${WHEELS_DIR:-}" ]; then
  [ "${WHEELS_DIR}" = "wheels" ] || ln -s "${WHEELS_DIR}" wheels 2>/dev/null || cp -r "${WHEELS_DIR}" wheels
  ls -al wheels | sed 's/^/  /' || true
fi

# ---- SSL-safe pip install ----
PIP_OPTS="--no-cache-dir --retries 5 --timeout 60 --index-url https://pypi.org/simple --trusted-host pypi.org --trusted-host files.pythonhosted.org"
python -m pip install -U pip wheel setuptools $PIP_OPTS || true

# requirements.txt (필터)
if [ -f "${REQS}" ] && [ "$(basename "${REQS}")" = "requirements.txt" ]; then
  grep -viE '^(torch|torchvision|torchaudio|pytorch-triton|triton|nvidia-|cuda|cudnn|cublas|cusparse|cusolver|nccl|cudatoolkit)' "${REQS}" > .req_filtered.txt || true
  [ -s .req_filtered.txt ] && python -m pip install $PIP_OPTS -r .req_filtered.txt || true
fi

# wheels 폴백
if [ -d wheels ]; then
  [ -f wheels/requirements_wheels.txt ] && python -m pip install --no-index --find-links wheels -r wheels/requirements_wheels.txt || true
  ls wheels/*.whl >/dev/null 2>&1 && python -m pip install --no-index --find-links wheels wheels/*.whl || true
fi

# 자동 보충
MISS=$(python - <<'PY'
mods=["pandas","pyarrow","scikit-learn","tqdm","pyyaml","matplotlib","statsmodels","patsy","rotary_embedding_torch","einops"]
import importlib.util
print(" ".join([m for m in mods if not importlib.util.find_spec(m)]))
PY
)
[ -n "${MISS}" ] && python -m pip install $PIP_OPTS ${MISS} || true

# ---- args 구성 ----
ARGS=(
  --cls-file "${CLS}" --feat-file "${FEAT}" --mutations-file "${MUTS}"
  --out-dir "${OUT_DIR}" --lr "${LR}" --batch-size "${BATCH}" --epochs "${EPOCHS}" --seed "${SEED}"
  --d-model "${DMODEL}" --nhead "${NHEAD}" --num-layers "${NLAYERS}" --dim-feedforward "${DFF}"
  --dropout "${DROPOUT}" --max-seq-len "${MAXLEN}" --overlap-factor "${OVERLAP}" --huber-factor "${HUBER}"
  --cutmix-p "${CUTMIX}" --num-data-workers "${NWORKERS}" --torch-threads "${TTHREADS}"
  --len-alpha "${LEN_ALPHA}" --res-beta "${RES_BETA}" --label-roll-width "${LABEL_W}"
  --pipeline-out-dir "${P_OUT}" --pipeline-chunk-size "${P_CHUNK}" --pipeline-chunk-overlap "${P_OVERLAP}"
  --pipeline-min-distance "${P_MINDIST}" --pipeline-max-distance "${P_MAXDIST}" --pipeline-sample-frac "${P_FRAC}"
  --pipeline-gmm-k "${P_GMMK}" --pipeline-beta "${P_BETA}" --pipeline-gamma "${P_GAMMA}"
  --pipeline-seed "${P_SEED}" --pipeline-dp-gap-bp "${P_GAP}"
  --postsel-fdr-method "${FDR_METHOD}" --postsel-bootstrap "${BOOT}"
  --postsel-lambda-start "${LAMBDA_S}" --postsel-lambda-end "${LAMBDA_E}" --postsel-lambda-step "${LAMBDA_T}"
  --postsel-pi0-floor "${PI0_F}" --postsel-pi0-ceil "${PI0_C}"
)
# segment_lengths: 문자열 정규화 → 숫자만 → 있으면 추가
if [ -n "${SEGLEN}" ]; then
  S=$(echo "${SEGLEN}" | tr ',' ' ' | sed -e 's/^ *//; s/ *$//' -e 's/  \+/ /g' -e 's/^"//; s/"$//')
  S=$(for t in $S; do case "$t" in (''|*[!0-9]*) ;; (*) printf '%s ' "$t";; esac; done | sed 's/ *$//')
  [ -n "${S}" ] && ARGS+=( --segment-lengths ${S} )
fi

[ "${USE_MAD}" = "true" ]   && ARGS+=( --use-mad )
[ "${LABEL_ROLL}" = "true" ]&& ARGS+=( --label-roll )
[ "${RUN_PIPE}" = "true" ]  && ARGS+=( --run-pipeline )

# ---- 실행 (모듈 우선) ----
if python - <<'PY'; then
import importlib.util as u; print(1 if u.find_spec("driverformer") else 0)
PY
then
  set -x; python -u -m driverformer "${ARGS[@]}"; set +x
elif [ -f "trainDriverFormer.py" ]; then
  set -x; python -u trainDriverFormer.py "${ARGS[@]}"; set +x
else
  echo "[ERROR] Neither 'driverformer' module nor 'trainDriverFormer.py' present."; exit 2
fi
