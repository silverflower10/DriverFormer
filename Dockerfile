FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive

# 필수 최소 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
   && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/driverformer

# ⚠️ 리포 내용(코드 + wheels/)을 그대로 이미지에 복사
#    → git clone 대신 COPY . 를 쓰는 이유가 바로 wheels를 넣기 위함
COPY . /opt/driverformer/repo

# 파이썬 기본 도구
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 1) 로컬 wheels만으로 설치(가능하면 PyPI 접속 없이)
RUN if [ -d repo/wheels ] && ls repo/wheels/*.whl >/dev/null 2>&1; then \
      echo "[INFO] Installing local wheels..."; \
      pip install --no-cache-dir --no-index --find-links=repo/wheels repo/wheels/*.whl ; \
    else \
      echo "[WARN] No wheels found under repo/wheels"; \
    fi

# 2) (선택) 남은 의존성 보충 – 필요할 때만 주석 해제
# RUN if [ -f repo/requirements.txt ]; then \
#       pip install --no-cache-dir -r repo/requirements.txt ; \
#     fi

# 드라이버포머 모듈 import 경로
ENV PYTHONPATH=/opt/driverformer/repo:${PYTHONPATH}

# 실행 스크립트(컨테이너 내부 경로 고정)
COPY <<'BASH' /opt/driverformer/run.sh
#!/usr/bin/env bash
set -euo pipefail

CLS_FILE="${CLS_FILE:?CLS_FILE is required}"
FEAT_FILE="${FEAT_FILE:?FEAT_FILE is required}"
MUTATIONS_FILE="${MUTATIONS_FILE:-}"
OUT_DIR="${OUT_DIR:-results/run}"

LR="${LR:-2e-4}"; BATCH_SIZE="${BATCH_SIZE:-128}"; EPOCHS="${EPOCHS:-20}"; SEED="${SEED:-42}"
D_MODEL="${D_MODEL:-768}"; NHEAD="${NHEAD:-8}"; NUM_LAYERS="${NUM_LAYERS:-6}"; DIM_FEEDFORWARD="${DIM_FEEDFORWARD:-3072}"
DROPOUT="${DROPOUT:-0.2}"; MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"; SEGMENT_LENGTHS="${SEGMENT_LENGTHS:-}"; OVERLAP_FACTOR="${OVERLAP_FACTOR:-0.3}"
USE_MAD="${USE_MAD:-false}"; HUBER_FACTOR="${HUBER_FACTOR:-3.0}"; CUTMIX_P="${CUTMIX_P:-0.2}"
NUM_DATA_WORKERS="${NUM_DATA_WORKERS:-8}"; TORCH_THREADS="${TORCH_THREADS:-8}"; LEN_ALPHA="${LEN_ALPHA:-0.5}"; RES_BETA="${RES_BETA:-0.5}"
LABEL_ROLL="${LABEL_ROLL:-false}"; LABEL_ROLL_WIDTH="${LABEL_ROLL_WIDTH:-2}"
RUN_PIPELINE="${RUN_PIPELINE:-false}"; PIPELINE_OUT_DIR="${PIPELINE_OUT_DIR:-}"; PIPELINE_CHUNK_SIZE="${PIPELINE_CHUNK_SIZE:-}"
PIPELINE_CHUNK_OVERLAP="${PIPELINE_CHUNK_OVERLAP:-}"; PIPELINE_MIN_DISTANCE="${PIPELINE_MIN_DISTANCE:-}"; PIPELINE_MAX_DISTANCE="${PIPELINE_MAX_DISTANCE:-}"
PIPELINE_SAMPLE_FRAC="${PIPELINE_SAMPLE_FRAC:-}"; PIPELINE_GMM_K="${PIPELINE_GMM_K:-}"; PIPELINE_BETA="${PIPELINE_BETA:-}"; PIPELINE_GAMMA="${PIPELINE_GAMMA:-}"
PIPELINE_SEED="${PIPELINE_SEED:-}"; PIPELINE_DP_GAP_BP="${PIPELINE_DP_GAP_BP:-}"; PIPELINE_PRESMOOTH_BINS="${PIPELINE_PRESMOOTH_BINS:-}"
POSTSEL_FDR_METHOD="${POSTSEL_FDR_METHOD:-}"; POSTSEL_BOOTSTRAP="${POSTSEL_BOOTSTRAP:-}"; POSTSEL_LAMBDA_START="${POSTSEL_LAMBDA_START:-}"
POSTSEL_LAMBDA_END="${POSTSEL_LAMBDA_END:-}"; POSTSEL_LAMBDA_STEP="${POSTSEL_LAMBDA_STEP:-}"; POSTSEL_PI0_FLOOR="${POSTSEL_PI0_FLOOR:-}"; POSTSEL_PI0_CEIL="${POSTSEL_PI0_CEIL:-}"

export OMP_NUM_THREADS="${TORCH_THREADS}" MKL_NUM_THREADS="${TORCH_THREADS}" OPENBLAS_NUM_THREADS="${TORCH_THREADS}" NUMEXPR_NUM_THREADS="${TORCH_THREADS}"
export MPLBACKEND=Agg TOKENIZERS_PARALLELISM=false

cmd=( python -u -m driverformer
  --cls-file "${CLS_FILE}" --feat-file "${FEAT_FILE}" --out-dir "${OUT_DIR}"
  --lr "${LR}" --batch-size "${BATCH_SIZE}" --epochs "${EPOCHS}" --seed "${SEED}"
  --d-model "${D_MODEL}" --nhead "${NHEAD}" --num-layers "${NUM_LAYERS}" --dim-feedforward "${DIM_FEEDFORWARD}"
  --dropout "${DROPOUT}" --max-seq-len "${MAX_SEQ_LEN}" --overlap-factor "${OVERLAP_FACTOR}"
  --huber-factor "${HUBER_FACTOR}" --cutmix-p "${CUTMIX_P}" --num-data-workers "${NUM_DATA_WORKERS}"
  --torch-threads "${TORCH_THREADS}" --len-alpha "${LEN_ALPHA}" --res-beta "${RES_BETA}" )

[ -n "${MUTATIONS_FILE}" ] && cmd+=( --mutations-file "${MUTATIONS_FILE}" )
[ "${USE_MAD}" = "true" ] && cmd+=( --use-mad )
[ "${LABEL_ROLL}" = "true" ] && cmd+=( --label-roll )
[ -n "${LABEL_ROLL_WIDTH}" ] && cmd+=( --label-roll-width "${LABEL_ROLL_WIDTH}" )
[ -n "${SEGMENT_LENGTHS}" ] && cmd+=( --segment-lengths ${SEGMENT_LENGTHS} )

if [ "${RUN_PIPELINE}" = "true" ]; then
  cmd+=( --run-pipeline )
  [ -n "${PIPELINE_OUT_DIR}" ]        && cmd+=( --pipeline-out-dir "${PIPELINE_OUT_DIR}" )
  [ -n "${PIPELINE_CHUNK_SIZE}" ]     && cmd+=( --pipeline-chunk-size "${PIPELINE_CHUNK_SIZE}" )
  [ -n "${PIPELINE_CHUNK_OVERLAP}" ]  && cmd+=( --pipeline-chunk-overlap "${PIPELINE_CHUNK_OVERLAP}" )
  [ -n "${PIPELINE_MIN_DISTANCE}" ]   && cmd+=( --pipeline-min-distance "${PIPELINE_MIN_DISTANCE}" )
  [ -n "${PIPELINE_MAX_DISTANCE}" ]   && cmd+=( --pipeline-max-distance "${PIPELINE_MAX_DISTANCE}" )
  [ -n "${PIPELINE_SAMPLE_FRAC}" ]    && cmd+=( --pipeline-sample-frac "${PIPELINE_SAMPLE_FRAC}" )
  [ -n "${PIPELINE_GMM_K}" ]          && cmd+=( --pipeline-gmm-k "${PIPELINE_GMM_K}" )
  [ -n "${PIPELINE_BETA}" ]           && cmd+=( --pipeline-beta "${PIPELINE_BETA}" )
  [ -n "${PIPELINE_GAMMA}" ]          && cmd+=( --pipeline-gamma "${PIPELINE_GAMMA}" )
  [ -n "${PIPELINE_SEED}" ]           && cmd+=( --pipeline-seed "${PIPELINE_SEED}" )
  [ -n "${PIPELINE_DP_GAP_BP}" ]      && cmd+=( --pipeline-dp-gap-bp "${PIPELINE_DP_GAP_BP}" )
  [ -n "${PIPELINE_PRESMOOTH_BINS}" ] && cmd+=( --pipeline-presmooth-bins "${PIPELINE_PRESMOOTH_BINS}" )
fi

[ -n "${POSTSEL_FDR_METHOD}" ]  && cmd+=( --postsel-fdr-method "${POSTSEL_FDR_METHOD}" )
[ -n "${POSTSEL_BOOTSTRAP}" ]   && cmd+=( --postsel-bootstrap "${POSTSEL_BOOTSTRAP}" )
[ -n "${POSTSEL_LAMBDA_START}" ]&& cmd+=( --postsel-lambda-start "${POSTSEL_LAMBDA_START}" )
[ -n "${POSTSEL_LAMBDA_END}" ]  && cmd+=( --postsel-lambda-end "${POSTSEL_LAMBDA_END}" )
[ -n "${POSTSEL_LAMBDA_STEP}" ] && cmd+=( --postsel-lambda-step "${POSTSEL_LAMBDA_STEP}" )
[ -n "${POSTSEL_PI0_FLOOR}" ]   && cmd+=( --postsel-pi0-floor "${POSTSEL_PI0_FLOOR}" )
[ -n "${POSTSEL_PI0_CEIL}" ]    && cmd+=( --postsel-pi0-ceil "${POSTSEL_PI0_CEIL}" )

echo "[INFO] Running: ${cmd[*]}"; exec "${cmd[@]}"
BASH

RUN chmod +x /opt/driverformer/run.sh

ENTRYPOINT ["/bin/bash", "-lc"]
CMD ["bash /opt/driverformer/run.sh"]
