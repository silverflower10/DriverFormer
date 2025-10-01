// main.nf — stage repo files via channels (DSL2), SSL-safe auto-install deps, run from workdir
nextflow.enable.dsl = 2

// ---- Param defaults (no params{} block) ----
params.cls_file        = params.cls_file        ?: null
params.feat_file       = params.feat_file       ?: null
params.mutations_file  = params.mutations_file  ?: null
params.out_dir         = params.out_dir         ?: 'results/run'

// 설치 스위치(온라인 불가 환경/사전 빌드 컨테이너에서는 false로)
params.setup_deps      = (params.setup_deps in [false,'false',0,'0']) ? false : true

// training/pipeline defaults
params.lr              = params.lr              ?: 2e-4
params.batch_size      = params.batch_size      ?: 16
params.epochs          = params.epochs          ?: 2
params.seed            = params.seed            ?: 42
params.d_model         = params.d_model         ?: 768
params.nhead           = params.nhead           ?: 8
params.num_layers      = params.num_layers      ?: 6
params.dim_feedforward = params.dim_feedforward ?: 3072
params.dropout         = params.dropout         ?: 0.2
params.max_seq_len     = params.max_seq_len     ?: 1024
params.segment_lengths = params.segment_lengths ?: [10,50,100]
params.overlap_factor  = params.overlap_factor  ?: 0.3
params.use_mad         = params.use_mad         ?: true
params.huber_factor    = params.huber_factor    ?: 3.0
params.cutmix_p        = params.cutmix_p        ?: 0.2
params.num_data_workers= params.num_data_workers?: 8
params.torch_threads   = params.torch_threads   ?: 8
params.len_alpha       = params.len_alpha       ?: 0.5
params.res_beta        = params.res_beta        ?: 0.5
params.label_roll      = params.label_roll      ?: true
params.label_roll_width= params.label_roll_width?: 2

params.run_pipeline          = params.run_pipeline          ?: true
params.pipeline_out_dir      = params.pipeline_out_dir      ?: "${params.out_dir}/postproc_k_auto"
params.pipeline_chunk_size   = params.pipeline_chunk_size   ?: 1000000
params.pipeline_chunk_overlap= params.pipeline_chunk_overlap?: 100000
params.pipeline_min_distance = params.pipeline_min_distance ?: 0
params.pipeline_max_distance = params.pipeline_max_distance ?: 100000
params.pipeline_sample_frac  = params.pipeline_sample_frac  ?: 0.01
params.pipeline_gmm_k        = params.pipeline_gmm_k        ?: 2
params.pipeline_beta         = params.pipeline_beta         ?: 1.0
params.pipeline_gamma        = params.pipeline_gamma        ?: 0.0
params.pipeline_seed         = params.pipeline_seed         ?: 42
params.pipeline_dp_gap_bp    = params.pipeline_dp_gap_bp    ?: 0
params.postsel_fdr_method    = params.postsel_fdr_method    ?: 'storey'
params.postsel_bootstrap     = params.postsel_bootstrap     ?: 400
params.postsel_lambda_start  = params.postsel_lambda_start  ?: 0.20
params.postsel_lambda_end    = params.postsel_lambda_end    ?: 0.95
params.postsel_lambda_step   = params.postsel_lambda_step   ?: 0.01
params.postsel_pi0_floor     = params.postsel_pi0_floor     ?: 0.01
params.postsel_pi0_ceil      = params.postsel_pi0_ceil      ?: 1.0

// ---- Process ----
process DRIVERFORMER_RUN {
  tag "driverformer"
  publishDir "${params.out_dir}", mode: 'copy', overwrite: true
  label 'gpu'        // ← config의 withLabel: gpu { accelerator 1 } 적용

  // 리스트/문자열 → "10 50 100" 으로 정규화해서 env로 주입(문자열 대입)
  def SEGLEN_VAL = (
    (params.segment_lengths instanceof List
      ? params.segment_lengths.join(' ')
      : (params.segment_lengths ?: '')
    ).toString().trim().replaceAll(',', ' ').replaceAll(/\s+/, ' ')
  )
  env 'SEGLEN', SEGLEN_VAL

  input:
    path CLS
    path FEAT
    path MUTS
    path DF_PKG,    stageAs: 'driverformer'
    path TRAIN_PY,  stageAs: 'trainDriverFormer.py'
    path REQS
    path WHEELS_DIR

  output:
    path "stdout.txt"
    path "stderr.txt"

  shell:
  '''
  set -euo pipefail
  SEGLEN="${SEGLEN:-}"
  exec > >(tee stdout.txt) 2> >(tee stderr.txt >&2)

  # ==== workspace/TMP 설정 (/dev/shm 압박 완화) ====
  mkdir -p .tmp .pip_cache || true
  export TMPDIR="$PWD/.tmp"
  export PIP_CACHE_DIR="$PWD/.pip_cache"

  echo "[INFO] PWD = $(pwd)"
  ls -al | sed 's/^/  /' || true

  # ==== stage paths to predictable names ====
  echo "[INFO] driverformer/:"; ls -al driverformer | sed 's/^/  /' || true
  [ -d driverformer ] || { echo "[FATAL] driverformer dir not staged"; ls -al; exit 3; }

  # ==== env ====
  export MPLBACKEND=Agg
  export TOKENIZERS_PARALLELISM=false
  export OMP_NUM_THREADS=!{task.cpus}
  export MKL_NUM_THREADS=!{task.cpus}
  export OPENBLAS_NUM_THREADS=!{task.cpus}
  export NUMEXPR_NUM_THREADS=!{task.cpus}
  export PYTHONPATH="$PWD:$PWD/driverformer${PYTHONPATH:+:$PYTHONPATH}"
  echo "[INFO] PYTHONPATH head = $(echo "$PYTHONPATH" | tr ':' '\n' | head -n 3)"

  # wheels link (optional)
  if [ -d "!{WHEELS_DIR}" ]; then
    [ "!{WHEELS_DIR}" = "wheels" ] || ln -snf "!{WHEELS_DIR}" wheels 2>/dev/null || cp -r "!{WHEELS_DIR}" wheels
    echo "[INFO] wheels/:"; ls -al wheels | sed 's/^/  /' || true
  else
    echo "[INFO] no wheels directory staged"
  fi

  # ----------------------------
  # pip 설치: 온라인 감지 → 온라인/오프라인 루트로 분기
  # ----------------------------
  ONLINE=0
  if ! ! { python - <<'PY'
import socket, sys
try:
    with socket.create_connection(("pypi.org", 443), timeout=3):
        pass
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
  }; then
    ONLINE=1
  fi
  echo "[INFO] Network to PyPI: $([ $ONLINE -eq 1 ] && echo ONLINE || echo OFFLINE)"

  # pip 공통 옵션(환경변수로 오버라이드 가능)
  : "${PIP_INDEX_URL:=https://pypi.org/simple}"
  : "${PIP_TRUSTED_HOST:=pypi.org files.pythonhosted.org}"
  PIP_OPTS=( --no-cache-dir --retries 5 --timeout 90 --progress-bar off )
  PIP_OPTS+=( --index-url "$PIP_INDEX_URL" )
  for host in $PIP_TRUSTED_HOST; do PIP_OPTS+=( --trusted-host "$host" ); done
  [ -n "${PIP_EXTRA_INDEX_URL:-}" ] && PIP_OPTS+=( --extra-index-url "$PIP_EXTRA_INDEX_URL" )

  # sitecustomize: 공유전략 file_system로 (SHM 사용량 완화)
  cat > sitecustomize.py <<'PY'
try:
    import torch, warnings
    torch.multiprocessing.set_sharing_strategy('file_system')
except Exception as e:
    print("[WARN] set_sharing_strategy failed:", e)
PY

  # constraints(제약) 파일: solver 폭주 방지용 최소 버전 고정
  cat > constraints.txt <<'REQ'
matplotlib==3.8.4
fonttools==4.53.1
kiwisolver==1.4.5
pillow>=10.2,<11
REQ

  SETUP_DEPS="!{params.setup_deps}"
  if [ "$SETUP_DEPS" = "true" ]; then
    echo "[SETUP] Installing Python runtime tools"
    python -m pip install -U pip wheel setuptools "${PIP_OPTS[@]}" || true

    if [ $ONLINE -eq 1 ]; then
      echo "[SETUP] Online mode: installing base constraints"
      # 제약 먼저 설치(충돌 완화)
      python -m pip install "${PIP_OPTS[@]}" -c constraints.txt -r constraints.txt || true

      if [ -f "!{REQS}" ] && [ "$(basename "!{REQS}")" = "requirements.txt" ]; then
        echo "[SETUP] Installing requirements.txt (filtered, constrained)"
        # CUDA/torch 등 무거운 항목 제외
        grep -viE '^(torch|torchvision|torchaudio|pytorch[-]?triton|triton|nvidia-|cuda|cudnn|cudatoolkit|cupy|jax|jaxlib)' "!{REQS}" > .req_filtered.txt || true
        if [ -s .req_filtered.txt ]; then
          # 1차: 제약 기반 설치 시도
          if ! python -m pip install "${PIP_OPTS[@]}" -c constraints.txt -r .req_filtered.txt; then
            echo "[WARN] Online install failed, trying --use-deprecated=legacy-resolver once"
            python -m pip install "${PIP_OPTS[@]}" -c constraints.txt -r .req_filtered.txt --use-deprecated=legacy-resolver || true
          fi
        else
          echo "[INFO] .req_filtered.txt empty – skipping requirements"
        fi
      fi
    fi

    # wheels 오프라인 폴백(온라인 실패 or 오프라인)
    if [ -d wheels ]; then
      echo "[SETUP] Installing from local wheels (offline fallback)"
      set +e
      [ -f wheels/requirements_wheels.txt ] && python -m pip install --no-index --find-links wheels -r wheels/requirements_wheels.txt
      ls wheels/*.whl >/dev/null 2>&1 && python -m pip install --no-index --find-links wheels wheels/*.whl
      set -e
    fi
  else
    echo "[SETUP] params.setup_deps=false → skipping pip installs"
  fi

  # import self-check
  python - <<'PYINFO'
import sys, importlib, traceback
print("python =", sys.executable)
print("sys.path head =", sys.path[:3])
try:
    import driverformer
    print("driverformer import: OK; version:", getattr(driverformer, "__version__", "NA"))
except Exception:
    print("driverformer import: FAIL — traceback:"); traceback.print_exc()
PYINFO

  # ---- args as bash arrays (quoting-safe) ----
  COMMON_ARGS=(
    --cls-file            '!{CLS}'
    --feat-file           '!{FEAT}'
    --mutations-file      '!{MUTS}'
    --out-dir             '!{params.out_dir}'
    --lr                  !{params.lr}
    --batch-size          !{params.batch_size}
    --epochs              !{params.epochs}
    --seed                !{params.seed}
    --d-model             !{params.d_model}
    --nhead               !{params.nhead}
    --num-layers          !{params.num_layers}
    --dim-feedforward     !{params.dim_feedforward}
    --dropout             !{params.dropout}
    --max-seq-len         !{params.max_seq_len}
    --overlap-factor      !{params.overlap_factor}
    --huber-factor        !{params.huber_factor}
    --cutmix-p            !{params.cutmix_p}
    --num-data-workers    !{params.num_data_workers}
    --torch-threads       !{params.torch_threads}
    --len-alpha           !{params.len_alpha}
    --res-beta            !{params.res_beta}
    --label-roll-width    !{params.label_roll_width}
    --pipeline-out-dir        '!{params.pipeline_out_dir}'
    --pipeline-chunk-size     !{params.pipeline_chunk_size}
    --pipeline-chunk-overlap  !{params.pipeline_chunk_overlap}
    --pipeline-min-distance   !{params.pipeline_min_distance}
    --pipeline-max-distance   !{params.pipeline_max_distance}
    --pipeline-sample-frac    !{params.pipeline_sample_frac}
    --pipeline-gmm-k          !{params.pipeline_gmm_k}
    --pipeline-beta           !{params.pipeline_beta}
    --pipeline-gamma          !{params.pipeline_gamma}
    --pipeline-seed           !{params.pipeline_seed}
    --pipeline-dp-gap-bp      !{params.pipeline_dp_gap_bp}
    --postsel-fdr-method      '!{params.postsel_fdr_method}'
    --postsel-bootstrap       !{params.postsel_bootstrap}
    --postsel-lambda-start    '!{params.postsel_lambda_start}'
    --postsel-lambda-end      '!{params.postsel_lambda_end}'
    --postsel-lambda-step     '!{params.postsel_lambda_step}'
    --postsel-pi0-floor       '!{params.postsel_pi0_floor}'
    --postsel-pi0-ceil        '!{params.postsel_pi0_ceil}'
  )
  [ -n "$SEGLEN" ] && COMMON_ARGS+=( --segment-lengths $SEGLEN )

  FLAGS=()
  [ "!{params.use_mad}"      = "true" ] && FLAGS+=( --use-mad )
  [ "!{params.label_roll}"   = "true" ] && FLAGS+=( --label-roll )
  [ "!{params.run_pipeline}" = "true" ] && FLAGS+=( --run-pipeline )

  # ---- run: module preferred, fallback to staged script ----
  # ---- run: module preferred, fallback to staged script ----
  # python 스크립트의 종료 코드를 직접 if문에서 확인하는 방식으로 변경
  if python - <<'PY'
  import importlib.util, sys
  sys.exit(0 if importlib.util.find_spec("driverformer") else 1)
  PY
  then
    # 종료 코드가 0일 때 (driverformer 모듈이 설치되어 있을 때)
      echo "[RUN] python -m driverformer ..."
      set -x; python -u -m driverformer "${COMMON_ARGS[@]}" "${FLAGS[@]}"; set +x
  elif [ -f "trainDriverFormer.py" ]; then
      # 종료 코드가 0이 아니고, trainDriverFormer.py 파일이 있을 때
      echo "[RUN] python trainDriverFormer.py ..."
      set -x; python -u "trainDriverFormer.py" "${COMMON_ARGS[@]}" "${FLAGS[@]}"; set +x
  else
      # 둘 다 아닐 때
      echo "[ERROR] Neither 'driverformer' package nor 'trainDriverFormer.py' staged."
      exit 2
  fi  

  echo "[DONE] DriverFormer finished."
  '''
}

// ---- Workflow ----
workflow {
  if( !params.cls_file )       error "Missing required param: --cls_file"
  if( !params.feat_file )      error "Missing required param: --feat_file"
  if( !params.mutations_file ) error "Missing required param: --mutations_file"

  def ch_cls   = Channel.fromPath(params.cls_file)
  def ch_feat  = Channel.fromPath(params.feat_file)
  def ch_muts  = Channel.fromPath(params.mutations_file)

  def ch_pkg    = Channel.fromPath("${projectDir}/driverformer",         checkIfExists: true)
  def ch_train  = Channel.fromPath("${projectDir}/trainDriverFormer.py", checkIfExists: true)

  def ch_reqs_exist = Channel.fromPath("${projectDir}/requirements.txt",         checkIfExists: true)
  def ch_reqs_dummy = Channel.fromPath("${projectDir}/driverformer/__init__.py", checkIfExists: true)
  def ch_reqs  = ch_reqs_exist.ifEmpty(ch_reqs_dummy)

  def ch_wheels_exist = Channel.fromPath("${projectDir}/wheels",                 checkIfExists: true)
  def ch_wheels_dummy = Channel.fromPath("${projectDir}/driverformer/__init__.py", checkIfExists: true)
  def ch_wheels = ch_wheels_exist.ifEmpty(ch_wheels_dummy)

  DRIVERFORMER_RUN(ch_cls, ch_feat, ch_muts, ch_pkg, ch_train, ch_reqs, ch_wheels)
}
