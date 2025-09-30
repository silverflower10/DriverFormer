// main.nf — stage repo files via channels (DSL2), SSL-safe auto-install deps, run from workdir
nextflow.enable.dsl = 2

// ---- Param defaults (no params{} block) ----
params.cls_file        = params.cls_file        ?: null
params.feat_file       = params.feat_file       ?: null
params.mutations_file  = params.mutations_file  ?: null
params.out_dir         = params.out_dir         ?: 'results/run'

// (학습/파이프라인 파라미터 — 동일)
params.lr              = (params.lr ?: 2e-4)
params.batch_size      = (params.batch_size ?: 128)
params.epochs          = (params.epochs ?: 2)
params.seed            = (params.seed ?: 42)
params.d_model         = (params.d_model ?: 768)
params.nhead           = (params.nhead  ?: 8)
params.num_layers      = (params.num_layers ?: 6)
params.dim_feedforward = (params.dim_feedforward ?: 3072)
params.dropout         = (params.dropout ?: 0.2)
params.max_seq_len     = (params.max_seq_len ?: 1024)
params.segment_lengths = (params.segment_lengths ?: [10,50,100])
params.overlap_factor  = (params.overlap_factor ?: 0.3)
params.use_mad         = (params.use_mad ?: true)
params.huber_factor    = (params.huber_factor ?: 3.0)
params.cutmix_p        = (params.cutmix_p ?: 0.2)
params.num_data_workers= (params.num_data_workers ?: 8)
params.torch_threads   = (params.torch_threads ?: 8)
params.len_alpha       = (params.len_alpha ?: 0.5)
params.res_beta        = (params.res_beta  ?: 0.5)
params.label_roll      = (params.label_roll ?: true)
params.label_roll_width= (params.label_roll_width ?: 2)

params.run_pipeline          = (params.run_pipeline ?: true)
params.pipeline_out_dir      = (params.pipeline_out_dir ?: "${params.out_dir}/postproc_k_auto")
params.pipeline_chunk_size   = (params.pipeline_chunk_size   ?: 1000000)
params.pipeline_chunk_overlap= (params.pipeline_chunk_overlap?: 100000)
params.pipeline_min_distance = (params.pipeline_min_distance ?: 0)
params.pipeline_max_distance = (params.pipeline_max_distance ?: 100000)
params.pipeline_sample_frac  = (params.pipeline_sample_frac  ?: 0.01)
params.pipeline_gmm_k        = (params.pipeline_gmm_k        ?: 2)
params.pipeline_beta         = (params.pipeline_beta         ?: 1.0)
params.pipeline_gamma        = (params.pipeline_gamma        ?: 0.0)
params.pipeline_seed         = (params.pipeline_seed         ?: 42)
params.pipeline_dp_gap_bp    = (params.pipeline_dp_gap_bp    ?: 0)
params.postsel_fdr_method    = (params.postsel_fdr_method    ?: 'storey')
params.postsel_bootstrap     = (params.postsel_bootstrap     ?: 400)
params.postsel_lambda_start  = (params.postsel_lambda_start  ?: 0.20)
params.postsel_lambda_end    = (params.postsel_lambda_end    ?: 0.95)
params.postsel_lambda_step   = (params.postsel_lambda_step   ?: 0.01)
params.postsel_pi0_floor     = (params.postsel_pi0_floor     ?: 0.01)
params.postsel_pi0_ceil      = (params.postsel_pi0_ceil      ?: 1.0)

// ---- Process ----
process DRIVERFORMER_RUN {
  tag "driverformer"
  cpus 8
  memory '32 GB'
  time '72h'
  publishDir "${params.out_dir}", mode: 'copy', overwrite: true

  input:
    path CLS
    path FEAT
    path MUTS
    path DF_PKG           // driverformer/ 디렉토리
    path TRAIN_PY         // trainDriverFormer.py
    path REQS             // requirements.txt 또는 더미 파일(항상 전달)
    path WHEELS_DIR       // wheels 디렉토리 또는 더미(항상 전달)

  output:
    path "stdout.txt"
    path "stderr.txt"

  shell:
  '''
  set -euo pipefail
  exec > >(tee stdout.txt) 2> >(tee stderr.txt >&2)

  echo "[INFO] PWD = $(pwd)"
  ls -al | sed 's/^/  /' || true
  echo "[INFO] driverformer/:"; ls -al driverformer | sed 's/^/  /' || true

  export MPLBACKEND=Agg
  export TOKENIZERS_PARALLELISM=false
  export OMP_NUM_THREADS=!{task.cpus}
  export MKL_NUM_THREADS=!{task.cpus}
  export OPENBLAS_NUM_THREADS=!{task.cpus}
  export NUMEXPR_NUM_THREADS=!{task.cpus}

  # PYTHONPATH
  export PYTHONPATH="$PWD:$PWD/driverformer${PYTHONPATH:+:$PYTHONPATH}"
  echo "[INFO] PYTHONPATH head = $(echo "$PYTHONPATH" | tr ':' '\n' | head -n 3)"

  # ---- wheels 디렉토리 표준 이름으로 연결(있을 때만) ----
  if [ -d "!{WHEELS_DIR}" ]; then
    [ "!{WHEELS_DIR}" = "wheels" ] || ln -s "!{WHEELS_DIR}" wheels 2>/dev/null || cp -r "!{WHEELS_DIR}" wheels
    echo "[INFO] wheels/ staged:"; ls -al wheels | sed 's/^/  /' || true
  else
    echo "[INFO] no wheels directory staged"
  fi

  # ===== pip 설치(SSL-safe) =====
  PIP_OPTS="--no-cache-dir --retries 5 --timeout 60 \
            --index-url https://pypi.org/simple \
            --trusted-host pypi.org --trusted-host files.pythonhosted.org"
  python -m pip install -U pip wheel setuptools $PIP_OPTS || true

  # requirements.txt 있을 때만 설치
  if [ -f "!{REQS}" ] && [ "$(basename "!{REQS}")" = "requirements.txt" ]; then
    echo "[SETUP] Installing requirements.txt (filtered)"
    grep -viE '^(torch|torchvision|torchaudio|pytorch-triton|triton|nvidia-|cuda|cudnn|cudatoolkit)' \
      "!{REQS}" > .req_filtered.txt || true
    [ -s .req_filtered.txt ] && python -m pip install $PIP_OPTS -r .req_filtered.txt || echo "[SETUP] nothing to install from requirements.txt"
  else
    echo "[SETUP] No requirements.txt — skipping"
  fi

  # 로컬 wheels 폴백
  if [ -d wheels ]; then
    echo "[SETUP] Installing from local wheels (offline fallback)"
    # 목록 파일이 있으면 우선 사용
    if [ -f wheels/requirements_wheels.txt ]; then
      python -m pip install --no-index --find-links wheels -r wheels/requirements_wheels.txt || true
    fi
    # 남은 *.whl 일괄 설치(순서 무관)
    ls wheels/*.whl >/dev/null 2>&1 && python -m pip install --no-index --find-links wheels wheels/*.whl || true
  fi

  # 핵심 패키지 보충
  MISSING=$(python - <<'PY'
mods = ["pandas","pyarrow","scikit-learn","tqdm","pyyaml","matplotlib","statsmodels","patsy","rotary_embedding_torch","einops"]
import importlib.util
print(" ".join([m for m in mods if not importlib.util.find_spec(m)]))
PY
)
  if [ -n "$MISSING" ]; then
    echo "[SETUP] Installing missing packages: $MISSING"
    python -m pip install $PIP_OPTS $MISSING || echo "[WARN] missing packages install failed"
  fi
  # ======================================

  which python || true
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

  COMMON_ARGS="\
    --cls-file            '!{CLS}' \
    --feat-file           '!{FEAT}' \
    --mutations-file      '!{MUTS}' \
    --out-dir             '!{params.out_dir}' \
    --lr                  !{params.lr} \
    --batch-size          !{params.batch_size} \
    --epochs              !{params.epochs} \
    --seed                !{params.seed} \
    --d-model             !{params.d_model} \
    --nhead               !{params.nhead} \
    --num-layers          !{params.num_layers} \
    --dim-feedforward     !{params.dim_feedforward} \
    --dropout             !{params.dropout} \
    --max-seq-len         !{params.max_seq_len} \
    --overlap-factor      !{params.overlap_factor} \
    --huber-factor        !{params.huber_factor} \
    --cutmix-p            !{params.cutmix_p} \
    --num-data-workers    !{params.num_data_workers} \
    --torch-threads       !{params.torch_threads} \
    --len-alpha           !{params.len_alpha} \
    --res-beta            !{params.res_beta} \
    --label-roll-width    !{params.label_roll_width} \
    --pipeline-out-dir         '!{params.pipeline_out_dir}' \
    --pipeline-chunk-size      !{params.pipeline_chunk_size} \
    --pipeline-chunk-overlap   !{params.pipeline_chunk_overlap} \
    --pipeline-min-distance    !{params.pipeline_min_distance} \
    --pipeline-max-distance    !{params.pipeline_max_distance} \
    --pipeline-sample-frac     !{params.pipeline_sample_frac} \
    --pipeline-gmm-k           !{params.pipeline_gmm_k} \
    --pipeline-beta            !{params.pipeline_beta} \
    --pipeline-gamma          !{params.pipeline_gamma} \
    --pipeline-seed            !{params.pipeline_seed} \
    --pipeline-dp-gap-bp       !{params.pipeline_dp_gap_bp} \
    --postsel-fdr-method       '!{params.postsel_fdr_method}' \
    --postsel-bootstrap        !{params.postsel_bootstrap} \
    --postsel-lambda-start     !{params.postsel_lambda_start} \
    --postsel-lambda-end       !{params.postsel_lambda_end} \
    --postsel-lambda-step      !{params.postsel_lambda_step} \
    --postsel-pi0-floor        !{params.postsel_pi0_floor} \
    --postsel-pi0-ceil         !{params.postsel_pi0_ceil}'"

  # --- robust segment_lengths handling (CloudOS UI-safe) ---
  _raw_seglen="!{ (params.segment_lengths instanceof List) ? params.segment_lengths.join(' ') : (params.segment_lengths ? params.segment_lengths.toString() : '') }"
  # 쉼표→공백, 양끝/중복 공백/따옴표 제거
  SEGLEN=$(echo "$_raw_seglen" | tr ',' ' ' | sed -e 's/^ *//; s/ *$//' -e 's/  \+/ /g' -e 's/^"//; s/"$//')
  # 값이 있을 때만 옵션 추가(비어있으면 붙이지 않음)

  FLAGS=""
  [ "!{params.use_mad}"      = "true" ] && FLAGS="${FLAGS} --use-mad"
  [ "!{params.label_roll}"   = "true" ] && FLAGS="${FLAGS} --label-roll"
  [ "!{params.run_pipeline}" = "true" ] && FLAGS="${FLAGS} --run-pipeline"

  if python - <<<'import importlib.util; import sys; print(1 if importlib.util.find_spec("driverformer") else 0)'; then
    echo "[RUN] python -m driverformer ..."
    set -x; python -u -m driverformer ${COMMON_ARGS} ${FLAGS}; set +x
  else
    if [ -f "trainDriverFormer.py" ]; then
      echo "[RUN] python trainDriverFormer.py ..."
      set -x; python -u trainDriverFormer.py ${COMMON_ARGS} ${FLAGS}; set +x
    else
      echo "[ERROR] Neither 'driverformer' package nor 'trainDriverFormer.py' staged."
      exit 2
    fi
  fi

  echo "[DONE] DriverFormer finished."
  '''
}

// ---- Workflow ----
workflow {
  if( !params.cls_file )       error "Missing required param: --cls_file"
  if( !params.feat_file )      error "Missing required param: --feat_file"
  if( !params.mutations_file ) error "Missing required param: --mutations_file"

  ch_cls   = Channel.fromPath(params.cls_file)
  ch_feat  = Channel.fromPath(params.feat_file)
  ch_muts  = Channel.fromPath(params.mutations_file)

  ch_pkg    = Channel.fromPath("${projectDir}/driverformer",         checkIfExists: true)
  ch_train  = Channel.fromPath("${projectDir}/trainDriverFormer.py", checkIfExists: true)

  // requirements.txt(있으면) 또는 더미 파일(항상 전달)
  ch_reqs_exist = Channel.fromPath("${projectDir}/requirements.txt",         checkIfExists: true)
  ch_reqs_dummy = Channel.fromPath("${projectDir}/driverformer/__init__.py", checkIfExists: true)
  ch_reqs  = ch_reqs_exist.ifEmpty(ch_reqs_dummy)

  // wheels 디렉토리(있으면) 또는 더미(항상 전달)
  ch_wheels_exist = Channel.fromPath("${projectDir}/wheels",                 checkIfExists: true)
  ch_wheels_dummy = Channel.fromPath("${projectDir}/driverformer/__init__.py", checkIfExists: true)
  ch_wheels = ch_wheels_exist.ifEmpty(ch_wheels_dummy)

  DRIVERFORMER_RUN(ch_cls, ch_feat, ch_muts, ch_pkg, ch_train, ch_reqs, ch_wheels)
}
