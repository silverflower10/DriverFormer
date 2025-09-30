// main.nf — inline shell only (no external .sh), parser-safe
nextflow.enable.dsl = 2

params.cls_file        = params.cls_file        ?: null
params.feat_file       = params.feat_file       ?: null
params.mutations_file  = params.mutations_file  ?: null
params.out_dir         = params.out_dir         ?: 'results/run'

process DRIVERFORMER_RUN {
  cpus 8
  memory '32 GB'
  time '72h'
  publishDir "${params.out_dir}", mode: 'copy', overwrite: true

  input:
    path CLS
    path FEAT
    path MUTS
    path DF_PKG
    path TRAIN_PY
    path REQS
    path WHEELS_DIR

  output:
    path "stdout.txt"
    path "stderr.txt"

  // inline bash (triple SINGLE quotes only)
  shell:
  '''
  set -euo pipefail
  exec > >(tee stdout.txt) 2> >(tee stderr.txt >&2)

  echo "[INFO] PWD=$(pwd)"
  ls -al | sed 's/^/  /' || true
  echo "[INFO] driverformer/:"
  ls -al driverformer | sed 's/^/  /' || true

  # ===== ENV =====
  export MPLBACKEND=Agg
  export TOKENIZERS_PARALLELISM=false
  export OMP_NUM_THREADS=!{task.cpus}
  export MKL_NUM_THREADS=!{task.cpus}
  export OPENBLAS_NUM_THREADS=!{task.cpus}
  export NUMEXPR_NUM_THREADS=!{task.cpus}
  export PYTHONPATH="$PWD:$PWD/driverformer${PYTHONPATH:+:$PYTHONPATH}"

  # ===== wheels fallback =====
  if [ -d "!{WHEELS_DIR}" ]; then
    [ "!{WHEELS_DIR}" = "wheels" ] || ln -s "!{WHEELS_DIR}" wheels 2>/dev/null || cp -r "!{WHEELS_DIR}" wheels
    ls -al wheels | sed 's/^/  /' || true
  fi

  # ===== SSL-safe pip installs =====
  PIP_OPTS="--no-cache-dir --retries 5 --timeout 60 --index-url https://pypi.org/simple --trusted-host pypi.org --trusted-host files.pythonhosted.org"
  python -m pip install -U pip wheel setuptools $PIP_OPTS || true

  if [ -f "!{REQS}" ] && [ "$(basename "!{REQS}")" = "requirements.txt" ]; then
    grep -viE '^(torch|torchvision|torchaudio|pytorch-triton|triton|nvidia-|cuda|cudnn|cublas|cusparse|cusolver|nccl|cudatoolkit)' "!{REQS}" > .req_filtered.txt || true
    [ -s .req_filtered.txt ] && python -m pip install $PIP_OPTS -r .req_filtered.txt || true
  fi

  if [ -d wheels ]; then
    [ -f wheels/requirements_wheels.txt ] && python -m pip install --no-index --find-links wheels -r wheels/requirements_wheels.txt || true
    ls wheels/*.whl >/dev/null 2>&1 && python -m pip install --no-index --find-links wheels wheels/*.whl || true
  fi

  MISS=$(python - <<'PY'
mods=["pandas","pyarrow","scikit-learn","tqdm","pyyaml","matplotlib","statsmodels","patsy","rotary_embedding_torch","einops"]
import importlib.util
print(" ".join([m for m in mods if not importlib.util.find_spec(m)]))
PY
)
  [ -n "$MISS" ] && python -m pip install $PIP_OPTS $MISS || true

  # ===== COMMON_ARGS (segment_lengths는 조건부 추가) =====
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
    --postsel-lambda-start     '!{params.postsel_lambda_start}' \
    --postsel-lambda-end       '!{params.postsel_lambda_end}' \
    --postsel-lambda-step      '!{params.postsel_lambda_step}' \
    --postsel-pi0-floor        '!{params.postsel_pi0_floor}' \
    --postsel-pi0-ceil         '!{params.postsel_pi0_ceil}'"

  _raw_seglen="!{ (params.segment_lengths instanceof List) ? params.segment_lengths.join(' ') : (params.segment_lengths ? params.segment_lengths.toString() : '') }"
  SEGLEN=$(echo "$_raw_seglen" | tr ',' ' ' | sed -e 's/^ *//; s/ *$//' -e 's/  \+/ /g' -e 's/^\"//; s/\"$//')
  SEGLEN=$(for t in $SEGLEN; do case "$t" in (*[!0-9]*) ;; (*) printf '%s ' "$t";; esac; done | sed 's/ *$//')
  [ -n "$SEGLEN" ] && COMMON_ARGS="$COMMON_ARGS --segment-lengths $SEGLEN"

  FLAGS=""
  [ "!{params.use_mad}"      = "true" ] && FLAGS="$FLAGS --use-mad"
  [ "!{params.label_roll}"   = "true" ] && FLAGS="$FLAGS --label-roll"
  [ "!{params.run_pipeline}" = "true" ] && FLAGS="$FLAGS --run-pipeline"

  # 모듈 우선, 폴백 스크립트
  if python - <<'PY'; then
import importlib.util as u; print(1 if u.find_spec("driverformer") else 0)
PY
  then
    set -x; python -u -m driverformer ${COMMON_ARGS} ${FLAGS}; set +x
  elif [ -f "trainDriverFormer.py" ]; then
    set -x; python -u trainDriverFormer.py ${COMMON_ARGS} ${FLAGS}; set +x
  else
    echo "[ERROR] Neither 'driverformer' module nor 'trainDriverFormer.py' present."; exit 2
  fi

  echo "[DONE] DriverFormer finished."
  '''
}

// ---- Workflow ----
workflow {
  if( !params.cls_file )       error "Missing required param: --cls_file"
  if( !params.feat_file )      error "Missing required param: --feat_file"
  if( !params.mutations_file ) error "Missing required param: --mutations_file"

  ch_cls  = Channel.fromPath(params.cls_file)
  ch_feat = Channel.fromPath(params.feat_file)
  ch_muts = Channel.fromPath(params.mutations_file)

  ch_pkg   = Channel.fromPath("${projectDir}/driverformer",           checkIfExists: true)
  ch_train = Channel.fromPath("${projectDir}/trainDriverFormer.py",   checkIfExists: true)

  ch_reqs_exist = Channel.fromPath("${projectDir}/requirements.txt",  checkIfExists: true)
  ch_dummy      = Channel.fromPath("${projectDir}/driverformer/__init__.py", checkIfExists: true)
  ch_reqs  = ch_reqs_exist.ifEmpty(ch_dummy)

  ch_wheels_exist = Channel.fromPath("${projectDir}/wheels",          checkIfExists: true)
  ch_wheels = ch_wheels_exist.ifEmpty(ch_dummy)

  DRIVERFORMER_RUN(ch_cls, ch_feat, ch_muts, ch_pkg, ch_train, ch_reqs, ch_wheels)
}
