// main.nf — DriverFormer DSL2 (parser-safe shell: + ''' + !{..}; robust pkg/script detection)
nextflow.enable.dsl = 2

// ---- Param defaults (no params{} block) ----
params.cls_file        = params.cls_file        ?: null
params.feat_file       = params.feat_file       ?: null
params.mutations_file  = params.mutations_file  ?: null

params.out_dir         = params.out_dir         ?: 'results/run'
params.post_dir        = params.post_dir        ?: "${params.out_dir}/postproc_k_auto"

params.lr              = (params.lr              ?: 2e-4)
params.batch_size      = (params.batch_size      ?: 128)
params.epochs          = (params.epochs          ?: 2)
params.seed            = (params.seed            ?: 42)
params.d_model         = (params.d_model         ?: 768)
params.nhead           = (params.nhead           ?: 8)
params.num_layers      = (params.num_layers      ?: 6)
params.dim_feedforward = (params.dim_feedforward ?: 3072)
params.dropout         = (params.dropout         ?: 0.2)
params.max_seq_len     = (params.max_seq_len     ?: 1024)
params.segment_lengths = (params.segment_lengths ?: [10,50,100])
params.overlap_factor  = (params.overlap_factor  ?: 0.3)
params.use_mad         = (params.use_mad         ?: true)
params.huber_factor    = (params.huber_factor    ?: 3.0)
params.cutmix_p        = (params.cutmix_p        ?: 0.2)
params.num_data_workers= (params.num_data_workers?: 8)
params.torch_threads   = (params.torch_threads   ?: 8)
params.len_alpha       = (params.len_alpha       ?: 0.5)
params.res_beta        = (params.res_beta        ?: 0.5)
params.label_roll      = (params.label_roll      ?: true)
params.label_roll_width= (params.label_roll_width?: 2)

params.run_pipeline          = (params.run_pipeline          ?: true)
params.pipeline_out_dir      = (params.pipeline_out_dir      ?: "${params.post_dir}")
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
params.pipeline_presmooth_bins = (params.pipeline_presmooth_bins ?: 2)

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

  output:
    path "stdout.txt"
    path "stderr.txt"

  shell:
  '''
  set -euo pipefail
  exec > >(tee stdout.txt) 2> >(tee stderr.txt >&2)

  echo "[INFO] projectDir = !{projectDir}"
  echo "[INFO] workdir    = $(pwd)"
  HEAD=$(git -C "!{projectDir}" rev-parse --short HEAD 2>/dev/null || true)
  [ -n "$HEAD" ] && echo "[INFO] repo HEAD  = $HEAD" || echo "[INFO] repo HEAD  = unknown"

  # Safe env
  export MPLBACKEND=Agg
  export TOKENIZERS_PARALLELISM=false
  export OMP_NUM_THREADS=!{task.cpus}
  export MKL_NUM_THREADS=!{task.cpus}
  export OPENBLAS_NUM_THREADS=!{task.cpus}
  export NUMEXPR_NUM_THREADS=!{task.cpus}

  # PYTHONPATH: repo root + package dir
  if [ -z "${PYTHONPATH+x}" ]; then
    export PYTHONPATH="!{projectDir}:!{projectDir}/driverformer"
  else
    export PYTHONPATH="!{projectDir}:!{projectDir}/driverformer:${PYTHONPATH}"
  fi
  echo "[INFO] PYTHONPATH head = $(echo "$PYTHONPATH" | tr ':' '\n' | head -n 3)"

  echo "=== repo tree (root) ==="; (cd "!{projectDir}" && ls -al | sed 's/^/  /') || true
  echo "=== repo tree (driverformer) ==="; (cd "!{projectDir}/driverformer" && ls -al | sed 's/^/  /') || echo "[WARN] driverformer dir missing"

  # Auto-install requirements (skip torch/CUDA)
  if [ -f "!{projectDir}/requirements.txt" ]; then
    echo "[SETUP] Installing requirements.txt"
    python -m pip install --no-python-version-warning --no-cache-dir -U pip wheel setuptools
    grep -viE '^(torch|torchvision|torchaudio|pytorch-triton|triton|nvidia-|cuda|cudnn|cudatoolkit)' \
      "!{projectDir}/requirements.txt" > .req_filtered.txt || true
    if [ -s .req_filtered.txt ]; then
      python -m pip install --no-cache-dir -r .req_filtered.txt || echo "[WARN] pip install issues; continue"
    else
      echo "[SETUP] nothing to install"
    fi
  else
    echo "[SETUP] No requirements.txt — skip"
  fi

  # Debug: env + import (print full traceback on failure)
  nvidia-smi || true
  which python || true
  python - <<'PYINFO'
import os, sys, importlib, traceback, torch
print("python =", sys.executable)
print("sys.path head =", sys.path[:3])
print("torch  =", torch.__version__, "| cuda?", torch.cuda.is_available(), "| #GPU =", torch.cuda.device_count())
try:
    m = importlib.import_module("driverformer")
    print("driverformer import: OK; version:", getattr(m, "__version__", "NA"))
except Exception:
    print("driverformer import: FAIL — traceback:"); traceback.print_exc()
PYINFO
  echo "================================================="

  # Prefer simple directory check for package presence (robust)
  PKG_DIR="!{projectDir}/driverformer"
  if [ -d "$PKG_DIR" ] && [ -f "$PKG_DIR/__init__.py" ]; then
    LAUNCH="pkgdir"
  else
    LAUNCH="nomodule"
  fi
  echo "[INFO] launch mode by dir check = ${LAUNCH}"

  # Fallback script lookup (recursive)
  SCRIPT_PATH=$(python - <<'PY'
import os, glob
root = r"!{projectDir}"
cands = []
patterns = [
  "trainDriverFormer.py",
  "**/trainDriverFormer.py",
  "**/*train*DriverFormer*.py",
  "scripts/*.py",
  "train/*.py",
]
for pat in patterns:
  cands.extend(glob.glob(os.path.join(root, pat), recursive=True))
print(cands[0] if cands else "")
PY
)
  echo "[INFO] script candidate = ${SCRIPT_PATH:-<none>}"

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
    --segment-lengths     !{params.segment_lengths.join(' ')} \
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
    --pipeline-presmooth-bins  !{params.pipeline_presmooth_bins} \
    --postsel-fdr-method       '!{params.postsel_fdr_method}' \
    --postsel-bootstrap        !{params.postsel_bootstrap} \
    --postsel-lambda-start     !{params.postsel_lambda_start} \
    --postsel-lambda-end       !{params.postsel_lambda_end} \
    --postsel-lambda-step      !{params.postsel_lambda_step} \
    --postsel-pi0-floor        !{params.postsel_pi0_floor} \
    --postsel-pi0-ceil         !{params.postsel_pi0_ceil}'"

  FLAGS=""
  [ "!{params.use_mad}"      = "true" ] && FLAGS="${FLAGS} --use-mad"
  [ "!{params.label_roll}"   = "true" ] && FLAGS="${FLAGS} --label-roll"
  [ "!{params.run_pipeline}" = "true" ] && FLAGS="${FLAGS} --run-pipeline"

  if [ "${LAUNCH}" = "pkgdir" ]; then
    echo "[RUN] python -m driverformer ..."
    set -x; python -u -m driverformer ${COMMON_ARGS} ${FLAGS}; set +x
  elif [ -n "${SCRIPT_PATH:-}" ] && [ -f "${SCRIPT_PATH}" ]; then
    echo "[RUN] python '${SCRIPT_PATH}' ..."
    set -x; python -u "${SCRIPT_PATH}" ${COMMON_ARGS} ${FLAGS}; set +x
  else
    echo "[ERROR] Neither 'driverformer' package (dir+__init__) nor 'trainDriverFormer.py' found."
    echo "[HINT] Check repo tree logs above. Ensure 'driverformer/__init__.py' exists or add trainDriverFormer.py to repo."
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

  ch_cls  = Channel.fromPath(params.cls_file)
  ch_feat = Channel.fromPath(params.feat_file)
  ch_muts = Channel.fromPath(params.mutations_file)

  DRIVERFORMER_RUN(ch_cls, ch_feat, ch_muts)
}
