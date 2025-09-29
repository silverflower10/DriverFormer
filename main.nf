// main.nf — DriverFormer DSL2 (robust: module import or auto-find script; PYTHONPATH safe)
nextflow.enable.dsl = 2

// ===== Params =====
params.cls_file       = null
params.feat_file      = null
params.mutations_file = null

params.out_dir        = 'results/run'
params.post_dir       = "${params.out_dir}/postproc_k_auto"

params.lr = 2e-4; params.batch_size = 128; params.epochs = 2; params.seed = 42
params.d_model = 768; params.nhead = 8; params.num_layers = 6
params.dim_feedforward = 3072; params.dropout = 0.2; params.max_seq_len = 1024
params.segment_lengths = [10,50,100]; params.overlap_factor = 0.3
params.use_mad = true; params.huber_factor = 3.0; params.cutmix_p = 0.2
params.num_data_workers = 8; params.torch_threads = 8
params.len_alpha = 0.5; params.res_beta = 0.5
params.label_roll = true; params.label_roll_width = 2

// pipeline/post-selection
params.run_pipeline = true
params.pipeline_out_dir = "${params.post_dir}"
params.pipeline_chunk_size = 1_000_000
params.pipeline_chunk_overlap = 100_000
params.pipeline_min_distance = 0; params.pipeline_max_distance = 100_000
params.pipeline_sample_frac = 0.01; params.pipeline_gmm_k = 2
params.pipeline_beta = 1.0; params.pipeline_gamma = 0.0; params.pipeline_seed = 42
params.pipeline_dp_gap_bp = 0; params.pipeline_presmooth_bins = 2

params.postsel_fdr_method = 'storey'; params.postsel_bootstrap = 400
params.postsel_lambda_start = 0.20; params.postsel_lambda_end = 0.95
params.postsel_lambda_step = 0.01; params.postsel_pi0_floor = 0.01; params.postsel_pi0_ceil = 1.0

def need(n,v){ if(!v) error "Missing required param: --${n}" }

process DRIVERFORMER_RUN {
  tag "driverformer"
  cpus 8; memory '32 GB'; time '72h'
  publishDir "${params.out_dir}", mode: 'copy', overwrite: true

  input:
    path CLS
    path FEAT
    path MUTS

  output:
    path "stdout.txt"
    path "stderr.txt"

  script:
  """
  set -euo pipefail
  exec > >(tee stdout.txt) 2> >(tee stderr.txt >&2)

  echo "[INFO] projectDir = ${projectDir}"
  echo "[INFO] workdir    = \$(pwd)"
  echo "[INFO] repo HEAD  = \$(git -C '${projectDir}' rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
  echo "[INFO] repo tree  :"
  (cd '${projectDir}' && ls -al | sed 's/^/  /') || true

  export MPLBACKEND=Agg
  export TOKENIZERS_PARALLELISM=false
  export OMP_NUM_THREADS=${task.cpus}
  export MKL_NUM_THREADS=${task.cpus}
  export OPENBLAS_NUM_THREADS=${task.cpus}
  export NUMEXPR_NUM_THREADS=${task.cpus}
  # safe PYTHONPATH
  export PYTHONPATH="${projectDir}${PYTHONPATH:+:\$PYTHONPATH}"

  # ---- Debug: GPU/Python/driverformer import ----
  nvidia-smi || true
  which python || true
  python - <<'PYINFO'
import torch, sys, importlib
print("python =", sys.executable)
print("torch  =", torch.__version__, "| cuda?", torch.cuda.is_available(), "| #GPU =", torch.cuda.device_count())
try:
    m = importlib.import_module("driverformer")
    print("driverformer import: OK; version:", getattr(m, "__version__", "NA"))
except Exception as e:
    print("driverformer import: FAIL;", repr(e))
PYINFO
  echo "================================================="

  # Decide launch mode
  LAUNCH=\$(python - <<'PY'
import importlib
try:
    import driverformer
    print("module")
except Exception:
    print("nomodule")
PY
)
  echo "[INFO] launch mode = \${LAUNCH}"

  # Try to locate trainDriverFormer.py within repo (depth≤3)
  SCRIPT_PATH=\$(find '${projectDir}' -maxdepth 3 -type f -name 'trainDriverFormer.py' | head -n1 || true)
  echo "[INFO] script candidate = \${SCRIPT_PATH:-<none>}"

  COMMON_ARGS="\\
    --cls-file            '${CLS}' \\
    --feat-file           '${FEAT}' \\
    --mutations-file      '${MUTS}' \\
    --out-dir             '${params.out_dir}' \\
    --lr                  ${params.lr} \\
    --batch-size          ${params.batch_size} \\
    --epochs              ${params.epochs} \\
    --seed                ${params.seed} \\
    --d-model             ${params.d_model} \\
    --nhead               ${params.nhead} \\
    --num-layers          ${params.num_layers} \\
    --dim-feedforward     ${params.dim_feedforward} \\
    --dropout             ${params.dropout} \\
    --max-seq-len         ${params.max_seq_len} \\
    --segment-lengths     ${params.segment_lengths.join(' ')} \\
    --overlap-factor      ${params.overlap_factor} \\
    --huber-factor        ${params.huber_factor} \\
    --cutmix-p            ${params.cutmix_p} \\
    --num-data-workers    ${params.num_data_workers} \\
    --torch-threads       ${params.torch_threads} \\
    --len-alpha           ${params.len_alpha} \\
    --res-beta            ${params.res_beta} \\
    --label-roll-width    ${params.label_roll_width} \\
    --pipeline-out-dir         '${params.pipeline_out_dir}' \\
    --pipeline-chunk-size      ${params.pipeline_chunk_size} \\
    --pipeline-chunk-overlap   ${params.pipeline_chunk_overlap} \\
    --pipeline-min-distance    ${params.pipeline_min_distance} \\
    --pipeline-max-distance    ${params.pipeline_max_distance} \\
    --pipeline-sample-frac     ${params.pipeline_sample_frac} \\
    --pipeline-gmm-k           ${params.pipeline_gmm_k} \\
    --pipeline-beta            ${params.pipeline_beta} \\
    --pipeline-gamma          ${params.pipeline_gamma} \\
    --pipeline-seed            ${params.pipeline_seed} \\
    --pipeline-dp-gap-bp       ${params.pipeline_dp_gap_bp} \\
    --pipeline-presmooth-bins  ${params.pipeline_presmooth_bins} \\
    --postsel-fdr-method       '${params.postsel_fdr_method}' \\
    --postsel-bootstrap        ${params.postsel_bootstrap} \\
    --postsel-lambda-start     ${params.postsel_lambda_start} \\
    --postsel-lambda-end       ${params.postsel_lambda_end} \\
    --postsel-lambda-step      ${params.postsel_lambda_step} \\
    --postsel-pi0-floor        ${params.postsel_pi0_floor} \\
    --postsel-pi0-ceil         ${params.postsel_pi0_ceil}
  "

  FLAGS=""
  [ "${params.use_mad}"     = "true" ] && FLAGS="\$FLAGS --use-mad"
  [ "${params.label_roll}"  = "true" ] && FLAGS="\$FLAGS --label-roll"
  [ "${params.run_pipeline}"= "true" ] && FLAGS="\$FLAGS --run-pipeline"

  if [ "\$LAUNCH" = "module" ]; then
    echo "[RUN] python -m driverformer ..."
    set -x; python -u -m driverformer \$COMMON_ARGS \$FLAGS; set +x
  elif [ -n "\${SCRIPT_PATH:-}" ] && [ -f "\$SCRIPT_PATH" ]; then
    echo "[RUN] python '\$SCRIPT_PATH' ..."
    set -x; python -u "\$SCRIPT_PATH" \$COMMON_ARGS \$FLAGS; set +x
  else
    echo "[ERROR] Neither 'driverformer' module nor 'trainDriverFormer.py' found in repo."
    exit 2
  fi

  echo "[DONE] DriverFormer finished."
  """
}

workflow {
  need('cls_file', params.cls_file)
  need('feat_file', params.feat_file)
  need('mutations_file', params.mutations_file)

  ch_cls  = Channel.fromPath(params.cls_file)
  ch_feat = Channel.fromPath(params.feat_file)
  ch_muts = Channel.fromPath(params.mutations_file)

  DRIVERFORMER_RUN(ch_cls, ch_feat, ch_muts)
}
