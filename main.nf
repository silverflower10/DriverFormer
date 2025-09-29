// main.nf — DriverFormer minimal DSL2 workflow (no package install; run repo script directly)
nextflow.enable.dsl = 2

// -------------------------
// Params (기본값은 로컬 실행 예시와 동일하게 맞춤)
// -------------------------
params.cls_file              = null
params.feat_file             = null
params.mutations_file        = null

params.out_dir               = 'results/run'
params.post_dir              = "${params.out_dir}/postproc_k_auto"

params.lr                    = 2e-4
params.batch_size            = 128
params.epochs                = 2
params.seed                  = 42
params.d_model               = 768
params.nhead                 = 8
params.num_layers            = 6
params.dim_feedforward       = 3072
params.dropout               = 0.2
params.max_seq_len           = 1024
params.segment_lengths       = [10,50,100]
params.overlap_factor        = 0.3
params.use_mad               = true
params.huber_factor          = 3.0
params.cutmix_p              = 0.2
params.num_data_workers      = 8
params.torch_threads         = 8
params.len_alpha             = 0.5
params.res_beta              = 0.5
params.label_roll            = true
params.label_roll_width      = 2

// pipeline
params.run_pipeline          = true
params.pipeline_out_dir      = "${params.post_dir}"
params.pipeline_chunk_size   = 1_000_000
params.pipeline_chunk_overlap= 100_000
params.pipeline_min_distance = 0
params.pipeline_max_distance = 100_000
params.pipeline_sample_frac  = 0.01
params.pipeline_gmm_k        = 2
params.pipeline_beta         = 1.0
params.pipeline_gamma        = 0.0
params.pipeline_seed         = 42
params.pipeline_dp_gap_bp    = 0
params.pipeline_presmooth_bins = 2

// post-selection
params.postsel_fdr_method    = 'storey'
params.postsel_bootstrap     = 400
params.postsel_lambda_start  = 0.20
params.postsel_lambda_end    = 0.95
params.postsel_lambda_step   = 0.01
params.postsel_pi0_floor     = 0.01
params.postsel_pi0_ceil      = 1.0

// -------------------------
// Helpers
// -------------------------
def checkRequired(name, val) {
  if( !val ) error "Missing required param: --${name}"
}

// -------------------------
// Process
// -------------------------
process DRIVERFORMER_TRAIN {
  tag "driverformer-train"

  cpus 8
  memory '32 GB'
  time '72h'

  publishDir "${params.out_dir}", mode: 'copy', overwrite: true

  input:
    path CLS  from file(params.cls_file)
    path FEAT from file(params.feat_file)
    path MUTS from file(params.mutations_file)

  output:
    path "stdout.txt"
    path "stderr.txt"
    path "${params.out_dir}", emit: OUTDIR, optional: true
    path "${params.post_dir}", emit: POSTDIR, optional: true

  /*
    - 패키지 설치 없이 리포 소스 직접 실행
    - trainDriverFormer.py 는 리포 루트에 있다고 가정(projectDir)
    - 필요시 파일명/경로만 바꾸면 됨
  */
  script:
  """
  set -euo pipefail
  exec > >(tee stdout.txt) 2> >(tee stderr.txt >&2)

  echo "[INFO] projectDir = ${projectDir}"
  echo "[INFO] workdir    = \$(pwd)"

  export MPLBACKEND=Agg
  export TOKENIZERS_PARALLELISM=false
  export OMP_NUM_THREADS=${task.cpus}
  export MKL_NUM_THREADS=${task.cpus}
  export OPENBLAS_NUM_THREADS=${task.cpus}
  export NUMEXPR_NUM_THREADS=${task.cpus}

  # GPU/환경 정보
  echo "===== ENV CHECK ====="
  nvidia-smi || true
  which python || true
  python - <<'PY'
import torch, sys, os
print("python =", sys.executable)
print("torch  =", torch.__version__,
      "| cuda?", torch.cuda.is_available(),
      "| #GPU =", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU 0 =", torch.cuda.get_device_name(0))
print("CUDA (torch) =", torch.version.cuda)
PY
  echo "====================="

  # 리포 소스 직접 참조
  export PYTHONPATH="${projectDir}:\$PYTHONPATH"

  # === 실행 ===
  python -u "${projectDir}/trainDriverFormer.py" \\
    --cls-file            "${CLS}" \\
    --feat-file           "${FEAT}" \\
    --mutations-file      "${MUTS}" \\
    --out-dir             "${params.out_dir}" \\
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
    ${ params.use_mad           ? '--use-mad'           : '' } \\
    --huber-factor        ${params.huber_factor} \\
    --cutmix-p            ${params.cutmix_p} \\
    --num-data-workers    ${params.num_data_workers} \\
    --torch-threads       ${params.torch_threads} \\
    --len-alpha           ${params.len_alpha} \\
    --res-beta            ${params.res_beta} \\
    ${ params.label_roll        ? '--label-roll'        : '' } \\
    --label-roll-width    ${params.label_roll_width} \\
    ${ params.run_pipeline      ? '--run-pipeline'      : '' } \\
    --pipeline-out-dir         "${params.pipeline_out_dir}" \\
    --pipeline-chunk-size      ${params.pipeline_chunk_size} \\
    --pipeline-chunk-overlap   ${params.pipeline_chunk_overlap} \\
    --pipeline-min-distance    ${params.pipeline_min_distance} \\
    --pipeline-max-distance    ${params.pipeline_max_distance} \\
    --pipeline-sample-frac     ${params.pipeline_sample_frac} \\
    --pipeline-gmm-k           ${params.pipeline_gmm_k} \\
    --pipeline-beta            ${params.pipeline_beta} \\
    --pipeline-gamma           ${params.pipeline_gamma} \\
    --pipeline-seed            ${params.pipeline_seed} \\
    --pipeline-dp-gap-bp       ${params.pipeline_dp_gap_bp} \\
    --pipeline-presmooth-bins  ${params.pipeline_presmooth_bins} \\
    --postsel-fdr-method       ${params.postsel_fdr_method} \\
    --postsel-bootstrap        ${params.postsel_bootstrap} \\
    --postsel-lambda-start     ${params.postsel_lambda_start} \\
    --postsel-lambda-end       ${params.postsel_lambda_end} \\
    --postsel-lambda-step      ${params.postsel_lambda_step} \\
    --postsel-pi0-floor        ${params.postsel_pi0_floor} \\
    --postsel-pi0-ceil         ${params.postsel_pi0_ceil}

  echo "[DONE] DriverFormer finished."
  """
}

// -------------------------
// Workflow
// -------------------------
workflow {
  checkRequired('cls_file', params.cls_file)
  checkRequired('feat_file', params.feat_file)
  checkRequired('mutations_file', params.mutations_file)

  DRIVERFORMER_TRAIN()
}
