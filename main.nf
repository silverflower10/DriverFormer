// main.nf — DriverFormer (DSL2, CloudOS-ready, tuple inputs, no multiline quotes)
nextflow.enable.dsl = 2

// ── 필수 입력 ──
params.cls_file        = params.cls_file        ?: null
params.feat_file       = params.feat_file       ?: null
params.mutations_file  = params.mutations_file  ?: null
params.out_dir         = params.out_dir         ?: 'results/run'

// ── 하이퍼파라미터 기본값 ──
params.lr              = params.lr              ?: 2e-4
params.batch_size      = params.batch_size      ?: 32
params.epochs          = params.epochs          ?: 20
params.seed            = params.seed            ?: 42
params.d_model         = params.d_model         ?: 768
params.nhead           = params.nhead           ?: 8
params.num_layers      = params.num_layers      ?: 6
params.dim_feedforward = params.dim_feedforward ?: 3072
params.dropout         = params.dropout         ?: 0.2
params.max_seq_len     = params.max_seq_len     ?: 1024
params.segment_lengths = params.segment_lengths ?: ('10 50 100')
params.overlap_factor  = params.overlap_factor  ?: 0.3
params.use_mad         = (params.use_mad in [true,'true',1,'1'])
params.huber_factor    = params.huber_factor    ?: 3.0
params.cutmix_p        = params.cutmix_p        ?: 0.2
params.num_data_workers= params.num_data_workers?: 2
params.torch_threads   = params.torch_threads   ?: 2
params.len_alpha       = params.len_alpha       ?: 0.5
params.res_beta        = params.res_beta        ?: 0.5
params.label_roll      = (params.label_roll in [true,'true',1,'1'])
params.label_roll_width= params.label_roll_width?: 2

// ── 파이프라인/포스트선택 ──
params.run_pipeline           = (params.run_pipeline in [true,'true',1,'1'])
params.pipeline_out_dir       = params.pipeline_out_dir       ?: "${params.out_dir}/postproc_k_auto"
params.pipeline_chunk_size    = params.pipeline_chunk_size    ?: 1_000_000
params.pipeline_chunk_overlap = params.pipeline_chunk_overlap ?: 100_000
params.pipeline_min_distance  = params.pipeline_min_distance  ?: 0
params.pipeline_max_distance  = params.pipeline_max_distance  ?: 100_000
params.pipeline_sample_frac   = params.pipeline_sample_frac   ?: 0.01
params.pipeline_gmm_k         = params.pipeline_gmm_k         ?: 2
params.pipeline_beta          = params.pipeline_beta          ?: 1.0
params.pipeline_gamma         = params.pipeline_gamma         ?: 0.0
params.pipeline_seed          = params.pipeline_seed          ?: 42
params.pipeline_dp_gap_bp     = params.pipeline_dp_gap_bp     ?: 0
params.pipeline_presmooth_bins= params.pipeline_presmooth_bins?: 2

params.postsel_fdr_method     = params.postsel_fdr_method     ?: 'storey'
params.postsel_bootstrap      = params.postsel_bootstrap      ?: 400
params.postsel_lambda_start   = params.postsel_lambda_start   ?: 0.20
params.postsel_lambda_end     = params.postsel_lambda_end     ?: 0.95
params.postsel_lambda_step    = params.postsel_lambda_step    ?: 0.01
params.postsel_pi0_floor      = params.postsel_pi0_floor      ?: 0.01
params.postsel_pi0_ceil       = params.postsel_pi0_ceil       ?: 1.0

// ── 자원 기본 ──
params.cpus     = params.cpus     ?: 8
params.memory   = params.memory   ?: '64 GB'
params.time     = params.time     ?: '24h'

// (컨테이너는 nextflow.config의 cloudos 프로필에서 다이제스트로 고정됨)

workflow {
  // 프로필/컨테이너/필수 파라미터 진단(원인 추적용)
  log.info "active_profile = ${workflow.profile}"
  log.info "params.cls_file=${params.cls_file}"
  log.info "params.feat_file=${params.feat_file}"
  log.info "params.mutations_file=${params.mutations_file}"

  // 필수 3개 가드
  def missing = []
  if( !params.cls_file )        missing << 'cls_file'
  if( !params.feat_file )       missing << 'feat_file'
  if( !params.mutations_file )  missing << 'mutations_file'
  if( missing ) {
    log.error "Missing params: ${missing.join(', ')}"
    System.exit(1)
  }

  // 세 파일을 한 번에 넘기는 튜플 채널
  Channel.of( tuple( file(params.cls_file), file(params.feat_file), file(params.mutations_file) ) ) \
    | TRAIN_DRIVERFORMER
}
process TRAIN_DRIVERFORMER {
  tag "driverformer"

  // === 자원/컨테이너를 프로세스 안에서 '직접' 강제 ===
  cpus   (params.cpus   ?: 16)
  memory (params.memory ?: '128 GB')
  time   (params.time   ?: '24h')

  // 1) GPU 직접 요청 (config 없어도 적용)
  accelerator 1

  // 2) 컨테이너 직접 고정 (다이제스트)
  container 'docker.io/silverflower10/driverformer@sha256:ac15ea10f138b6f03552e0c59d804fac2903392623a5cfb92b6d7340564237c8'

  publishDir params.out_dir, mode: 'copy'

  // 환경변수(멀티라인 회피)
  env PYTHONPATH             : (System.getenv('PYTHONPATH') ?: '.') + ":$PWD"
  env OMP_NUM_THREADS        : params.torch_threads.toString()
  env MKL_NUM_THREADS        : params.torch_threads.toString()
  env OPENBLAS_NUM_THREADS   : params.torch_threads.toString()
  env NUMEXPR_NUM_THREADS    : params.torch_threads.toString()
  env MPLBACKEND             : 'Agg'
  env TOKENIZERS_PARALLELISM : 'false'

  input:
  tuple path cls_file, path feat_file, path mut_file

  output:
  path "${params.out_dir}"

  // (나머지 script: { ... } 부분은 네가 올린 그대로 유지)
  script:
  {
    def pre = [
      'bash','-lc',
      [
        'echo "[INFO] Host: $(hostname)  Date: $(date)"',
        'if command -v nvidia-smi >/dev/null 2>&1; then echo "[INFO] nvidia-smi:"; nvidia-smi || true; else echo "[WARN] nvidia-smi not found"; fi',
        'python - <<PY\n' +
        'import sys, torch\n' +
        'print("[PY] python =", sys.executable)\n' +
        'print("[PY] torch  =", torch.__version__)\n' +
        'print("[PY] cuda?  =", torch.cuda.is_available(), " nGPU =", torch.cuda.device_count())\n' +
        'print("[PY] dev0   =", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NA")\n' +
        'PY'
      ].join(' && ')
    ].join(' ')

    def args = [
      'python','-u','-m','driverformer',
      '--cls-file',       cls_file.toString(),
      '--feat-file',      feat_file.toString(),
      '--mutations-file', mut_file.toString(),
      '--out-dir',        params.out_dir.toString(),
      '--lr',             params.lr.toString(),
      '--batch-size',     params.batch_size.toString(),
      '--epochs',         params.epochs.toString(),
      '--seed',           params.seed.toString(),
      '--d-model',        params.d_model.toString(),
      '--nhead',          params.nhead.toString(),
      '--num-layers',     params.num_layers.toString(),
      '--dim-feedforward',params.dim_feedforward.toString(),
      '--dropout',        params.dropout.toString(),
      '--max-seq-len',    params.max_seq_len.toString(),
      '--overlap-factor', params.overlap_factor.toString(),
      '--huber-factor',   params.huber_factor.toString(),
      '--cutmix-p',       params.cutmix_p.toString(),
      '--num-data-workers', params.num_data_workers.toString(),
      '--torch-threads',    params.torch_threads.toString(),
      '--len-alpha',      params.len_alpha.toString(),
      '--res-beta',       params.res_beta.toString()
    ]

    if( params.use_mad ) { args << '--use-mad' }
    if( params.label_roll ) {
      args << '--label-roll'
      args << '--label-roll-width' << params.label_roll_width.toString()
    }

    def seg = params.segment_lengths?.toString()?.trim()
    if( seg ) { args << '--segment-lengths'; args.addAll( seg.split(/\s+/) as List ) }

    if( params.run_pipeline ) {
      args << '--run-pipeline'
      args.addAll([
        '--pipeline-out-dir',        params.pipeline_out_dir.toString(),
        '--pipeline-chunk-size',     params.pipeline_chunk_size.toString(),
        '--pipeline-chunk-overlap',  params.pipeline_chunk_overlap.toString(),
        '--pipeline-min-distance',   params.pipeline_min_distance.toString(),
        '--pipeline-max-distance',   params.pipeline_max_distance.toString(),
        '--pipeline-sample-frac',    params.pipeline_sample_frac.toString(),
        '--pipeline-gmm-k',          params.pipeline_gmm_k.toString(),
        '--pipeline-beta',           params.pipeline_beta.toString(),
        '--pipeline-gamma',          params.pipeline_gamma.toString(),
        '--pipeline-seed',           params.pipeline_seed.toString(),
        '--pipeline-dp-gap-bp',      params.pipeline_dp_gap_bp.toString(),
        '--pipeline-presmooth-bins', params.pipeline_presmooth_bins.toString()
      ])
    }

    args.addAll([
      '--postsel-fdr-method',   params.postsel_fdr_method.toString(),
      '--postsel-bootstrap',    params.postsel_bootstrap.toString(),
      '--postsel-lambda-start', params.postsel_lambda_start.toString(),
      '--postsel-lambda-end',   params.postsel_lambda_end.toString(),
      '--postsel-lambda-step',  params.postsel_lambda_step.toString(),
      '--postsel-pi0-floor',    params.postsel_pi0_floor.toString(),
      '--postsel-pi0-ceil',     params.postsel_pi0_ceil.toString()
    ])

    [ pre, args.join(' ') ].join(' && ')
  }
}
