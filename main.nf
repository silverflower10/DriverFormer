// main.nf — install (venv via requirements/wheels) + run; no triple quotes; no optional(true)
nextflow.enable.dsl = 2

// ---- Params ----
params.cls_file        = params.cls_file        ?: null
params.feat_file       = params.feat_file       ?: null
params.mutations_file  = params.mutations_file  ?: null
params.out_dir         = params.out_dir         ?: 'results/run'

// 설치 단계 on/off (기본: true)
params.setup_deps      = (params.setup_deps in [false,'false',0,'0']) ? false : true

// 학습/파이프라인 주요 파라미터
params.lr              = (params.lr ?: 2e-4)
params.batch_size      = (params.batch_size ?: 128)
params.epochs          = (params.epochs ?: 20)
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

// ===== 1) 설치(venv) =====
process SETUP_DEPS {
  cpus 1; memory '4 GB'; time '2h'
  publishDir "${params.out_dir}", mode: 'copy', overwrite: true

  input:
    path REQS        // requirements.txt 또는 더미
    path WHEELS_DIR  // wheels/ 디렉토리 또는 더미

  output:
    path "venv", emit: VENV

  // 한 줄 script (triple quotes 없음)
  script:
    "python -m venv venv && " +
    ". venv/bin/activate && " +
    "python -m pip install -U pip wheel setuptools --no-cache-dir && " +
    // requirements.txt 있으면 설치
    "[ -f '!{REQS}' ] && [ \"$(basename '!{REQS}')\" = 'requirements.txt' ] && python -m pip install -r '!{REQS}' --no-cache-dir || true && " +
    // wheels 폴더 있으면 오프라인 설치
    "[ -d '!{WHEELS_DIR}' ] && python -m pip install --no-index --find-links '!{WHEELS_DIR}' '!{WHEELS_DIR}'/*.whl || true"
}

// ===== 2) 실행 =====
process DRIVERFORMER_RUN {
  cpus 8; memory '32 GB'; time '72h'
  publishDir "${params.out_dir}", mode: 'copy', overwrite: true

  input:
    path CLS
    path FEAT
    path MUTS
    path VENV   // venv 또는 더미(항상 전달)

  output:
    path "stdout.txt"
    path "stderr.txt"

  // script 블록: Groovy로 한 줄 명령 구성(삼중따옴표/!{} 없음)
  script:
  {
    // segment_lengths 정규화 (리스트/문자열 → 공백/숫자만)
    def segRaw  = params.segment_lengths ? (params.segment_lengths instanceof List ? params.segment_lengths.join(' ') : params.segment_lengths.toString()) : ''
    def segNorm = segRaw.replace(',', ' ').trim().replaceAll(/\s+/, ' ').replaceAll(/^\"|\"$/, '')
    def segNums = segNorm ? segNorm.split(/\s+/).findAll{ it ==~ /\d+/ }.join(' ') : ''
    def segOpt  = segNums ? "--segment-lengths ${segNums} " : ""

    // 플래그
    def flags = ''
    if (params.use_mad)      flags += '--use-mad '
    if (params.label_roll)   flags += '--label-roll '
    if (params.run_pipeline) flags += '--run-pipeline '

    // venv 활성화(디렉토리인 경우에만)
    def venvAct = "[ -d '!{VENV}' ] && . '!{VENV}/bin/activate' || true; "

    // 실행 커맨드
    def cmd =
      venvAct +
      "python -u -m driverformer " +
      "--cls-file ${CLS} --feat-file ${FEAT} --mutations-file ${MUTS} " +
      "--out-dir ${params.out_dir} " +
      "--lr ${params.lr} --batch-size ${params.batch_size} --epochs ${params.epochs} --seed ${params.seed} " +
      "--d-model ${params.d_model} --nhead ${params.nhead} --num-layers ${params.num_layers} " +
      "--dim-feedforward ${params.dim_feedforward} --dropout ${params.dropout} --max-seq-len ${params.max_seq_len} " +
      "--overlap-factor ${params.overlap_factor} --huber-factor ${params.huber_factor} --cutmix-p ${params.cutmix_p} " +
      "--num-data-workers ${params.num_data_workers} --torch-threads ${params.torch_threads} " +
      "--len-alpha ${params.len_alpha} --res-beta ${params.res_beta} --label-roll-width ${params.label_roll_width} " +
      "--pipeline-out-dir ${params.pipeline_out_dir} --pipeline-chunk-size ${params.pipeline_chunk_size} " +
      "--pipeline-chunk-overlap ${params.pipeline_chunk_overlap} --pipeline-min-distance ${params.pipeline_min_distance} " +
      "--pipeline-max-distance ${params.pipeline_max_distance} --pipeline-sample-frac ${params.pipeline_sample_frac} " +
      "--pipeline-gmm-k ${params.pipeline_gmm_k} --pipeline-beta ${params.pipeline_beta} --pipeline-gamma ${params.pipeline_gamma} " +
      "--pipeline-seed ${params.pipeline_seed} --pipeline-dp-gap-bp ${params.pipeline_dp_gap_bp} " +
      "--postsel-fdr-method ${params.postsel_fdr_method} --postsel-bootstrap ${params.postsel_bootstrap} " +
      "--postsel-lambda-start ${params.postsel_lambda_start} --postsel-lambda-end ${params.postsel_lambda_end} " +
      "--postsel-lambda-step ${params.postsel_lambda_step} --postsel-pi0-floor ${params.postsel_pi0_floor} --postsel-pi0-ceil ${params.postsel_pi0_ceil} " +
      segOpt + flags +
      "1>stdout.txt 2>stderr.txt"

    return cmd
  }
}

// ===== 워크플로 =====
workflow {
  if( !params.cls_file )       error "Missing required param: --cls_file"
  if( !params.feat_file )      error "Missing required param: --feat_file"
  if( !params.mutations_file ) error "Missing required param: --mutations_file"

  // 데이터
  ch_cls  = Channel.fromPath(params.cls_file)
  ch_feat = Channel.fromPath(params.feat_file)
  ch_muts = Channel.fromPath(params.mutations_file)

  // requirements & wheels 채널 (없으면 더미)
  ch_reqs_exist = Channel.fromPath("${projectDir}/requirements.txt", checkIfExists: true)
  ch_dummy      = Channel.fromPath("${projectDir}/driverformer/__init__.py", checkIfExists: true)
  ch_reqs   = ch_reqs_exist.ifEmpty(ch_dummy)

  ch_wheels_exist = Channel.fromPath("${projectDir}/wheels", checkIfExists: true)
  ch_wheels = ch_wheels_exist.ifEmpty(ch_dummy)

  if (params.setup_deps) {
    SETUP_DEPS(ch_reqs, ch_wheels)
    DRIVERFORMER_RUN(ch_cls, ch_feat, ch_muts, SETUP_DEPS.out.VENV)
  } else {
    // 설치 off: 더미를 VENV 자리에 넣어 입력 항상 만족 (optional 사용 안 함)
    DRIVERFORMER_RUN(ch_cls, ch_feat, ch_muts, ch_dummy)
  }
}
