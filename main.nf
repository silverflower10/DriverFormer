// main.nf — call external runner (no ''' / """ inside)
nextflow.enable.dsl = 2

params.cls_file        = params.cls_file        ?: null
params.feat_file       = params.feat_file       ?: null
params.mutations_file  = params.mutations_file  ?: null
params.out_dir         = params.out_dir         ?: 'results/run'

process DRIVERFORMER_RUN {
  cpus 8; memory '32 GB'; time '72h'
  publishDir "${params.out_dir}", mode: 'copy', overwrite: true

  input:
    path CLS; path FEAT; path MUTS; path DF_PKG; path TRAIN_PY; path REQS; path WHEELS_DIR

  output:
    path "stdout.txt"; path "stderr.txt"

  script:
    "NXF_CLS='!{CLS}' NXF_FEAT='!{FEAT}' NXF_MUTS='!{MUTS}' " +
    "NXF_REQS='!{REQS}' NXF_WHEELS='!{WHEELS_DIR}' NXF_OUT_DIR='!{params.out_dir}' " +
    "NXF_LR='!{params.lr}' NXF_BATCH='!{params.batch_size}' NXF_EPOCHS='!{params.epochs}' NXF_SEED='!{params.seed}' " +
    "NXF_DMODEL='!{params.d_model}' NXF_NHEAD='!{params.nhead}' NXF_NLAYERS='!{params.num_layers}' NXF_DFF='!{params.dim_feedforward}' " +
    "NXF_DROPOUT='!{params.dropout}' NXF_MAXLEN='!{params.max_seq_len}' " +
    "NXF_SEGLEN='!{ (params.segment_lengths instanceof List) ? params.segment_lengths.join(\" \") : (params.segment_lengths ? params.segment_lengths.toString() : \"\") }' " +
    "NXF_OVERLAP='!{params.overlap_factor}' NXF_USE_MAD='!{params.use_mad}' NXF_HUBER='!{params.huber_factor}' NXF_CUTMIX='!{params.cutmix_p}' " +
    "NXF_WORKERS='!{params.num_data_workers}' NXF_TTHREADS='!{params.torch_threads}' NXF_LEN_ALPHA='!{params.len_alpha}' NXF_RES_BETA='!{params.res_beta}' " +
    "NXF_LABEL_ROLL='!{params.label_roll}' NXF_LABEL_W='!{params.label_roll_width}' " +
    "NXF_RUN_PIPE='!{params.run_pipeline}' NXF_P_OUT='!{params.pipeline_out_dir}' NXF_P_CHUNK='!{params.pipeline_chunk_size}' " +
    "NXF_P_OVERLAP='!{params.pipeline_chunk_overlap}' NXF_P_MINDIST='!{params.pipeline_min_distance}' NXF_P_MAXDIST='!{params.pipeline_max_distance}' " +
    "NXF_P_FRAC='!{params.pipeline_sample_frac}' NXF_P_GMMK='!{params.pipeline_gmm_k}' NXF_P_BETA='!{params.pipeline_beta}' NXF_P_GAMMA='!{params.pipeline_gamma}' " +
    "NXF_P_SEED='!{params.pipeline_seed}' NXF_P_GAP='!{params.pipeline_dp_gap_bp}' " +
    "NXF_FDR_METHOD='!{params.postsel_fdr_method}' NXF_BOOT='!{params.postsel_bootstrap}' NXF_LAMBDA_S='!{params.postsel_lambda_start}' " +
    "NXF_LAMBDA_E='!{params.postsel_lambda_end}' NXF_LAMBDA_T='!{params.postsel_lambda_step}' NXF_PI0_F='!{params.postsel_pi0_floor}' NXF_PI0_C='!{params.postsel_pi0_ceil}' " +
    "bash ${projectDir}/scripts/driverformer_run.sh"
}

workflow {
  if( !params.cls_file )       error "Missing required param: --cls_file"
  if( !params.feat_file )      error "Missing required param: --feat_file"
  if( !params.mutations_file ) error "Missing required param: --mutations_file"

  ch_cls = Channel.fromPath(params.cls_file)
  ch_feat= Channel.fromPath(params.feat_file)
  ch_muts= Channel.fromPath(params.mutations_file)

  ch_pkg   = Channel.fromPath("${projectDir}/driverformer",           checkIfExists: true)
  ch_train = Channel.fromPath("${projectDir}/trainDriverFormer.py",   checkIfExists: true)

  ch_reqs_exist = Channel.fromPath("${projectDir}/requirements.txt",  checkIfExists: true)
  ch_dummy      = Channel.fromPath("${projectDir}/driverformer/__init__.py", checkIfExists: true)
  ch_reqs  = ch_reqs_exist.ifEmpty(ch_dummy)

  ch_wheels_exist = Channel.fromPath("${projectDir}/wheels",          checkIfExists: true)
  ch_wheels = ch_wheels_exist.ifEmpty(ch_dummy)

  DRIVERFORMER_RUN(ch_cls, ch_feat, ch_muts, ch_pkg, ch_train, ch_reqs, ch_wheels)
}
