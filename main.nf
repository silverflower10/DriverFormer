nextflow.enable.dsl = 2

// ==============================
// Parameters
// ==============================
params.pipeline_only      = params.pipeline_only      ?: false
params.mutations_file     = params.mutations_file     ?: null
params.all_pred           = params.all_pred           ?: null

params.out_dir            = params.out_dir            ?: "results/run"
params.post_dir           = params.post_dir           ?: "${params.out_dir}/postproc"

params.lr                 = params.lr                 ?: 2e-4
params.batch_size         = params.batch_size         ?: 128
params.epochs             = params.epochs             ?: 20
params.segment_lengths    = params.segment_lengths    ?: [10,50,100]
params.label_roll         = params.label_roll         ?: true
params.label_roll_width   = params.label_roll_width   ?: 2
params.save_attention     = params.save_attention     ?: false

params.d_model            = params.d_model            ?: 768
params.nhead              = params.nhead              ?: 8
params.num_layers         = params.num_layers         ?: 6
params.dim_feedforward    = params.dim_feedforward    ?: 3072
params.dropout            = params.dropout            ?: 0.2
params.max_seq_len        = params.max_seq_len        ?: 1024
params.overlap_factor     = params.overlap_factor     ?: 0.3
params.use_mad            = params.use_mad            ?: true
params.huber_factor       = params.huber_factor       ?: 3.0
params.cutmix_p           = params.cutmix_p           ?: 0.2
params.num_data_workers   = params.num_data_workers   ?: 8
params.torch_threads      = params.torch_threads      ?: 8
params.len_alpha          = params.len_alpha          ?: 0.5
params.res_beta           = params.res_beta           ?: 0.5

// pipeline (LLR→GMM→DP)
params.run_pipeline       = params.run_pipeline       ?: true
params.pipeline_gmm_auto  = params.pipeline_gmm_auto  ?: false
params.pipeline_gmm_k     = params.pipeline_gmm_k     ?: 2
params.pipe_beta          = params.pipe_beta          ?: 1.0
params.pipe_gamma         = params.pipe_gamma         ?: 0.0
params.pipeline_dp_gap_bp = params.pipeline_dp_gap_bp ?: 0
params.pipeline_chunk_size    = params.pipeline_chunk_size    ?: 1_000_000
params.pipeline_chunk_overlap = params.pipeline_chunk_overlap ?: 100_000
params.pipeline_min_distance  = params.pipeline_min_distance  ?: 0
params.pipeline_max_distance  = params.pipeline_max_distance  ?: 100_000
params.pipeline_sample_frac   = params.pipeline_sample_frac   ?: 0.01
params.pipeline_presmooth_bins= params.pipeline_presmooth_bins?: 2

// post-selection
params.postsel_fdr_method = params.postsel_fdr_method ?: 'storey'
params.postsel_bootstrap  = params.postsel_bootstrap  ?: 400
params.postsel_lambda_start = params.postsel_lambda_start ?: 0.20
params.postsel_lambda_end   = params.postsel_lambda_end   ?: 0.95
params.postsel_lambda_step  = params.postsel_lambda_step  ?: 0.01
params.postsel_pi0_floor    = params.postsel_pi0_floor    ?: 0.01
params.postsel_pi0_ceil     = params.postsel_pi0_ceil     ?: 1.0

// Release download options
params.use_release  = params.use_release ?: false
params.gh_repo      = params.gh_repo     ?: null
params.release_tag  = params.release_tag ?: null
params.asset_name   = params.asset_name  ?: 'parts'   // 'parts' or tar.gz name
params.gh_token     = params.gh_token    ?: null

// ==============================
// Helpers (build CLI args)
// ==============================
def asList(v) { (v instanceof List) ? v : v.toString().trim().split(/\s+/)*.toInteger() }

def trainArgs() {
  def segs = asList(params.segment_lengths)
  def a = []
  a += "--out-dir ${params.out_dir}"
  a += "--segment-lengths ${segs.join(' ')}"
  a += "--batch-size ${params.batch_size} --epochs ${params.epochs} --lr ${params.lr}"
  if (params.label_roll) a += "--label-roll"
  a += "--label-roll-width ${params.label_roll_width}"
  if (params.save_attention) a += "--save-attention"
  a += "--d-model ${params.d_model} --nhead ${params.nhead} --num-layers ${params.num_layers}"
  a += "--dim-feedforward ${params.dim_feedforward} --dropout ${params.dropout} --max-seq-len ${params.max_seq_len}"
  a += "--overlap-factor ${params.overlap_factor} --huber-factor ${params.huber_factor}"
  if (params.use_mad) a += "--use-mad"
  a += "--cutmix-p ${params.cutmix_p} --num-data-workers ${params.num_data_workers} --torch-threads ${params.torch_threads}"
  a += "--len-alpha ${params.len_alpha} --res-beta ${params.res_beta}"
  if (params.run_pipeline) a += "--run-pipeline"
  // pipeline options
  if (params.pipeline_gmm_auto) a += "--pipeline-gmm-auto"
  else a += "--pipeline-gmm-k ${params.pipeline_gmm_k}"
  a += "--pipeline-beta ${params.pipe_beta} --pipeline-gamma ${params.pipe_gamma}"
  a += "--pipeline-dp-gap-bp ${params.pipeline_dp_gap_bp}"
  a += "--pipeline-chunk-size ${params.pipeline_chunk_size}"
  a += "--pipeline-chunk-overlap ${params.pipeline_chunk_overlap}"
  a += "--pipeline-min-distance ${params.pipeline_min_distance}"
  a += "--pipeline-max-distance ${params.pipeline_max_distance}"
  a += "--pipeline-sample-frac ${params.pipeline_sample_frac}"
  a += "--pipeline-presmooth-bins ${params.pipeline_presmooth_bins}"
  a += "--postsel-fdr-method ${params.postsel_fdr_method}"
  a += "--postsel-bootstrap ${params.postsel_bootstrap}"
  a += "--postsel-lambda-start ${params.postsel_lambda_start}"
  a += "--postsel-lambda-end ${params.postsel_lambda_end}"
  a += "--postsel-lambda-step ${params.postsel_lambda_step}"
  a += "--postsel-pi0-floor ${params.postsel_pi0_floor}"
  a += "--postsel-pi0-ceil ${params.postsel_pi0_ceil}"
  a += "--pipeline-out-dir ${params.post_dir}"
  return a.join(' ')
}

def pipeArgs() {
  def b = []
  b += "--pipeline-only"
  b += "--pipeline-out-dir ${params.post_dir}"
  if (params.pipeline_gmm_auto) b += "--pipeline-gmm-auto"
  else b += "--pipeline-gmm-k ${params.pipeline_gmm_k}"
  b += "--pipeline-beta ${params.pipe_beta} --pipeline-gamma ${params.pipe_gamma}"
  b += "--pipeline-dp-gap-bp ${params.pipeline_dp_gap_bp}"
  b += "--pipeline-chunk-size ${params.pipeline_chunk_size}"
  b += "--pipeline-chunk-overlap ${params.pipeline_chunk_overlap}"
  b += "--pipeline-min-distance ${params.pipeline_min_distance}"
  b += "--pipeline-max-distance ${params.pipeline_max_distance}"
  b += "--pipeline-sample-frac ${params.pipeline_sample_frac}"
  b += "--pipeline-presmooth-bins ${params.pipeline_presmooth_bins}"
  b += "--postsel-fdr-method ${params.postsel_fdr_method}"
  b += "--postsel-bootstrap ${params.postsel_bootstrap}"
  b += "--postsel-lambda-start ${params.postsel_lambda_start}"
  b += "--postsel-lambda-end ${params.postsel_lambda_end}"
  b += "--postsel-lambda-step ${params.postsel_lambda_step}"
  b += "--postsel-pi0-floor ${params.postsel_pi0_floor}"
  b += "--postsel-pi0-ceil ${params.postsel_pi0_ceil}"
  return b.join(' ')
}

// ==============================
// Inputs
// ==============================
if (!params.pipeline_only) {
  if (!params.mutations_file) exit 1, "ERROR: --mutations_file is required (training mode)"
}
MUT_FILE = params.pipeline_only ? Channel.empty()
                                : Channel.fromPath(params.mutations_file, checkIfExists: true)

if (params.pipeline_only && !params.all_pred)
  exit 1, "ERROR: --all_pred is required (pipeline-only mode)"
ALL_PRED = params.pipeline_only ? Channel.fromPath(params.all_pred, checkIfExists: true)
                                : Channel.empty()

// ==============================
// Processes
// ==============================

// Download PKLs from GitHub Release (parts or tar.gz)
process DOWNLOAD_BREAST_RELEASE {
  tag "download:${params.release_tag ?: 'NA'}"
  cpus 1; memory '3 GB'; time '3h'
  output: path "cls_embedding.pkl", emit: cls
          path "feature_dict_BRCA.pkl", emit: feat
  when: params.use_release
  script:
  if (!params.gh_repo || !params.release_tag) {
    throw new RuntimeException("use_release=true requires --gh_repo and --release_tag")
  }
  def baseApi = "https://api.github.com/repos/${params.gh_repo}/releases/tags/${params.release_tag}"
  def asset   = params.asset_name
  """
  set -euo pipefail
  AUTH=${params.gh_token ? 1 : 0}
  HDR=""
  [ "\$AUTH" -eq 1 ] && HDR="-H 'Authorization: Bearer ${params.gh_token}'" || true

  if echo '${asset}' | grep -qE '\\.tar\\.gz\$'; then
    URL="https://github.com/${params.gh_repo}/releases/download/${params.release_tag}/${asset}"
    echo "[DL] \$URL"
    if [ "\$AUTH" -eq 1 ]; then eval "curl -fL \$HDR -o '${asset}' '\$URL'"; else curl -fL -o '${asset}' "\$URL"; fi
    tar -xzf '${asset}'
    test -f cls_embedding.pkl && test -f feature_dict_BRCA.pkl
    exit 0
  fi

  # parts mode
  echo "[API] ${baseApi}"
  if [ "\$AUTH" -eq 1 ]; then eval "curl -s \$HDR '${baseApi}' > rel.json"; else curl -s '${baseApi}' > rel.json; fi
  python - <<'PY'
import json, subprocess
rel=json.load(open('rel.json'))
urls=[a['browser_download_url'] for a in rel.get('assets',[])]
parts=sorted([u for u in urls if 'cls_embedding.pkl.part_' in u])+\
      sorted([u for u in urls if 'feature_dict_BRCA.pkl.part_' in u])
hdr=[]
# GH_TOKEN presence handled in shell, not needed here
for u in parts:
    fn=u.split('/')[-1]
    print("[DL]", fn)
    subprocess.check_call(['bash','-lc', 'curl --http1.1 -fL -o '+fn+' '+u])
PY
  cat \$(printf "%s\\n" cls_embedding.pkl.part_* | LC_ALL=C sort) > cls_embedding.pkl
  cat \$(printf "%s\\n" feature_dict_BRCA.pkl.part_* | LC_ALL=C sort) > feature_dict_BRCA.pkl
  ls -lh cls_embedding.pkl feature_dict_BRCA.pkl
  """
}

// Use repo-local PKLs (or assemble from parts)
process LOCAL_BREAST {
  tag "local"
  cpus 1; memory '1 GB'; time '1h'
  output: path "cls_embedding.pkl", emit: cls
          path "feature_dict_BRCA.pkl", emit: feat
  when: !params.use_release
  script:
  """
  set -euo pipefail
  BREAST="\${projectDir}/data/breast"
  if compgen -G "\${BREAST}/cls_embedding.pkl.part_*" > /dev/null; then
    cat \$(printf "%s\\n" \${BREAST}/cls_embedding.pkl.part_* | LC_ALL=C sort) > cls_embedding.pkl
  else
    ln -s "\${BREAST}/cls_embedding.pkl" cls_embedding.pkl
  fi
  if compgen -G "\${BREAST}/feature_dict_BRCA.pkl.part_*" > /dev/null; then
    cat \$(printf "%s\\n" \${BREAST}/feature_dict_BRCA.pkl.part_* | LC_ALL=C sort) > feature_dict_BRCA.pkl
  else
    ln -s "\${BREAST}/feature_dict_BRCA.pkl" feature_dict_BRCA.pkl
  fi
  """
}

// Train + (optionally) run pipeline
process DRIVERFORMER_TRAIN {
  tag "train"
  cpus 8; memory '32 GB'; time '48h'
  publishDir params.out_dir, mode: 'copy', overwrite: true
  input:
    path cls_pkl
    path feat_pkl
    path mut_file
  script:
  def A = trainArgs()
  """
  set -euo pipefail
  REPO="\${projectDir}"
  if python - <<'PY' >/dev/null 2>&1; then
import importlib; importlib.import_module('driverformer'); print('ok')
PY
    then echo "[INFO] driverformer installed"
  else
    pip install -e "\${REPO}" --no-deps || true
  fi

  if command -v driverformer >/dev/null 2>&1; then
    driverformer --cls-file "\${cls_pkl}" --feat-file "\${feat_pkl}" --mutations-file "\${mut_file}" ${A}
  else
    python -m driverformer.cli --cls-file "\${cls_pkl}" --feat-file "\${feat_pkl}" --mutations-file "\${mut_file}" ${A}
  fi
  """
}

// Pipeline-only
process DRIVERFORMER_PIPE {
  tag "pipe"
  cpus 4; memory '16 GB'; time '12h'
  publishDir params.post_dir, mode: 'copy', overwrite: true
  input: path all_pred
  script:
  def B = pipeArgs()
  """
  set -euo pipefail
  REPO="\${projectDir}"
  if ! python - <<'PY' >/dev/null 2>&1; then
import importlib; importlib.import_module('driverformer'); print('ok')
PY
  then pip install -e "\${REPO}" --no-deps || true; fi
  if command -v driverformer >/dev/null 2>&1; then
    driverformer --all-pred "\${all_pred}" ${B}
  else
    python -m driverformer.cli --all-pred "\${all_pred}" ${B}
  fi
  """
}

// ==============================
// Workflow
// ==============================
workflow {

  if (params.pipeline_only) {
    DRIVERFORMER_PIPE( ALL_PRED )
    return
  }

  // prepare CLS/FEAT channels (either release or local)
  def out_chs = params.use_release ? DOWNLOAD_BREAST_RELEASE() : LOCAL_BREAST()
  def CLS_CH  = out_chs.cls
  def FEAT_CH = out_chs.feat

  // run training with three inputs (cls, feat, mut)
  DRIVERFORMER_TRAIN( CLS_CH, FEAT_CH, MUT_FILE )
}
