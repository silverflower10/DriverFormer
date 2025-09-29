nextflow.enable.dsl = 2

// ==============================
// Parameters
// ==============================
params.pipeline_only     = params.pipeline_only     ?: false
params.mutations_file    = params.mutations_file    ?: null                // required if !pipeline_only
params.all_pred          = params.all_pred          ?: null                // required if  pipeline_only

params.out_dir           = params.out_dir           ?: "results/run"
params.post_dir          = params.post_dir          ?: "${params.out_dir}/postproc"

params.lr                = params.lr                ?: 2e-4
params.batch_size        = params.batch_size        ?: 128
params.epochs            = params.epochs            ?: 20
params.segment_lengths   = params.segment_lengths   ?: [10,50,100]
params.label_roll        = params.label_roll        ?: true
params.label_roll_width  = params.label_roll_width  ?: 2
params.save_attention    = params.save_attention    ?: false

// ---- pipeline knobs
params.pipeline_gmm_auto = params.pipeline_gmm_auto ?: true
params.pipeline_gmm_k    = params.pipeline_gmm_k    ?: null
params.pipe_beta         = params.pipe_beta         ?: 1.0
params.pipe_gamma        = params.pipe_gamma        ?: 0.0
params.pipeline_dp_gap_bp= params.pipeline_dp_gap_bp?: 0

// ==============================
// Release download options
// ==============================
// 사용: --use_release --gh_repo "silverflower10/DriverFormer" --release_tag "breast-data-v1"
//      --asset_name "breast_data.tar.gz" [--asset_sha256 "<hex>"] [--gh_token "xxxxx"]
params.use_release  = params.use_release ?: false
params.gh_repo      = params.gh_repo     ?: null           // e.g., "silverflower10/DriverFormer"
params.release_tag  = params.release_tag ?: null           // e.g., "breast-data-v1"
params.asset_name   = params.asset_name  ?: "breast_data.tar.gz"
params.asset_sha256 = params.asset_sha256?: null           // (선택) 체크섬 검증
params.gh_token     = params.gh_token    ?: null           // Private일 때만 필요

// repo 내부 고정 경로(백업 플랜)
def LOCAL_CLS  = "${projectDir}/data/breast/cls_embedding.pkl"
def LOCAL_FEAT = "${projectDir}/data/breast/feature_dict_BRCA.pkl"

// ==============================
// Helpers
// ==============================
def asList(v) {
  (v instanceof List) ? v : v.toString().trim().split(/\s+/)*.toInteger()
}

def trainArgs() {
  def segs = asList(params.segment_lengths)
  def a = []
  a += "--out-dir ${params.out_dir}"
  a += "--segment-lengths ${segs.join(' ')}"
  a += "--batch-size ${params.batch_size} --epochs ${params.epochs} --lr ${params.lr}"
  if (params.label_roll) a += "--label-roll"
  a += "--label-roll-width ${params.label_roll_width}"
  if (params.save_attention) a += "--save-attention"
  return a.join(' ')
}

def pipeArgs() {
  def b = []
  b += "--pipeline-only"
  b += "--pipeline-out-dir ${params.post_dir}"
  if (params.pipeline_gmm_auto) b += "--pipeline-gmm-auto"
  if (!params.pipeline_gmm_auto && params.pipeline_gmm_k)
    b += "--pipeline-gmm-k ${params.pipeline_gmm_k}"
  b += "--pipeline-beta ${params.pipe_beta} --pipeline-gamma ${params.pipe_gamma}"
  b += "--pipeline-dp-gap-bp ${params.pipeline_dp_gap_bp}"
  return b.join(' ')
}

// ==============================
// Inputs
// ==============================
if (!params.pipeline_only) {
  if (!params.mutations_file) exit 1, "ERROR: --mutations_file is required (training mode)"
}
MUT_FILE = params.pipeline_only
  ? Channel.empty()
  : Channel.fromPath(params.mutations_file, checkIfExists: true)

if (params.pipeline_only && !params.all_pred)
  exit 1, "ERROR: --all_pred is required (pipeline-only mode)"
ALL_PRED = params.pipeline_only
  ? Channel.fromPath(params.all_pred, checkIfExists: true)
  : Channel.empty()

// ==============================
// Processes
// ==============================

/*
 * DOWNLOAD_BREAST_RELEASE
 * - GitHub Releases asset (tar.gz)에 동봉된
 *   cls_embedding.pkl / feature_dict_BRCA.pkl 다운로드 & 검증 & 해제
 * - 출력: cls.pkl, feat.pkl (work 디렉토리 경로)
 */
process DOWNLOAD_BREAST_RELEASE {
  tag "download:${params.release_tag ?: 'NA'}"
  cpus 1
  memory '2 GB'
  time '2h'

  output:
  path "cls_embedding.pkl"
  path "feature_dict_BRCA.pkl"

  when:
  params.use_release

  script:
  if (!params.gh_repo || !params.release_tag) {
    throw new RuntimeException("use_release=true 이면 --gh_repo 와 --release_tag 를 지정해야 합니다.")
  }
  def url = "https://github.com/${params.gh_repo}/releases/download/${params.release_tag}/${params.asset_name}"
  def auth = params.gh_token ? "-H \"Authorization: Bearer ${params.gh_token}\"" : ""
  def sha  = params.asset_sha256
  """
  set -euo pipefail

  ASSET_URL='${url}'
  ASSET='${params.asset_name}'
  AUTH=${params.gh_token ? 1 : 0}

  echo "[DL] \$ASSET_URL"
  if [ \$AUTH -eq 1 ]; then
    curl -fL -H "Authorization: Bearer ${params.gh_token}" -o "\$ASSET" "\$ASSET_URL"
  else
    curl -fL -o "\$ASSET" "\$ASSET_URL"
  fi

  # (선택) 체크섬 검증
  ${ sha ? "echo \"${sha}  \$ASSET\" | sha256sum -c -" : "true" }

  # 압축 해제 (두 PKL 포함되어 있다고 가정)
  tar -xzf "\$ASSET"

  # 표준 파일명으로 존재 확인/리네임
  test -f "cls_embedding.pkl" || { echo "[ERR] cls_embedding.pkl not found in archive"; exit 2; }
  test -f "feature_dict_BRCA.pkl" || { echo "[ERR] feature_dict_BRCA.pkl not found in archive"; exit 2; }
  """
}

/*
 * LOCAL_BREAST (fallback)
 * - repo 내부의 data/breast/*.pkl 사용
 * - 분할 파일(.part_*)만 올렸다면 자동 조립
 */
process LOCAL_BREAST {
  tag "local"
  cpus 1
  memory '1 GB'
  time '1h'

  output:
  path "cls_embedding.pkl"
  path "feature_dict_BRCA.pkl"

  when:
  !params.use_release

  script:
  """
  set -euo pipefail
  REPO="\${projectDir}"
  BREAST="\${REPO}/data/breast"

  # CLS
  if compgen -G "\${BREAST}/cls_embedding.pkl.part_*" > /dev/null; then
    cat \$(printf "%s\\n" \${BREAST}/cls_embedding.pkl.part_* | LC_ALL=C sort) > cls_embedding.pkl
  else
    [ -f "\${BREAST}/cls_embedding.pkl" ] || { echo "[ERR] not found: \${BREAST}/cls_embedding.pkl"; exit 2; }
    ln -s "\${BREAST}/cls_embedding.pkl" cls_embedding.pkl
  fi

  # FEAT
  if compgen -G "\${BREAST}/feature_dict_BRCA.pkl.part_*" > /dev/null; then
    cat \$(printf "%s\\n" \${BREAST}/feature_dict_BRCA.pkl.part_* | LC_ALL=C sort) > feature_dict_BRCA.pkl
  else
    [ -f "\${BREAST}/feature_dict_BRCA.pkl" ] || { echo "[ERR] not found: \${BREAST}/feature_dict_BRCA.pkl"; exit 2; }
    ln -s "\${BREAST}/feature_dict_BRCA.pkl" feature_dict_BRCA.pkl
  fi
  """
}

/*
 * DRIVERFORMER_TRAIN
 * - 입력 CLS/FEAT 파일은 위 두 프로세스 중 하나에서 전달
 * - mutations_file 은 params 로 전달
 */
process DRIVERFORMER_TRAIN {
  tag "train"
  cpus 8
  memory '32 GB'
  time '48h'
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
  if python -c "import driverformer" >/dev/null 2>&1; then
    echo "[INFO] using installed driverformer"
  else
    echo "[INFO] pip install -e repo (no-deps)"
    pip install -e "\${REPO}" --no-deps || true
  fi

  if command -v driverformer >/dev/null 2>&1; then
    driverformer \
      --cls-file  "\${cls_pkl}" \
      --feat-file "\${feat_pkl}" \
      --mutations-file "\${mut_file}" \
      ${A}
  else
    python -m driverformer.cli \
      --cls-file  "\${cls_pkl}" \
      --feat-file "\${feat_pkl}" \
      --mutations-file "\${mut_file}" \
      ${A}
  fi
  """
}

/*
 * DRIVERFORMER_PIPE
 * - pipeline-only 모드에서 postproc만 실행
 */
process DRIVERFORMER_PIPE {
  tag "pipe"
  cpus 4
  memory '16 GB'
  time '12h'
  publishDir params.post_dir, mode: 'copy', overwrite: true

  input:
  path all_pred

  script:
  def B = pipeArgs()
  """
  set -euo pipefail

  REPO="\${projectDir}"
  if python -c "import driverformer" >/dev/null 2>&1; then
    echo "[INFO] using installed driverformer"
  else
    echo "[INFO] pip install -e repo (no-deps)"
    pip install -e "\${REPO}" --no-deps || true
  fi

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
  } else {
    Channel
      .of(params.use_release ? "release" : "local")
      .switch {
        it == "release"  -> DOWNLOAD_BREAST_RELEASE()
        it == "local"    -> LOCAL_BREAST()
      }
      .set { BREAST_PKLS }             // emits: [cls_embedding.pkl, feature_dict_BRCA.pkl]

    // zip 두 경로를 한 묶음으로 전달
    BREAST_PKLS
      .toSortedList()
      .ifEmpty { error "No breast PKLs prepared." }
      .flatten()
      .collate(2)
      .map { it[0] + "\t" + it[1] }
      .set { CLS_FEAT_PAIR }

    // run training
    CLS_FEAT_PAIR
      .combine( MUT_FILE )
      .map { pair, mut -> tuple( pair.tokenize('\t')[0], pair.tokenize('\t')[1], mut ) }
      .into { TRAIN_IN }

    TRAIN_IN | DRIVERFORMER_TRAIN
  }
}
