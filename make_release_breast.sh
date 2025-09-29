#!/usr/bin/env bash
set -euo pipefail

# ========= 설정 =========
OWNER="silverflower10"                 # ← 깃허브 ID
REPO="DriverFormer"                    # ← 저장소
TAG="${1:-breast-data-v1}"             # ← 릴리즈 태그(인자로 교체 가능)
NAME="Breast data ${TAG}"              # 릴리즈 제목
BODY="CLS + FEAT PKL (split parts) for training. See CHECKSUMS and manifest."

# 원본 파일(로컬)
CLS_SRC="/home/silverflo/BORI/data/cls_embedding.pkl"
FEAT_SRC="/home/silverflo/BORI/data/BRCA/feature_dict_BRCA.pkl"

# 에셋 작업 디렉토리(임시)
ASSET_DIR="$(pwd)/release_assets_${TAG}"
PART_SIZE_MB=1024                      # 파트 크기(1GiB; GitHub asset 2GiB 제한 대비 안전)
mkdir -p "$ASSET_DIR"

# ========= 토큰 입력 =========
if [ -z "${GH_PAT:-}" ]; then
  read -s -p "GitHub PAT: " GH_PAT; echo
fi

# ========= 파일 준비: 분할 + 체크섬 =========
echo "[INFO] preparing assets in $ASSET_DIR"
cd "$ASSET_DIR"

# 1) 분할
echo "[SPLIT] $CLS_SRC"
split -b ${PART_SIZE_MB}M "$CLS_SRC" "cls_embedding.pkl.part_"
echo "[SPLIT] $FEAT_SRC"
split -b ${PART_SIZE_MB}M "$FEAT_SRC" "feature_dict_BRCA.pkl.part_"

# 2) 체크섬
echo "[CHECKSUM] generating CHECKSUMS.sha256"
sha256sum cls_embedding.pkl.part_* feature_dict_BRCA.pkl.part_* > CHECKSUMS.sha256

# 3) 매니페스트(다운로드/조립용)
cat > manifest.txt <<MAN
# Manifest for breast data ${TAG}
# How to reconstruct:
#  cat cls_embedding.pkl.part_* > cls_embedding.pkl
#  cat feature_dict_BRCA.pkl.part_* > feature_dict_BRCA.pkl
#  sha256sum -c CHECKSUMS.sha256   # optional
MAN

# ========= 릴리즈 생성 (이미 있으면 그 ID 재사용) =========
API="https://api.github.com"
REL_REQ='{"tag_name":"'"$TAG"'","name":"'"$NAME"'","body":"'"$BODY"'","draft":false,"prerelease":false}'
echo "[REL] creating or fetching release $TAG"

set +e
REL_JSON=$(curl -s -H "Authorization: token ${GH_PAT}" \
               -H "Accept: application/vnd.github+json" \
               "${API}/repos/${OWNER}/${REPO}/releases/tags/${TAG}")
set -e

if echo "$REL_JSON" | grep -q '"id"'; then
  RELEASE_ID=$(echo "$REL_JSON" | sed -n 's/.*"id":[ ]*\([0-9]\+\).*/\1/p' | head -1)
  echo "[REL] found existing release id=${RELEASE_ID}"
else
  REL_JSON=$(curl -s -X POST -H "Authorization: token ${GH_PAT}" \
                   -H "Accept: application/vnd.github+json" \
                   -d "$REL_REQ" \
                   "${API}/repos/${OWNER}/${REPO}/releases")
  RELEASE_ID=$(echo "$REL_JSON" | sed -n 's/.*"id":[ ]*\([0-9]\+\).*/\1/p' | head -1)
  if [ -z "$RELEASE_ID" ]; then
    echo "[ERR] failed to create release"; echo "$REL_JSON"; exit 1
  fi
  echo "[REL] created release id=${RELEASE_ID}"
fi

UPLOAD_BASE="https://uploads.github.com/repos/${OWNER}/${REPO}/releases/${RELEASE_ID}/assets?name="

# ========= 업로드 함수 =========
upload_asset () {
  local FILE="$1"
  local NAME="$(basename "$FILE")"
  echo "[UP] $NAME"
  curl --retry 3 -s -X POST \
    -H "Authorization: token ${GH_PAT}" \
    -H "Content-Type: application/octet-stream" \
    --data-binary @"$FILE" \
    "${UPLOAD_BASE}${NAME}" \
    >/dev/null || { echo "[ERR] upload failed: $NAME"; exit 1; }
}

# ========= 기존 동일 이름 에셋 있으면 삭제(중복 방지) =========
echo "[CLEAN] deleting existing assets with same names if any"
ASSETS_JSON=$(curl -s -H "Authorization: token ${GH_PAT}" \
                   -H "Accept: application/vnd.github+json" \
                   "${API}/repos/${OWNER}/${REPO}/releases/${RELEASE_ID}/assets")
for f in CHECKSUMS.sha256 manifest.txt cls_embedding.pkl.part_* feature_dict_BRCA.pkl.part_*; do
  N=$(basename "$f")
  ID=$(echo "$ASSETS_JSON" | awk -v n="\"${N}\"" '
    $0 ~ "\"name\":" n {getline; getline; if ($0 ~ /"id":/) {gsub(/[^0-9]/,"",$0); print $0; exit}}')
  if [ -n "$ID" ]; then
    echo "  - delete $N (asset id=$ID)"
    curl -s -X DELETE -H "Authorization: token ${GH_PAT}" \
         "${API}/repos/${OWNER}/${REPO}/releases/assets/${ID}" >/dev/null || true
  fi
done

# ========= 업로드 실행 =========
upload_asset CHECKSUMS.sha256
upload_asset manifest.txt
for p in cls_embedding.pkl.part_*; do upload_asset "$p"; done
for p in feature_dict_BRCA.pkl.part_*; do upload_asset "$p"; done

echo "[DONE] Release assets uploaded for ${OWNER}/${REPO} tag ${TAG}"
echo "[NOTE] Download & reconstruct:"
echo "  curl -LO https://github.com/${OWNER}/${REPO}/releases/download/${TAG}/manifest.txt"
echo "  curl -LO https://github.com/${OWNER}/${REPO}/releases/download/${TAG}/CHECKSUMS.sha256"
echo "  # and all *.part_* files, then:"
echo "  cat cls_embedding.pkl.part_* > cls_embedding.pkl"
echo "  cat feature_dict_BRCA.pkl.part_* > feature_dict_BRCA.pkl"
echo "  sha256sum -c CHECKSUMS.sha256   # optional"
