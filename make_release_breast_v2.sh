#!/usr/bin/env bash
set -euo pipefail

OWNER="silverflower10"
REPO="DriverFormer"
TAG="${1:-breast-data-v1}"
NAME="Breast data ${TAG}"
BODY="CLS + FEAT PKL (split parts). See CHECKSUMS and manifest."

CLS_SRC="/home/silverflo/BORI/data/cls_embedding.pkl"
FEAT_SRC="/home/silverflo/BORI/data/BRCA/feature_dict_BRCA.pkl"

ASSET_DIR="$(pwd)/release_assets_${TAG}"
PART_SIZE_MB=${PART_SIZE_MB:-256}   # ← 256MB로 안정화 (필요시 512로 올려도 됨)

mkdir -p "$ASSET_DIR"
cd "$ASSET_DIR"

if [ -z "${GH_PAT:-}" ]; then
  read -s -p "GitHub PAT: " GH_PAT; echo
fi

echo "[INFO] preparing assets in $ASSET_DIR (part=${PART_SIZE_MB}MB)"
# 분할
split -b ${PART_SIZE_MB}M "$CLS_SRC"  cls_embedding.pkl.part_
split -b ${PART_SIZE_MB}M "$FEAT_SRC" feature_dict_BRCA.pkl.part_

# 체크섬/매니페스트
sha256sum cls_embedding.pkl.part_* feature_dict_BRCA.pkl.part_* > CHECKSUMS.sha256
cat > manifest.txt <<MAN
# Manifest for ${TAG}
cat cls_embedding.pkl.part_* > cls_embedding.pkl
cat feature_dict_BRCA.pkl.part_* > feature_dict_BRCA.pkl
sha256sum -c CHECKSUMS.sha256
MAN

API="https://api.github.com"
# 릴리즈 만들기/얻기
REL_JSON=$(curl -s -H "Authorization: token ${GH_PAT}" \
               -H "Accept: application/vnd.github+json" \
               "${API}/repos/${OWNER}/${REPO}/releases/tags/${TAG}")
if echo "$REL_JSON" | grep -q '"id"'; then
  RELEASE_ID=$(echo "$REL_JSON" | sed -n 's/.*"id":[ ]*\([0-9]\+\).*/\1/p' | head -1)
  echo "[REL] found existing release id=${RELEASE_ID}"
else
  REL_JSON=$(curl -s -X POST -H "Authorization: token ${GH_PAT}" \
                   -H "Accept: application/vnd.github+json" \
                   -d '{"tag_name":"'"$TAG"'","name":"'"$NAME"'","body":"'"$BODY"'","draft":false,"prerelease":false}' \
                   "${API}/repos/${OWNER}/${REPO}/releases")
  RELEASE_ID=$(echo "$REL_JSON" | sed -n 's/.*"id":[ ]*\([0-9]\+\).*/\1/p' | head -1)
  [ -n "$RELEASE_ID" ] || { echo "[ERR] create release failed"; echo "$REL_JSON"; exit 1; }
  echo "[REL] created id=${RELEASE_ID}"
fi

# 기존 에셋 목록 (중복 skip 용)
ASSETS_JSON=$(curl -s -H "Authorization: token ${GH_PAT}" \
                   -H "Accept: application/vnd.github+json" \
                   "${API}/repos/${OWNER}/${REPO}/releases/${RELEASE_ID}/assets")
has_asset () {  # $1=name  -> returns 0 if exists
  echo "$ASSETS_JSON" | grep -q "\"name\": *\"$1\""
}

UPLOAD_BASE="https://uploads.github.com/repos/${OWNER}/${REPO}/releases/${RELEASE_ID}/assets?name="

# 스트리밍 업로드 (HTTP/1.1 + stdin)
upload_asset () {
  local FILE="$1"; local NAME="$(basename "$FILE")"
  if has_asset "$NAME"; then
    echo "[SKIP] $NAME (exists)"
    return 0
  fi
  echo "[UP] $NAME"
  # 파일 크기(Content-Length) 계산 (GNU/BSD 호환)
  if stat --version >/dev/null 2>&1; then
    CLEN=$(stat -c%s "$FILE")
  else
    CLEN=$(wc -c < "$FILE")
  fi
  # 스트리밍 업로드
  CURLCMD=(
    curl --http1.1 --retry 3 -sS -X POST
      -H "Authorization: token ${GH_PAT}"
      -H "Content-Type: application/octet-stream"
      -H "Content-Length: ${CLEN}"
      --data-binary @-
      "${UPLOAD_BASE}${NAME}"
  )
  "${CURLCMD[@]}" < "$FILE" >/dev/null
}

# 우선 작은 파일
upload_asset CHECKSUMS.sha256
upload_asset manifest.txt

# 큰 파트들(정렬해서 순서 보장)
for p in $(printf "%s\n" cls_embedding.pkl.part_* | LC_ALL=C sort); do upload_asset "$p"; done
for p in $(printf "%s\n" feature_dict_BRCA.pkl.part_* | LC_ALL=C sort); do upload_asset "$p"; done

echo "[DONE] Upload completed for tag ${TAG}"
echo "Download example:"
echo "  base=https://github.com/${OWNER}/${REPO}/releases/download/${TAG};"
echo "  curl -LO \$base/manifest.txt && curl -LO \$base/CHECKSUMS.sha256"
echo "  # download all parts (use aria2c or parallel curl) then:"
echo "  cat cls_embedding.pkl.part_* > cls_embedding.pkl"
echo "  cat feature_dict_BRCA.pkl.part_* > feature_dict_BRCA.pkl"
echo "  sha256sum -c CHECKSUMS.sha256"
