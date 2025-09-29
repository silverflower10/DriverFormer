#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cat "$ROOT"/data/breast/cls_embedding.pkl.part_* > "$ROOT"/data/breast/cls_embedding.pkl
cat "$ROOT"/data/breast/feature_dict_BRCA.pkl.part_* > "$ROOT"/data/breast/feature_dict_BRCA.pkl
if command -v sha256sum >/dev/null 2>&1 && [ -f "$ROOT/data/breast/CHECKSUMS.sha256" ]; then
  (cd "$ROOT/data/breast" && sha256sum -c CHECKSUMS.sha256 || true)
fi
echo "[OK] assembled."
