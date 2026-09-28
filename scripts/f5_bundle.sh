#!/usr/bin/env bash
# Build the Colab upload set for F5 (K60) into dist/f5/: jev_f5_bundle.zip + f5_colab.ipynb.
# The bundle is taken from the committed HEAD (git archive), so its code hash matches a commit.
set -euo pipefail
cd "$(dirname "$0")/.."
git rev-parse -q --verify refs/tags/prereg-v1 >/dev/null || { echo "tag prereg-v1 missing"; exit 1; }
if [ -n "$(git status --porcelain -- src splits data/MANIFEST.json pyproject.toml notebooks)" ]; then
  echo "uncommitted changes in src/splits/notebooks: commit first"; exit 1
fi
out=dist/f5
rm -rf "$out" && mkdir -p "$out"
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
mkdir "$stage/jev"
git archive HEAD src splits data/MANIFEST.json pyproject.toml | tar -x -C "$stage/jev"
echo "prereg-v1 $(git rev-list -n1 prereg-v1) bundle-from $(git rev-parse --short HEAD)" > "$stage/jev/PREREG_TAG"
(cd "$stage" && zip -qr - jev) > "$out/jev_f5_bundle.zip"
cp notebooks/f5_colab.ipynb "$out/"
code_sha=$(.venv/bin/python -c "import sys; sys.path.insert(0,'$stage/jev/src'); from jevexp.ft import worker; print(worker.code_sha()[:12])")
echo "bundle from $(git rev-parse --short HEAD), code $code_sha, plan $(sha256sum splits/f5_plan.json | cut -c1-12)"
ls -la "$out"
