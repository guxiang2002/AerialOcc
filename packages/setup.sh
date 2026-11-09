#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPENDENCY="$ROOT/.deps/cgformer-runtime"
REVISION=2efd2c70cb2fe9e89a206f2ee6d3a62bae85f891
EXPECTED_TREE=c018afa1f76b1880437bcc3c8628f23dd11749cc

# The unchanged upstream framework is an installed dependency, not a second
# collection of datasets, examples and historical experiments in this release.
if [[ ! -d "$DEPENDENCY/.git" ]]; then
    mkdir -p "$ROOT/.deps"
    git clone --filter=blob:none --no-checkout https://github.com/pkqbajng/CGFormer.git "$DEPENDENCY"
    git -C "$DEPENDENCY" sparse-checkout init --cone
    git -C "$DEPENDENCY" sparse-checkout set packages/mmdetection3d
fi
if [[ -n "$(git -C "$DEPENDENCY" status --porcelain)" ]]; then
    echo "Dependency checkout contains local changes: $DEPENDENCY" >&2
    exit 1
fi
git -C "$DEPENDENCY" fetch --depth=1 origin "$REVISION"
git -C "$DEPENDENCY" checkout --detach "$REVISION"
ACTUAL_TREE="$(git -C "$DEPENDENCY" rev-parse "$REVISION:packages/mmdetection3d")"
[[ "$ACTUAL_TREE" == "$EXPECTED_TREE" ]] || { echo 'Upstream runtime tree mismatch' >&2; exit 1; }

python -m pip install -r "$DEPENDENCY/packages/mmdetection3d/requirements/runtime.txt"
python -m pip install --no-build-isolation "$DEPENDENCY/packages/mmdetection3d"
python -m pip install --no-build-isolation "$ROOT/packages/DFA3D"
