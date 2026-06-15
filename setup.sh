#!/usr/bin/env bash
#
# One-shot environment setup for the Sentinel Challenge.
#
# Builds on Virtual Community's uv-managed environment, then layers the
# release-only dependencies (vLLM for the Qwen-VL client and the scene-graph
# perception backbones) in the specific order that avoids known build
# conflicts:
#   - the uv venv ships without pip, so we bootstrap it;
#   - torch's C++ extension build needs pkg_resources (setuptools < 70);
#   - GroundingDINO's build pulls in numpy 2.x, which we pin back afterwards.
#
# Prerequisites: uv (https://docs.astral.sh/uv/) and a CUDA 11.7 toolchain.
# Run from the repository root:  bash setup.sh

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

echo "[setup] 1/7  Fetching submodules (Virtual-Community + Genesis)…"
git submodule update --init --recursive

echo "[setup] 2/7  Creating the uv environment (installs Genesis at the pinned commit)…"
( cd Virtual-Community && uv sync )

# Activate the venv for the remaining steps (persists for the rest of this script only).
# shellcheck disable=SC1091
source Virtual-Community/.venv/bin/activate

echo "[setup] 3/7  Bootstrapping pip + build tooling into the venv…"
python -m ensurepip --upgrade
pip install "setuptools<70" wheel        # torch's cpp_extension imports pkg_resources

echo "[setup] 4/7  Installing vLLM (Qwen-VL multimodal client)…"
uv pip install vllm

echo "[setup] 5/7  Building scene-graph perception backbones (CLIP / RAM / GroundingDINO / SAM)…"
( cd agents/sg && ./setup.sh )

echo "[setup] 6/7  Pinning numpy back to the version Genesis requires…"
pip install "numpy==1.26.4"              # GroundingDINO's build pulls numpy 2.x

echo "[setup] 7/7  Linking release scenes into the asset tree…"
bash tools/link_release_assets.sh

echo
echo "[setup] Done. In every new shell, activate the environment with:"
echo "    source Virtual-Community/.venv/bin/activate"
