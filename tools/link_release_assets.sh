#!/usr/bin/env bash
#
# Replace Virtual-Community/vico/assets/scenes with a symlink to this
# release's assets/scenes/. Idempotent; safe to re-run.
#
# The change lives in the submodule's working tree but is intentionally
# not committed back to upstream — Sentinel's release-only scenes shadow
# upstream's small built-in set.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
upstream_scenes="$repo_root/Virtual-Community/vico/assets/scenes"
release_scenes_rel="../../../assets/scenes"

if [ -L "$upstream_scenes" ]; then
    echo "[link_release_assets] scenes symlink already in place; nothing to do"
    exit 0
fi

if [ -d "$upstream_scenes" ]; then
    rm -rf "$upstream_scenes"
fi

ln -s "$release_scenes_rel" "$upstream_scenes"
echo "[link_release_assets] linked Virtual-Community/vico/assets/scenes -> $release_scenes_rel"
