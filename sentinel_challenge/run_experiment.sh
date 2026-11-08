#!/usr/bin/env bash
# Shared launch settings for the standard benchmark methods. Run inside a Slurm allocation on Unity.
set -euo pipefail

usage() {
    echo "Usage: $0 {gt|no_gt} METHOD SCENE AGENT_NUM {patrol|stationary} SENTINEL_NUM JOB_ID [CHECKPOINT]" >&2
    exit 2
}

[[ $# -ge 7 ]] || usage
perception=$1
method=$2
scene=$3
agent_num=$4
sentinel_type=$5
sentinel_num=$6
job_id=$7

case "$perception" in gt|no_gt) ;; *) usage ;; esac
case "$method" in
    center|center_no_avoidance|coela|cosar|cosar_no_analyzer|cosar_no_emergency_avoidance|cosar_no_refine|cosar_no_spatial_memory|fixed|mat|mcts|rl|roco) ;;
    *) usage ;;
esac
case "$sentinel_type" in patrol|stationary) ;; *) usage ;; esac
[[ -n "$scene" && "$agent_num" =~ ^[0-9]+$ && "$agent_num" =~ [1-9] ]] || usage
[[ "$sentinel_num" =~ ^[0-9]+$ && "$job_id" =~ ^[0-9]+$ ]] || usage
if [[ "$method" == rl || "$method" == mat ]]; then
    [[ $# -le 8 ]] || usage
else
    [[ $# -eq 7 ]] || usage
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

agent_type=$method
case "$method" in
    cosar_*) agent_type=cosar ;;
    center_no_avoidance) agent_type=center ;;
esac

args=(--head_less --backend gpu)
if [[ "$method" != mat ]]; then
    args+=(--multi_process)
fi
args+=(--skip_avatar_animation --enable_gt_segmentation)
if [[ "$perception" == no_gt ]]; then
    args+=(--gt_only_for_sentinels)
fi
if [[ "$perception" == no_gt || "$method" == coela ]]; then
    args+=(--detect_interval 1)
fi
args+=(--output_dir sentinel_challenge/output --scene "$scene" --job_id "$job_id")
case "$method" in
    cosar_no_analyzer) args+=(--ablate analyzer) ;;
    cosar_no_emergency_avoidance) args+=(--ablate emergency_avoidance) ;;
    cosar_no_spatial_memory) args+=(--ablate spatial_memory) ;;
esac
args+=(--enable_outdoor_objects --enable_indoor_scene --outdoor_objects_max_num 5
       --resolution 512 --config agents_num_15 --agent_type "$agent_type"
       --agent_num "$agent_num" --sentinel_type "$sentinel_type" --sentinel_num "$sentinel_num")
if [[ "$method" != fixed && "$method" != center_no_avoidance ]]; then
    args+=(--enable_danger_zone)
fi
if [[ "$method" == cosar_no_refine ]]; then
    args+=(--refine_retry -1)
fi
args+=(--save_per_seconds 200 --step_limit 1500)
case "$method" in
    mat) args+=(--mat_ckpt "${8:-sentinel_challenge/checkpoints_mat/latest.pt}" --mat_planning_interval 50) ;;
    rl) args+=(--rl_ckpt "${8:-sentinel_challenge/checkpoints/latest.pt}") ;;
    *) args+=(--lm_source azure --lm_id gpt-4o) ;;
esac
args+=(--debug --overwrite)
exec python sentinel_challenge/challenge.py "${args[@]}"
