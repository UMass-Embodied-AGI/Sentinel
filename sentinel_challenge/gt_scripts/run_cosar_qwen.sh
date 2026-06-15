#!/bin/bash

scene=$1
agent_num=$2
sentinel_type=$3
sentinel_num=$4
job_id=$5

# --- CONFIGURATION ---
BASE_PORT=8000

# Define scene list and find index for port offset
scenes=("MADRID" "HARVARD" "AMSTERDAM" "AUSTIN" "BERLIN" "CALGARY" "COLUMBUS" "DENVER" "DETROIT" "EL_PASO" "HAMBURG" "LONGISLAND" "MADISON" "LONDON")

# Find scene index (0-13)
scene_index=0
for i in "${!scenes[@]}"; do
    if [[ "${scenes[$i]}" == "${scene}" ]]; then
        scene_index=$i
        break
    fi
done

# Sentinel type offset (patrol=0, stationary=1)
if [[ "${sentinel_type}" == "patrol" ]]; then
    sentinel_offset=0
elif [[ "${sentinel_type}" == "stationary" ]]; then
    sentinel_offset=1
else
    sentinel_offset=0
fi

# --- PORT CALCULATION ---
# Formula: BASE + (Scene * 100) + (SentinelType * 10) + (JobID % 10)
# This ensures unique ports for different scenes/types even on the same node
PORT=$((BASE_PORT + (scene_index * 100) + (sentinel_offset * 10) + (job_id % 10)))

export PYTHONPATH=${PWD}

echo "=== Job Configuration ==="
echo "Scene: ${scene} (index: ${scene_index})"
echo "Sentinel Type: ${sentinel_type} (offset: ${sentinel_offset})"
echo "Job ID: ${job_id}"
echo "Assigned Port: ${PORT}"
echo "========================="

# 1. Start the server in the background with the specific port
echo "Starting ModelServer on port ${PORT}..."
conda run -n vico_nav_qwen python tools/qwen_manager/server.py --port ${PORT} &
SERVER_PID=$!

# 2. Wait for server to initialize (reduced from 3m to 30s)
sleep 30

# 3. Run the challenge script, pointing it to the specific port
python sentinel_challenge/challenge.py --head_less \
--backend gpu \
--multi_process \
--skip_avatar_animation \
--enable_gt_segmentation \
--output_dir sentinel_challenge/output \
--scene "${scene}" \
--job_id "${job_id}" \
--enable_outdoor_objects \
--enable_indoor_scene \
--outdoor_objects_max_num 5 \
--resolution 512 \
--config agents_num_15 \
--agent_type cosar \
--agent_num ${agent_num} \
--sentinel_type ${sentinel_type} \
--sentinel_num ${sentinel_num} \
--enable_danger_zone \
--save_per_seconds 200 \
--step_limit 1500 \
--lm_source local_qwen \
--lm_id Qwen2.5-VL-14B-Instruct \
--server_port ${PORT} \
--debug \
--overwrite

# 4. Cleanup: Kill the server process after the challenge finishes
echo "Cleaning up server (PID: ${SERVER_PID})..."
kill ${SERVER_PID} 2>/dev/null
wait ${SERVER_PID} 2>/dev/null

# Optional flags you had commented out:
# --enable_indoor_activities