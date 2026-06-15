scene=$1
agent_num=$2
sentinel_type=$3
sentinel_num=$4
job_id=$5

export PYTHONPATH=${PWD}

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
--refine_retry "-1" \
--save_per_seconds 200 \
--step_limit 1500 \
--lm_source azure \
--lm_id gpt-4o \
--debug \
--overwrite

# Optional flags you had commented out:
# --enable_indoor_activities
