import argparse
import copy
import json
from datetime import datetime
import os
import sys
import shutil, errno
import genesis as gs
import numpy as np
import multiprocessing
from PIL import Image
import threading, time


from agents.sentinel_challenge import *
from agents.memory import SemanticMemory
from env import SentinelVicoEnv as VicoEnv
from vico.agents.agent import AgentProcess
from agents.demo_agent import DemoAgentProcess
from vico.modules import *

keep_running = False

def gpu_occupy_loop():
    import torch
    if not torch.cuda.is_available():
        return
    while keep_running:
        a = torch.randn(4096, 4096, device='cuda')
        _ = torch.mm(a, a)
        time.sleep(0.5)

def end_processes():
    for p in multiprocessing.active_children():
        print(f"[Multiprocessing] PID={p.pid} Name={p.name}")
        p.terminate()
        p.join()

def get_gpu_info(timeout=3):
    import subprocess
    print("Getting gpu info")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
    except subprocess.TimeoutExpired:
        return [{"error": "nvidia-smi timed out"}]
    except FileNotFoundError:
        return [{"error": "nvidia-smi not found"}]
    except subprocess.CalledProcessError as e:
        return [{"error": e.stderr.strip()}]

    gpus = []
    for line in result.stdout.strip().split("\n"):
        name, vram = line.split(", ")
        gpus.append({"name": name, "vram_MB": int(vram)})
    print(f"gpu info gotten: {gpus}")
    return gpus

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", type=str, default='32')
    parser.add_argument("--logging_level", type=str, default='info')
    parser.add_argument("--backend", type=str, default='cpu')
    parser.add_argument("--head_less", '-l', action='store_true')
    parser.add_argument("--multi_process", '-m', action='store_true')
    parser.add_argument("--output_dir", "-o", type=str, default='sentinel_challenge/output')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--overwrite", action='store_true')
    parser.add_argument("--job_id", type=int, default=0)

    ### Simulation configurations
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--enable_collision", action='store_true')
    parser.add_argument("--skip_avatar_animation", action='store_true')
    parser.add_argument("--enable_gt_segmentation", action='store_true')
    parser.add_argument("--max_seconds", type=int, default=86400) # 24 hours
    parser.add_argument("--save_per_seconds", type=int, default=10)
    parser.add_argument("--enable_third_person_cameras", action='store_true')
    parser.add_argument("--curr_time", type=str)
    parser.add_argument("--start_id", type=int)
    parser.add_argument("--only_one_sample", action='store_true')
    parser.add_argument("--use_luisa_renderer", action='store_true')

    ### Scene configurations
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--enable_indoor_scene", action='store_true')
    parser.add_argument("--enable_indoor_activities", action='store_true')
    parser.add_argument("--enable_outdoor_objects", action='store_true')
    parser.add_argument("--outdoor_objects_assets_dir", type=str, default='scene/object_assets')
    parser.add_argument("--outdoor_objects_max_num", type=int, default=10)
    parser.add_argument("--no_load_scene", action='store_true')

    # Traffic configurations
    parser.add_argument("--no_traffic_manager", action='store_true')
    parser.add_argument("--tm_vehicle_num", type=int, default=0)
    parser.add_argument("--tm_avatar_num", type=int, default=0)
    parser.add_argument("--enable_tm_debug", action='store_true')

    ### Agent configurations
    parser.add_argument("--config", type=str, default='agents_num_25')
    parser.add_argument("--agent_num", type=int, default=5)
    parser.add_argument("--agent_type", type=str, choices=['center', 'roco', 'coela', 'fixed', 'cosar', 'mcts', 'demo', 'rl', 'mat'])
    parser.add_argument("--agent_type2", type=str, choices=['heuristic', 'llm', 'mcts', 'random'])
    parser.add_argument("--no_react", action='store_true')
    parser.add_argument("--lm_source", type=str, choices=["openai", "azure", "huggingface", "local_qwen"], default="azure", help="language model source")
    parser.add_argument("--lm_id", "-lm", type=str, default="gpt-35-turbo", help="language model id")
    parser.add_argument("--max_tokens", type=int, default=4096, help="maximum tokens")
    parser.add_argument("--temperature", "-t", type=float, default=0, help="temperature")
    parser.add_argument("--top_p", type=float, default=1, help="top p")
    parser.add_argument("--server_port", type=int, default=8000, help="port for local LM server, only used when lm_source is local_qwen")

    # meeting challenge
    parser.add_argument("--robot_as_agent", action='store_true')
    parser.add_argument("--enable_demo_camera", action='store_true')
    parser.add_argument("--step_limit", type=int, default=1500)
    parser.add_argument("--robot_policy_path", type=str, default="", help="Where to load robot policy")
    parser.add_argument("--sentinel_type", type=str, default="patrol", choices=['stationary', 'patrol'])
    parser.add_argument("--sentinel_num", type=int, default=5)
    parser.add_argument("--enable_danger_zone", action='store_true')
    parser.add_argument("--refine_retry", type=int, default=5)
    parser.add_argument("--gt_only_for_sentinels", action='store_true')
    parser.add_argument("--detect_interval", type=int, default=-1)
    parser.add_argument("--ablate", type=str, default="")
    parser.add_argument("--replay_mode", action='store_true')
    parser.add_argument("--rl_ckpt", type=str, default=None,
                        help="Path to PPO checkpoint for --agent_type rl. "
                             "If unset, agents use a random-init policy.")
    parser.add_argument("--rl_training_mode", action='store_true',
                        help="Internal flag set by sentinel_challenge/train_rl.py "
                             "to enable trajectory logging on RLMeetingAgent.")
    parser.add_argument("--mat_ckpt", type=str, default=None,
                        help="Path to MAT checkpoint for --agent_type mat.")
    parser.add_argument("--mat_training_mode", action='store_true',
                        help="Internal flag set by sentinel_challenge/train_mat.py.")
    parser.add_argument("--mat_planning_interval", type=int, default=50,
                        help="Macro-action duration for MAT (CoELA default 50).")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    run_challenge(args)


def run_challenge(args):
    random.seed(time.time())
    # Make output directories
    if args.agent_type == 'llm' and args.lm_id != 'gpt-4o':
        output_dir = os.path.join(args.output_dir, args.scene,
                                       f"{args.agent_type}-{args.lm_id.split('/')[0]}")
    else:
        agent_type = args.agent_type
        agent_type_name = f"{agent_type}"
        if not args.enable_danger_zone:
            agent_type_name = f"{agent_type}_no_avoidance"
        if args.refine_retry <= 0:
            agent_type_name = f"{agent_type}_no_refine"
        if args.ablate != "":
            agent_type_name = f"{agent_type}_ablate_{args.ablate}"
        if args.lm_source == "local_qwen":
            agent_type_name = f"{agent_type_name}_local_qwen"
        output_dir = os.path.join(args.output_dir, args.scene, f"{agent_type_name}_{'no_gt' if args.gt_only_for_sentinels else 'gt'}_{args.agent_num}", f"{args.sentinel_type}_{args.sentinel_num}", f"job_{args.job_id}")
    if args.replay_mode:
        replay_steps_info = None
        replay_steps_path = os.path.join(args.output_dir, args.scene, f"{agent_type_name}_{'no_gt' if args.gt_only_for_sentinels else 'gt'}_{args.agent_num}", f"{args.sentinel_type}_{args.sentinel_num}", f"job_{args.job_id}", "steps.json")
        if not os.path.exists(replay_steps_path):
            print(f"Replay steps {replay_steps_path} doesn't exist!")
            end_processes()
            exit(0)
        with open(replay_steps_path, "r") as f:
            replay_steps_info = json.load(f)
        replay_step_keys = sorted(replay_steps_info.keys(), key=lambda x: int(x))
        replay_output_dir = os.path.join("sentinel_challenge", "replay_outputs")
        if args.use_luisa_renderer:
            output_dir = os.path.join(replay_output_dir, args.scene, f"{agent_type_name}_{'no_gt' if args.gt_only_for_sentinels else 'gt'}_{args.agent_num}", f"{args.sentinel_type}_{args.sentinel_num}", f"job_{args.job_id}_luisa")
            args.enable_indoor_scene = False
        else:
            output_dir = os.path.join(replay_output_dir, args.scene, f"{agent_type_name}_{'no_gt' if args.gt_only_for_sentinels else 'gt'}_{args.agent_num}", f"{args.sentinel_type}_{args.sentinel_num}", f"job_{args.job_id}")
    # Make job result directories
    job_result_path = os.path.join(f"sentinel_challenge/results_{'no_gt' if args.gt_only_for_sentinels else 'gt'}/{args.agent_num}_{args.sentinel_type}_{args.sentinel_num}/", f"{agent_type_name}", args.scene)
    os.makedirs(job_result_path, exist_ok=True)
    job_result_path = os.path.join(job_result_path, f"result_{args.job_id}.json")
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, "result.json")
    if os.path.exists(result_path) and args.job_id != 0 and not args.replay_mode: # 0 is for debug
        result = json.load(open(result_path, 'r'))
        print(f"results exists: {result}")
        print(f"it's already done. Skip running simulation")
        end_processes()
        exit(0)
    if args.overwrite and os.path.exists(output_dir):
        print(f"Overwrite the output directory: {output_dir}")
        shutil.rmtree(output_dir)
    print(f"Initializing output directory...")
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'debug'), exist_ok=True)

    # Read and initialize scene configuration
    os.makedirs(output_dir, exist_ok=True)
    print(f"Dumping gpu environment...")
    json.dump({"gpus": get_gpu_info()}, open(os.path.join(output_dir, "gpus.json"), 'w'), indent=4)
    config_path = os.path.join(output_dir, 'curr_sim')
    if not os.path.exists(config_path):
        seed_config_path = os.path.join('assets/scenes', args.scene, args.config)
        print(f"Initiate new simulation from config: {seed_config_path}")
        try:
            shutil.copytree(seed_config_path, config_path)
        except OSError as exc:  # python >2.5
            if exc.errno in (errno.ENOTDIR, errno.EINVAL):
                shutil.copy(seed_config_path, config_path)
            else:
                raise
    else:
        print(f"Continue simulation from config: {config_path}")
    config = json.load(open(os.path.join(config_path, "config.json"), 'r'))
    num_agents = args.agent_num
    if config["num_agents"]>num_agents:
        config["num_agents"] = num_agents
        config["agent_names"]=config['agent_names'][:num_agents]
        config['agent_infos']=config['agent_infos'][:num_agents]
        config['agent_poses']=config['agent_poses'][:num_agents]
        config['locator_colors']=config['locator_colors'][:num_agents]
        config['locator_colors_rgb']=config['locator_colors_rgb'][:num_agents]
        config['agent_skins']=config['agent_skins'][:num_agents]
    sentinel_config_path = os.path.join('assets/scenes', args.scene, "sentinel_config", f"sentinel_config_{args.sentinel_type}.json")
    if os.path.exists(sentinel_config_path):
        print(f"adding sentinels to config...")
        sentinel_config = json.load(open(sentinel_config_path, "r"))
        num_sentinels = args.sentinel_num
        # num_sentinels = len(sentinel_config['agent_names'])
        if 'with_sentinel' not in config:
            config["agent_names"].extend(sentinel_config['agent_names'][:num_sentinels])
            config['agent_infos'].extend(sentinel_config['agent_infos'][:num_sentinels])
            config['agent_poses'].extend(sentinel_config['agent_poses'][:num_sentinels])
            config['locator_colors'].extend(sentinel_config['locator_colors'][:num_sentinels])
            config['locator_colors_rgb'].extend(sentinel_config['locator_colors_rgb'][:num_sentinels])
            config['agent_skins'].extend(sentinel_config['agent_skins'][:num_sentinels])
            config['num_agents']+=num_sentinels
            config['dt_control'].extend([config['dt_control'][0]]*(num_sentinels+num_agents-len(config['dt_control'])))
            config['dt_visual_obs'].extend([config['dt_visual_obs'][0]]*(num_sentinels+num_agents-len(config['dt_visual_obs'])))
            config['with_sentinel'] = True
            json.dump(config, open(os.path.join(config_path, "config.json"), 'w'), indent=4)
            for name in sentinel_config['agent_names']:
                if os.path.exists(os.path.join(config_path, name)): continue
                shutil.copytree(os.path.join(config_path, config['agent_names'][0]), os.path.join(config_path, name))
    else:
        print(f"No sentinel config found at {sentinel_config_path} !!!")
        sentinel_config = None
        num_sentinels = 0
    json.dump({"gpus": get_gpu_info(), "success_config_initialization": True}, open(os.path.join(output_dir, "gpus.json"), 'w'), indent=4)

    if args.debug:
        args.enable_third_person_cameras = True
    env = VicoEnv(
        seed=args.seed,
        precision=args.precision,
        logging_level=args.logging_level,
        backend=gs.cpu if args.backend == 'cpu' else gs.gpu,
        head_less=args.head_less,
        resolution=args.resolution,
        challenge='meeting',
        num_agents=config["num_agents"],
        config_path=config_path,
        scene=args.scene,
        enable_indoor_scene=args.enable_indoor_scene,
        enable_outdoor_objects=args.enable_outdoor_objects,
        outdoor_objects_max_num=args.outdoor_objects_max_num,
        enable_collision=args.enable_collision,
        skip_avatar_animation=args.skip_avatar_animation,
        enable_gt_segmentation=args.enable_gt_segmentation,
        no_load_scene=args.no_load_scene,
        output_dir=output_dir,
        enable_third_person_cameras=args.enable_third_person_cameras,
        # use_luisa_renderer=args.use_luisa_renderer,
        no_traffic_manager=args.no_traffic_manager,
        enable_tm_debug=args.enable_tm_debug,
        tm_vehicle_num=args.tm_vehicle_num,
        tm_avatar_num=args.tm_avatar_num,
        save_per_seconds=args.save_per_seconds,
        debug=args.debug,
        enable_demo_camera=args.enable_demo_camera,
    )
    obs = env.reset()

    # MAT runs need a single shared in-process controller. Construct it before
    # building subagents so MATSubAgent.__init__ can register with it.
    if agent_type == "mat":
        if args.multi_process:
            gs.logger.warning(
                "MAT requires in-process agents; forcing multi_process=False.")
            args.multi_process = False
        from agents.sentinel_challenge.MAT import (
            MATController, set_active_controller,
        )
        mat_save_path = os.path.join(output_dir, "mat_trajectory.pkl")
        mat_controller = MATController(
            num_agents=num_agents,
            policy_ckpt=args.mat_ckpt,
            step_limit=args.step_limit,
            planning_interval=args.mat_planning_interval,
            training_mode=args.mat_training_mode,
            save_path=mat_save_path,
            logger=gs.logger,
        )
        set_active_controller(mat_controller)
    else:
        mat_controller = None

    # Initialize the proposer agents and NPC agents
    name2idx = {}
    all_agent_processes: list[AgentProcess] = []
    all_agent_name=[]
    for i in range(num_agents):
        basic_kwargs = dict(
            name=config['agent_names'][i],
            pose=config["agent_poses"][i],
            info=config["agent_infos"][i],
            sim_path=config_path,
            no_react=args.no_react,
            debug=args.debug,
            logging_level=args.logging_level,
            multi_process=args.multi_process,
            detect_interval=args.detect_interval,
        )
        llm_kwargs = dict(
            lm_source=args.lm_source,
            lm_id=args.lm_id,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        challenge_kwargs = dict(
            enable_danger_zone=args.enable_danger_zone,
        )
        if args.replay_mode:
            steps_for_agent = []
            for k in replay_step_keys:
                a = replay_steps_info[k].get("action", {}).get(str(i), None)
                if a is None:
                    steps_for_agent.append({'type': 'wait'})
                else:
                    steps_for_agent.append(a)
            basic_kwargs['replay_actions'] = steps_for_agent
            all_agent_processes.append(AgentProcess(ReplayAgent, **basic_kwargs)) 
        elif agent_type == 'center':
            all_agent_processes.append(AgentProcess(HeuristicNavigationMeetingAgent, **basic_kwargs, **llm_kwargs, **challenge_kwargs))
        elif agent_type == 'coela':
            all_agent_processes.append(AgentProcess(CoelaMeetingAgent, **basic_kwargs, **llm_kwargs, **challenge_kwargs))
        elif agent_type == 'roco':
            all_agent_processes.append(AgentProcess(RoCoMeetingAgent, **basic_kwargs, **llm_kwargs, **challenge_kwargs))
        elif agent_type == "fixed":
            all_agent_processes.append(AgentProcess(FixedMeetingAgent, **basic_kwargs, **llm_kwargs, **challenge_kwargs))
        elif agent_type == "mcts":
            all_agent_processes.append(AgentProcess(MCTSMeetingAgent, **basic_kwargs, **llm_kwargs, **challenge_kwargs))
        elif agent_type == "rl":
            rl_kwargs = dict(
                policy_ckpt=args.rl_ckpt,
                training_mode=args.rl_training_mode,
                step_limit=args.step_limit,
            )
            all_agent_processes.append(AgentProcess(RLMeetingAgent, **basic_kwargs, **llm_kwargs, **challenge_kwargs, **rl_kwargs))
        elif agent_type == "mat":
            mat_kwargs = dict(
                planning_interval=args.mat_planning_interval,
                step_limit=args.step_limit,
            )
            all_agent_processes.append(AgentProcess(MATSubAgent, **basic_kwargs, **llm_kwargs, **challenge_kwargs, **mat_kwargs))
        elif agent_type == "cosar":
            basic_kwargs['refine_retry'] = args.refine_retry
            basic_kwargs['ablate'] = args.ablate
            llm_kwargs['server_port'] = args.server_port
            all_agent_processes.append(AgentProcess(CoSaRMeetingAgent, **basic_kwargs, **llm_kwargs, **challenge_kwargs))
        elif agent_type == 'demo':
            agent_actions = config.get('agent_actions', [])
            action_list = agent_actions[i] if i < len(agent_actions) else []
            all_agent_processes.append(DemoAgentProcess(config['agent_names'][i], action_list))
        else:
            raise NotImplementedError(f"agent type {agent_type} is not supported")
        all_agent_name.append(config['agent_names'][i])
        name2idx[config['agent_names'][i]] = i

    
    for i in range(num_sentinels):
        adversary_type = "agent"
        basic_kwargs = dict(
            name=config['agent_names'][num_agents+i],
            pose=config['agent_poses'][num_agents+i],
            info=config["agent_infos"][num_agents+i],
            sim_path=config_path,
            no_react=args.no_react,
            debug=args.debug,
            logging_level=args.logging_level,
            multi_process=args.multi_process,
        )
        llm_kwargs = dict(
            lm_source=args.lm_source,
            lm_id=args.lm_id,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        challenge_kwargs = dict(
            patrol_config=sentinel_config['patrol_config'][i],
        )
        if args.replay_mode:
            steps_for_agent = []
            for k in replay_step_keys:
                a = replay_steps_info[k].get("action", {}).get(str(num_agents+i), None)
                if a is None:
                    steps_for_agent.append({'type': 'wait'})
                else:
                    steps_for_agent.append(a)
            basic_kwargs['replay_actions'] = steps_for_agent
            all_agent_processes.append(AgentProcess(ReplayAgent, **basic_kwargs)) 
        elif adversary_type == 'agent':
            all_agent_processes.append(AgentProcess(BaseSentinelAgent, **basic_kwargs, **challenge_kwargs))
        else:
            raise NotImplementedError(f"agent type {adversary_type} is not supported")
        name2idx[config['agent_names'][num_agents+i]] = i


    if args.multi_process:
        gs.logger.info("Start agent processes")
        for agent_process in all_agent_processes:
            agent_process.start()
        gs.logger.info("Agent processes started")

    agent_actions = {}
    agent_actions_to_print = {}
    agent_actions_to_save = {}

    # Simulation loop
    env_dt_sim = 0.
    all_task_end = False
    max_distance = 0.
    total_length=[0. for i in range(num_agents+num_sentinels)]
    total_time=[0. for i in range(num_agents+num_sentinels)]
    last_agent_pos_dict=None
    infos={"time_used_by_step": np.zeros(5, dtype=float), "time_used_by_scene_step": np.zeros(5, dtype=float)}
    banned_agent_list = []
    detected_counter = 0
    agent_list_to_update = obs.pop('agent_list_to_update')
    while not all_task_end and env.steps < args.step_limit:
        lst_time = time.perf_counter()
        gs.logger.info(f"step {env.steps} starts")
        # prcess obs_printable
        obs_printable = [{k: copy.deepcopy(v) for k, v in obs[agent_id].items() if not isinstance(v, np.ndarray) \
                          and k != 'gt_seg_idxc_to_info' and not isinstance(v, datetime) and not isinstance(v, Route)} for agent_id in obs]
        for i in range(len(obs_printable)):
            for event in obs_printable[i]['events']:
                if isinstance(event['content'], Route):
                    event['content']=event['content'].to_dict()
                if event['type'] == 'app message':
                    event['content']='place holder'

        # update obs and do action
        extra_obs = {
            "agent_pos_dict": {
                env.config["agent_names"][i]: {
                    "place": env.obs[i]['current_place'],
                    "pose": env.config["agent_poses"][i] if env.obs[i]['current_building']=='open space' else env.agent_infos[i]["outdoor_pose"]
                    } for i in range(num_agents) if env.config["agent_names"][i] in all_agent_name and env.config["agent_names"][i] not in banned_agent_list
            },
        }
        # agents step
        gs.logger.info(f"step {env.steps}: agent step starts")
        for i, agent in enumerate(all_agent_processes):
            if i in agent_list_to_update:
                obs[i].update(extra_obs)
                obs[i].update({"name": env.config["agent_names"][i]})
                if args.gt_only_for_sentinels and 'Sentinel' not in env.config["agent_names"][i]:
                    obs[i].pop('gt_seg_idxc_to_info')
                    obs[i]['segmentation']=np.full_like(obs[i]['depth'], None)
                agent.update(obs[i])
        gs.logger.info(f"step {env.steps}: agent obs update finishes")
        for i, agent in enumerate(all_agent_processes):
            if i in agent_list_to_update:
                action = agent.act()
                agent_actions[i] = action
                agent_actions_to_print[agent.name] = agent_actions[i]['type'] if agent_actions[i] is not None else None
                agent_actions_to_save[agent.name] = copy.deepcopy(action)
                if agent_actions[i] is not None and agent_actions[i]['type'] == 'converse':
                    agent_actions[i]['request_chat_func'] = agent.request_chat
                    agent_actions[i]['get_utterance_func'] = agent.get_utterance
                # action types indicate it see a sentinel
                if action is not None and action['type'] in ['signal', 'look_after_left', 'look_after_right', 'chase']:
                    detected_counter += 1
                    if action['arg1']=='ban':
                        banned_agent_list.append(action['arg2'])
        agent_actions['agent_list_to_update'] = agent_list_to_update
        gs.logger.info(f"step {env.steps}: agent action finishes")

        steps_info_path = os.path.join(output_dir, "steps.json")
        if os.path.exists(steps_info_path):
            with open(steps_info_path, 'r') as file:
                steps_info = json.load(file)
        else:
            steps_info = {}

        step_info = {"curr_time": env.curr_time.strftime("%H:%M:%S"),
                     "obs": obs_printable,
                     "action": agent_actions,
                     }
        steps_info[env.steps] = step_info
        try:
            with open(steps_info_path, 'w') as file:
                json.dump(steps_info, file, indent=4)
        except Exception as e:
            gs.logger.error(f"Error in step info storage: {e} with traceback: {traceback.format_exc()}.\n The step_info is {step_info}")
            import pdb; pdb.set_trace()

        gs.logger.info(f"{args.scene}'s current time: {env.curr_time}, ViCo steps: {env.steps}/{args.step_limit}, agent_pose: {round_numericals(env.config['agent_poses'])}, agents actions: {agent_actions_to_print}")
        sps_agent = time.perf_counter() - lst_time
        env.config["sps_agent"] = (env.config["sps_agent"] * env.steps + sps_agent) / (env.steps + 1)
        lst_time = time.perf_counter()
        # transfer look_after and chase back to turn and move_forward
        for i, agent in enumerate(all_agent_processes):
            if i in agent_list_to_update:
                if agent_actions[i] is None: continue
                if agent_actions[i]['type']=='look_after_left':
                    agent_actions[i]['type']='turn_left'
                if agent_actions[i]['type']=='look_after_right':
                    agent_actions[i]['type']='turn_right'
                if agent_actions[i]['type']=='chase':
                    agent_actions[i]['type']='move_forward'
        # then step
        obs, _, done, info = env.step(agent_actions)
        agent_list_to_update = obs.pop('agent_list_to_update')
        sps_sim = time.perf_counter() - lst_time
        env.config["sps_sim"] = (env.config["sps_sim"] * (env.steps - 1) + sps_sim) / max(env.steps, 1)
        gs.logger.info(f"Time used: {sps_agent:.2f}s for agents and robots, {sps_sim:.2f}s for simulation, "
                       f"average {env.config['sps_agent']:.2f}s for agents, "
                       f"{env.config['sps_sim']:.2f}s for simulation, "
                       f"{env.config['sps_chat']:.2f}s for post-chatting over {env.steps} steps.")
        for key in info:
            infos[key]+=info[key]
        max_distance=0.
        for idx, agent_pose in enumerate(env.config['agent_poses']):
            if idx == num_agents: break
            if env.config['agent_names'][idx] in banned_agent_list: continue
            agent_pose = np.array(agent_pose[:2])
            if agent_pose[0]>500: agent_pose-=1000
            for idx2, agent2_pose in enumerate(env.config['agent_poses']):
                if idx2 == num_agents: break
                if env.config['agent_names'][idx2] in banned_agent_list: continue
                agent2_pose = np.array(agent2_pose[:2])
                if agent2_pose[0]>500: agent2_pose-=1000
                max_distance = max(max_distance, np.linalg.norm(agent_pose-agent2_pose))
        gs.logger.info(f"The longest distance between the agents: {max_distance:.2f}")

        all_task_end = True
        to_break = False
        for idx, agent in enumerate(agent_actions_to_print):
            if idx == num_agents: break
            action = agent_actions_to_print[agent]
            if action in ["task_terminate"]: to_break = True
            if action in ['move_forward', 'turn_left', 'turn_right', 'enter', 'force_enter']:
                total_time[idx]+=1
            if action in ['move_forward'] and last_agent_pos_dict is not None:
                total_length[idx]+=np.linalg.norm(np.array(extra_obs["agent_pos_dict"][agent]['pose'][:2])-np.array(last_agent_pos_dict[agent]['pose'][:2]))
            if (action is None or action != 'task_complete') and env.steps <= args.step_limit:
                all_task_end = False
        if to_break: break
        
        last_agent_pos_dict=extra_obs["agent_pos_dict"]

    json.dump({"gpus": get_gpu_info(), "success_config_initialization": True, "infos": {"average_agent": env.config["sps_agent"] * (env.steps), "average_sim": env.config["sps_sim"] * env.steps}}, open(os.path.join(output_dir, "gpus.json"), 'w'), indent=4)
    result = {"agent_poses": [agent_pose for agent_pose in env.config['agent_poses']],
              "time_spent_meeting": env.steps,
              "walk_spent_meeting": np.sum(total_length),
              "agent_navigation_time": list(total_time),
              "agent_navigation_length": list(total_length),
              "done": int(all_task_end and (max_distance<=20) and (0==len(banned_agent_list))),
              "caught_rate": (len(banned_agent_list))/(num_agents),
              "detection_rate": (detected_counter)/(num_agents*env.steps)}
    if result['done'] == 0:
        if len(banned_agent_list):
            result['reason_fail'] = 'agent caught'
        elif not all_task_end:
            result['reason_fail'] = 'not completing'
        elif max_distance > 20:
            result['reason_fail'] = 'not meeting'
        else:
            result['reason_fail'] = 'unknown'
    else:
        result['reason_fail'] = 'success'
    with open(result_path, 'w') as file:
        json.dump(result, file, indent=4)
    with open(job_result_path, 'w') as file:
        json.dump(result, file, indent=4)
    gs.logger.warning(f"{result}")
    env.close()
    del env
    end_processes()
    # Reset the MAT singleton so a subsequent run_challenge() in the same
    # python process (e.g., from train_mat.py) starts with a clean slate.
    if mat_controller is not None:
        try:
            from agents.sentinel_challenge.MAT import set_active_controller
            set_active_controller(None)
        except Exception:
            pass
    return {
        "result": result,
        "output_dir": output_dir,
        "config_path": config_path,
        "agent_names": all_agent_name,
        "banned_agent_list": banned_agent_list,
        "mat_trajectory_path": (mat_controller.save_path
                                if mat_controller is not None else None),
    }

if __name__ == '__main__':
    keep_running = os.environ.get('keep_running', '0') == '1'
    t = threading.Thread(target=gpu_occupy_loop, daemon=True)
    t.start()
    try:
        main()
    finally:
        keep_running = False
        t.join(timeout=1)
