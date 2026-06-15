import ast
import os
import pdb
import random
import copy
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import numpy as np
import pickle
import re
from enum import Enum
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
from PIL import Image

from vico.agents.agent import Agent
from agents.memory import SemanticMemory, EventInstance
from vico.modules.MapTool import Route, RouteNode
from vico.tools.utils import *
from tools.model_manager import global_model_manager
from agents.sg.builder.builder import Builder, BuilderConfig


class NavAgentState(Enum):
    DISCUSS    = "BASE_DISCUSSING"
    NAVIGATE  = "BASE_NAVIGATING"

@dataclass
class Place:
    name: str
    location: list[float, float] | None = None
    bbox: list[float, float, float, float] | None = None
    region: dict | None = None

    def __init__(self, target, region_name, s_mem):
        if type(target) == dict:
            # region
            target_region = target['region']
            self.name = region_name
            target_pos = [(target_region['x_min'] + target_region['x_max']) / 2,
                      (target_region['y_min'] + target_region['y_max']) / 2]
            self.location = round_numericals(target_pos)
            self.bbox = round_numericals([target_region['x_min'], target_region['y_min'],
                         target_region['x_max'], target_region['y_max']])
            self.region = target['region']
            return
        self.name = target
        if target in s_mem.get_places():
            # place
            place_dict = s_mem.get_knowledge(target)
            self.location = round_numericals([place_dict["location"][0] - 1000, place_dict["location"][1] - 1000])
            bbox = place_dict["bounding_box"]
            if bbox is None:
                # outdoor place
                self.bbox = [self.location[0] - 4, self.location[1] - 4, self.location[0] + 4, self.location[1] + 4]
            else:
                self.bbox = round_numericals(bbox3d_to_bbox2d(bbox_center_to_corners_repr(bbox)))
            return
        # agent

    def within(self, point: list[float, float]) -> bool:
        if self.region is not None:
            return (self.region['x_min'] <= point[0] <= self.region['x_max'] and
                    self.region['y_min'] <= point[1] <= self.region['y_max'])
        elif self.bbox is not None:
            return (self.bbox[0] <= point[0] <= self.bbox[2] and
                    self.bbox[1] <= point[1] <= self.bbox[3]) or \
                     (self.bbox[0] - 1000 <= point[0] <= self.bbox[2] - 1000 and
                        self.bbox[1] - 1000 <= point[1] <= self.bbox[3] - 1000) or \
                        (self.bbox[0] + 1000 <= point[0] <= self.bbox[2] + 1000 and
                        self.bbox[1] + 1000 <= point[1] <= self.bbox[3] + 1000)
        else:
            return False


@dataclass
class Action:
    action: dict
    start_time: datetime
    end_time: datetime

    def to_description(self):
        action_to_print = copy.deepcopy(self.action)
        if "arg2" in action_to_print:
            action_to_print.pop("arg2")
        if self.action["type"] == "remote_converse":
            action_to_print.pop("arg1")
        return f"{self.start_time.strftime('%H:%M:%S')} - {self.end_time.strftime('%H:%M:%S') if self.end_time else ''}: {action_to_print}"

    def judge_continue(self, current_plan):
        if self.action["type"] == "remote_converse" and current_plan["type"] == "remote_converse":
            return True
        return self.action == current_plan and self.action["type"] not in ["put", "pick"]


@dataclass
class Message:
    time: datetime
    subject: str
    content: str

    def to_description(self):
        return f"{self.time.strftime('%H:%M:%S')} {self.subject}: {self.content}"

import numpy as np

def pixel_to_world(u, v, z, w, h, fov, extrinsic=None):
    """
    Convert a pixel (u, v) and depth z to world coordinates (x, y, z),
    consistent with the C++ image_to_pcd() function.

    Parameters:
        u, v (int or float): Pixel coordinates (x=column, y=row)
        z (float): Depth value at that pixel
        w, h (int): Image width and height
        fov (float): Horizontal field of view in degrees
        extrinsic (array-like of length 12, optional): Flattened 3x4 extrinsic transform

    Returns:
        np.ndarray: 3D world coordinates [x, y, z]
    """

    # Compute intrinsics from FOV
    tan_fov = np.tan(np.deg2rad(fov) / 2) * w / h
    fx = w / 2.0 / tan_fov
    fy = h / 2.0 / tan_fov
    cx = w / 2.0
    cy = h / 2.0

    # Camera-space coordinates (match your C++ code)
    x = (u - cx) * z / fx
    y = (h - 1 - v - cy) * z / fy
    z = -z

    if extrinsic is not None:
        # extrinsic: [r00, r01, r02, t0, r10, r11, r12, t1, r20, r21, r22, t2]
        ex = np.asarray(extrinsic, dtype=np.float32)
        X = ex[0] * x + ex[1] * y + ex[2] * z + ex[3]
        Y = ex[4] * x + ex[5] * y + ex[6] * z + ex[7]
        Z = ex[8] * x + ex[9] * y + ex[10] * z + ex[11]
        return np.array([X, Y, Z])
    else:
        return np.array([x, y, z])



class ThinkingModule:
    def __init__(self, generator, logger, name, type="", ablate=""):
        self.logger = logger
        self.generator = generator
        self.task_decription = open(f"agents/sentinel_challenge/meeting_prompts/task_description.txt", "r").read()
        self.name = name
        self.ablate = ablate
        _subdir = {'cosar': 'cosar_prompts', 'roco': 'roco_prompts'}.get(type, 'roco_prompts')
        self.prompt_path = f"agents/sentinel_challenge/meeting_prompts/{_subdir}"

    def parse_json(self, prompt, response, last_call=False):
        json_str = None
        if "```json" in response:
            # Step 1: Extract the JSON part
            start = response.find("```json") + len("```json")
            end = response.find("```", start)
            json_str = response[start:end].strip()
        else:
            self.logger.warning(f"Error parsing JSON, the string was {response}")
            if not last_call:
                chat_history = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ]
                data = self.generator.generate(
                    f"The output format is wrong. Output the formatted json string enclosed in ```json``` only! Do not include any other character in the output!",
                    chat_history=chat_history)
                return self.parse_json(None, data, last_call=True)
            else:
                self.logger.error(f"Error parsing JSON, already last call, the string was {response}")
                return None

        # # Step 2: Clean up the JSON
        # # Replace single quotes with double quotes
        # # Safely evaluate the string to a Python dictionary
        # parsed_dict = ast.literal_eval(json_str)
        # # Convert the dictionary back to a JSON string
        # json_str = json.dumps(parsed_dict)

        # Step 3: Convert to dictionary
        try:
            response = json.loads(json_str)
        except json.JSONDecodeError as e:
            self.logger.warning(f"Error decoding JSON: {e}, the string was {json_str}")
            if not last_call:
                chat_history = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ]
                data = self.generator.generate(
                    f"The output format is wrong. Output the formatted json string enclosed in ```json``` only! Do not include any other character in the output!",
                    chat_history=chat_history)
                return self.parse_json(None, data, last_call=True)
        return response

class BaseNavigationMeetingAgent(Agent):
    def __init__(self, name, pose, info, sim_path, no_react=False, debug=False, logger=None,
                 lm_source='openai', lm_id='gpt-4o', max_tokens=4096, temperature=0, top_p=1.0, init_generator=True,
                 detect_interval=-1, num_agents=1, enable_danger_zone=False, ablate="", server_port=8000):
        super().__init__(name, pose, info, sim_path, no_react, debug, logger)
        self.looking_down = False
        self.num_agents = num_agents
        self.comm = self.num_agents > 1
        self.s_mem = SemanticMemory(os.path.join(self.storage_path, "semantic_memory"), detect_interval=detect_interval, debug=self.debug, logger=self.logger, knowledge_path=os.path.join(self.storage_path, "seed_knowledge.json"), enable_danger_zone=enable_danger_zone, detection_min_pixel_ratio=0.0005)
        os.makedirs(f"{self.storage_path}/episodic_memory", exist_ok=True)
        os.makedirs(f"{self.storage_path}/generated_waypoints", exist_ok=True)

        if init_generator:
            if lm_source in ["openai", "azure"]:
                self.generator = global_model_manager.get_generator(
                    lm_source, lm_id, max_tokens, temperature, top_p, logger
                )
            elif lm_source == "local_qwen":
                from agents.sentinel_challenge.qwen_local_generator import QwenLocalGenerator
                self.generator = QwenLocalGenerator(
                    lm_id=lm_id,
                    server_port=server_port,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    logger=self.logger,
                )
                self.logger.info(f"client port: {self.generator._mm_client.server_port}")
            else:
                raise ValueError(f"Unsupported lm_source: {lm_source}")
        else:
            self.generator = None

        self.end_time = None

        self.action_history: list[Action] = [] # not used so far
        self.current_plan = None
        self.plan_start_time = None
        self.conversation_history: list[Message] = []
        self.event_history: list[Message] = []
        self.app_message_history: list[Message] = []
        self.meeting_place = None # meeting place for discussion
        self.goal_place = None # goal place for navigation
        self.mode = None
        self.banned = False
        self.places_buffer = []
        self.ablate = ablate
        # Discussion
        self.mode_time_counter = 0
        self.discussion_trigger = ""
        self.discussion_plan = None
        self.agent_opinions = dict()
        self.known_poses = dict()
        self.known_eta = dict()
        self.eta_history = dict()
        self.collect_plan = None
        self.thinking = 0
        # Navigation
        self.last_estimated_arrival_time = None
        self.last_estimated_move_time = None
        self.last_route = Route() # waypoints, city level
        self.last_nav = [] # waypoints, local level
        self.last_action = None
        self.route_history = {"last_route": dict(), "last_nav": dict()}
        self.route_plan = list() # not using
        # Reasoning
        self.known_sentinel_poses = list()
        self.query_buffer = list()
        self.finding_route = 0
        self.grid_map = None

    def reset(self, name, pose):
        super().reset(name, pose)
        self.curr_time = datetime.strptime(self.scratch['curr_time'], "%B %d, %Y, %H:%M:%S") if self.scratch['curr_time'] is not None else None
        self.s_mem = SemanticMemory(os.path.join(self.storage_path, "semantic_memory"), debug=self.debug, logger=self.logger)
        self.meeting_place = None
        self.goal_place = None

    def _process_obs(self, obs):
        # if obs['action_status'] == "FAIL": commented because we don't use action_history, to prevent unknown behavior
        #     self.logger.info(f"{self.name} failed to execute last action {self.action_history[-1].action}.")
        start = time.time()
        num_new_objects = self.s_mem.update(obs, last_nav=self.last_nav)
        self.logger.debug(f"Update Semantic Memory: {start}, {time.time()}")
        self.curr_time = obs['curr_time']
        self.held_objects = obs['held_objects']
        self.current_place = obs['current_place']
        self.obs = obs
        self.curr_events = []

        # react to new objects
        if not self.no_react and num_new_objects > 0:
            new_objects = self.s_mem.object_builder.get_new_objects()
            curr_objects = self.s_mem.object_builder.get_curr_objects()
            kws = [object.name for object in curr_objects]
            img_path = os.path.join(self.storage_path, 'episodic_memory',
                                    f'img_{self.curr_time.strftime("%B %d, %Y, %H:%M:%S")}.png')
            Image.fromarray(obs['rgb']).save(img_path)
            if "gt_seg_idxc_to_info" in obs:
                desc = f"I see {', '.join([object.name for object in curr_objects])}."
            else:
                desc = self.generate_captioning(
                    f"Here's an image including {', '.join([object.name for object in new_objects])}. Describe what you see in one sentence. Start with 'I see'.",
                    img=img_path)
                desc += f" Entities detected: {', '.join([object.name for object in curr_objects])}."
            for object in new_objects:
                if 'sentinel' in object.name or 'white_humanoid' in object.name:
                    self.logger.info(f"There is a sentinel or humanoid in the view. its idx is {object.idx}")
                    self.s_mem.warning_labels.append(object.idx)
                # if 'person' in object.name:
                #     prompt = "You are analyzing an image from a simulation environment. Determine whether the depicted figure is a white humanoid model or a human model. Focus on visual cues such as: 1. Surface material (e.g., uniform matte or glossy white plastic vs. textured skin. The humanoid has a uniformly bright white skin) 2. Color consistency (pure or near-uniform white vs. natural skin tones or varied coloration) 3. Facial detail (simplified or stylized features vs. realistic eyes, skin texture, and expressions) 4. Joints and seams (visible mechanical joints or panel lines vs. organic anatomy) 5. Clothing and accessories (integrated body shell vs. separate wearable clothing) Is there a such humanoid in the image? Output only 'Yes' or 'No'. Don't include other words."
                #     is_sentinel = self.generate_captioning(prompt, img=img_path)
                #     if is_sentinel.lower() == 'yes':
                #         self.s_mem.warning_labels.append(object.idx)
                #         self.logger.info(f"There is a white humanoid in the view. its idx is {object.idx}")
            self.last_react_time = self.curr_time
            self.logger.debug(f"reacting to new objects: {desc}")
            self.add_event("observation", self.curr_time, self.pose[:3], obs['current_place'], kws, img_path, desc, None)
        # saving routes
        if self.obs['steps']%10==0:
            self.route_history['last_route'][self.obs['steps']]=copy.deepcopy(self.last_route.to_dict())
            self.route_history['last_nav'][self.obs['steps']]=copy.deepcopy(self.last_nav)
            json.dump(self.route_history, open(os.path.join(self.storage_path, "route_history.json"), "w"))

    def process_obs_with_sptial_knowledge(self, obs, enable_refine=False):
        # react to events
        if len(obs['events']) > 0:
            for event in obs['events']:
                if event["type"] == "speech":
                    self.conversation_history.append(Message(self.curr_time, event["subject"], event["content"]))
                    self.discuss_process_speech(obs)
                if event["type"] == "app message":
                    if event["subject"] != self.name:
                        continue
                    if self.last_action['type']=="query_map_tool":
                        if self.last_action['arg1']=="query_route":
                            if event['content'] is None:
                                time_to_arrival = timedelta(hours=23, minutes=59, seconds=59)
                            else:
                                time_to_arrival = timedelta(seconds=int(event['content'].calc_time(pose=self.get_outdoor_pose())))
                            if self.goal_place==self.last_action["arg2"]:
                                self.last_route=event["content"]
                                self.last_estimated_arrival_time = self.curr_time + time_to_arrival
                            self.app_message_history.append(Message(self.curr_time, event["subject"], f"The estimated time from current pose to {self.last_action['arg2']} is {time_to_arrival}s"))
                            if enable_refine==False or len(event["content"])==0:
                                self.update_known_eta(
                                    {
                                        self.last_action['arg2']:
                                        {
                                            self.name: str(time_to_arrival)
                                        }
                                    })
                            else:
                                route_validity, danger = self.spatial_resoner.check_waypoint_validity(self.known_sentinel_poses, event["content"], excluding_wps=[self.pose[:2], event["content"][-1].location])
                                if not route_validity:
                                    self.refine_target_place = self.last_action["arg2"]
                                    if self.refine_target_place not in self.refine_retry:
                                        self.refine_retry[self.refine_target_place] = self.max_refine_retry
                                    if self.refine_retry[self.refine_target_place] >= 0:
                                        self.ready_to_refine = True
                                        self.refine_target_danger = danger
                                        self.refine_reference = [list(wp.location) for wp in event["content"]]
                                    else:
                                        self.ready_to_refine = False
                                if not self.ready_to_refine:
                                    self.update_known_eta(
                                        {
                                            self.last_action['arg2']:
                                            {
                                                self.name: str(time_to_arrival)
                                            }
                                        })
                        elif self.last_action["arg1"]=="query_place":
                            if event['content'] is not None:
                                self.s_mem.update_with_new_knowledge(event["content"])
                            else:
                                self.logger.warning(f"Get no information for {self.last_action['arg2']}")
                        elif self.last_action["arg1"]=="query_nearby":
                            self.places_buffer.extend(event['content'])
                        elif self.last_action["arg1"]=="query_grid_map":
                            self.grid_map = event['content']
                if event["type"] == "sentinel signal":
                    if event['content']['arg2'] != self.name: continue
                    if event['content']['arg1'] == 'ban':
                        self.logger.info("I'm being banned...")
                        self.banned = True
        # processing sentinels
        values, counts = np.unique(self.obs['segmentation'], return_counts=True)
        freq = dict(zip(values, counts))
        values_labels, counts_labels = np.unique(self.s_mem.current_labels, return_counts=True)
        freq_labels = dict(zip(values_labels, counts_labels))
        self.logger.info(f"segmentation is {freq}, labels are {freq_labels}")
        self.visible_sentinels = dict()
        if 'gt_seg_idxc_to_info' in self.obs:
            for i in freq:
                if i not in self.obs["gt_seg_idxc_to_info"]: continue
                if freq[i] < 30: continue
                e = self.obs["gt_seg_idxc_to_info"][i]
                if 'type' in e and e['type'] == 'avatar': # e[-1] is None
                    if 'Sentinel' not in e['name']: continue
                    wp = self.get_position_from_view(self.obs['segmentation'], i)
                    self.logger.info(f"I see {i}: {e['name']}. World coordinates for {e['name']} are {wp}.")
                    self.visible_sentinels[e['name']] = wp
                    self.update_known_sentinel_poses([wp[:3]], shared=0)
        else:
            for i in freq_labels:
                if i not in self.s_mem.warning_labels: continue
                if freq_labels[i] < 30: continue
                wp = self.get_position_from_view(self.s_mem.current_labels, i)
                self.logger.info(f"I see a sentinel. World coordinates for its idx {i} are {wp}.")
                self.visible_sentinels[f'Sentinel {i}'] = wp
                self.update_known_sentinel_poses([wp[:3]], shared=0)

    def get_position_from_view(self, segmentation, label):
        mask = (segmentation == label)
        # Get pixel coordinates where mask == True
        v_indices, u_indices = np.nonzero(mask)   # v = row (y), u = column (x)
        # Compute median pixel position
        u_median = np.median(u_indices)
        v_median = np.median(v_indices)
        # Compute median depth
        selected_depths = self.obs['depth'][mask]
        z_median = np.median(selected_depths)
        wp = pixel_to_world(u_median, v_median, z_median, segmentation.shape[0], segmentation.shape[1], self.s_mem.fov, self.obs['extrinsics'].flatten())
        return wp

    def add_event(self, event_type, event_time, event_position, event_place, event_keywords, event_img, event_description, event_text_ft, event_poignancy=None, event_expiration=None):
        event_id = str(len(self.curr_events))
        self.logger.debug(f"adding new event, {event_id}: {event_description}")
        this_experience = EventInstance(event_id, event_type, event_time, event_time, event_position, event_place, event_keywords, event_img, event_description, event_poignancy, event_expiration)
        self.curr_events.append(this_experience)

    def generate_captioning(self, prompt, img):
        if self.no_react:
            return "Do not revoke the llm in no react mode."
        response = self.generator.generate(prompt, img=img, json_mode=False)
        return response

    def enter_discussion_mode(self, trigger):
        self.mode = NavAgentState.DISCUSS
        self.mode_time_counter = 0
        self.discussion_trigger = trigger
        self.discussion_plan = None
        for agent in self.agent_opinions:
            self.agent_opinions[agent] = self.agent_opinions[agent] + ", in the previous round of communication."
        # known_poses persists across discussions — positions are slow to invalidate
        # and re-collecting them every round wastes a lot of conversation bandwidth.
        # known_eta is reset on navigation entry instead (see enter_navigation_mode).
        # self.conversation_history = []
        self.collect_plan = None
        self.thinking = 0

    def enter_navigation_mode(self, goal_place):
        self.mode = NavAgentState.NAVIGATE
        self.mode_time_counter = 0
        # ETAs to other candidate places stop being useful once a goal is committed.
        self.known_eta = dict()
        if self.goal_place != goal_place:
            self.goal_place = goal_place
            self.eta_history = dict()
            self.last_route = Route()
            self.last_nav = []

    def discuss_process_speech(self, obs):
        pass

    def navigate(self, sg, goal_pos, goal_bbox=None):
        if goal_pos is None:
            return None
        cur_trans = np.array(self.pose[:2])
        goal_bbox = get_bbox(goal_bbox, goal_pos)
        path = sg.volume_grid_builder.navigate(cur_trans, goal_bbox, self.last_path)
        self.last_path = None
        self.last_path_for_estimation = None
        if path is None:
            self.logger.error(f"No path found when navigate agent {self.name} to {goal_pos}.")
            return None
        else:
            if self.current_vehicle == "bicycle":
                nav_grid_num = int(self.BIKE_SPEED // sg.volume_grid_builder.conf.nav_grid_size)
            elif self.current_vehicle is None:
                nav_grid_num = int(self.WALK_SPEED // sg.volume_grid_builder.conf.nav_grid_size)
            else:
                self.logger.error(f"Unsupported vehicle type {self.current_vehicle} for navigation.")
                nav_grid_num = int(self.WALK_SPEED // sg.volume_grid_builder.conf.nav_grid_size)
            cur_goal = path[min(nav_grid_num, len(path) - 1)]
            if sg.volume_grid_builder.has_obstacle(get_axis_aligned_bbox(np.array([cur_goal, cur_trans]), None)):
                cur_goal = path[min(2, len(path) - 1)]
        
        self.logger.debug(f"Path {path[:3]}\n...\n{path[-3:]}")
        from agents.sg.builder.volume_grid import convex_hull, dist_to_hull
        dist = dist_to_hull(path[-1], convex_hull(goal_bbox))
        if dist > 2:
            self.logger.warning(f"Unable to find a path to the target bounding box. The optimal available path is still a distance of {dist} away from the target bounding box. The optimal path has been automatically adopted.")
        if self.action_status == "COLLIDE":
            self.logger.warning(f"{self.name} at {self.pose} moving to {cur_goal} is colliding with obstacles, path found was {path}.")
        # move
        target_rad = np.arctan2(cur_goal[1] - cur_trans[1], cur_goal[0] - cur_trans[0])
        delta_rad = target_rad - self.pose[-1]
        if delta_rad > np.pi:
            delta_rad -= 2 * np.pi
        elif delta_rad < -np.pi:
            delta_rad += 2 * np.pi
        self.logger.debug(f"Current pose is {list(self.pose)}. Current Goal is {list(cur_goal)}. Target_deg is {np.rad2deg(target_rad)}, while curr_deg is {np.rad2deg(self.pose[-1])}")

        self.last_path_for_estimation = path
        if delta_rad > np.deg2rad(15):
            action = {
                'type': 'turn_left',
                'arg1': np.rad2deg(delta_rad),
            }
            self.last_path = path
        elif delta_rad < -np.deg2rad(15):
            action = {
                'type': 'turn_right',
                'arg1': np.rad2deg(-delta_rad),
            }
            self.last_path = path
        else: action = {
            'type': 'move_forward',
            'arg1': np.linalg.norm(cur_goal - cur_trans),
        }
        if action['arg1'] < 0.1:
            self.logger.warning(f"{self.name} at {self.pose} moving to {cur_goal} is too close, path found was {path}.")
        return action
    
    def city_navigate(self, goal_place, threshold=500., requery=True):
        # If not having the goal_place in memory, query for it.
        if goal_place not in self.s_mem.get_places():
            return {"type": "query_map_tool", "arg1": "query_place", "arg2": goal_place}, False
        # Check if already arrived.
        cur_trans = np.array(self.pose[:2])
        if goal_place == self.obs['current_place'] or (goal_place in self.obs['accessible_places'] and self.s_mem.get_knowledge(goal_place)["building"]=="open space"):
            self.logger.debug(f"{self.name} arrived at {goal_place}.")
            return self.last_action, True
        # 4. Check current place.
        # can enter the correct place
        if goal_place in self.obs['accessible_places']:
            self.logger.debug(f"{self.name} finished navigation to {goal_place}")
            self.last_action = {
                'type': 'enter',
                'arg1': goal_place
            }
            return self.last_action, False
        # at wrong place, need to enter open space
        if self.obs['current_place'] is not None:
            self.logger.debug(
                f"{self.name} at {self.obs['current_place']} is entering open space to move to {goal_place}.")
            self.last_action = {
                'type': 'enter',
                'arg1': 'open space'
            }
            return self.last_action, False
        if self.last_route.impossible: return None, False
        if self.last_route.empty():
            action = {"type": "query_map_tool", "arg1": "query_route", "arg2": goal_place}
            return action, False
        assert isinstance(self.last_route, Route)
        assert isinstance(self.last_route[0], RouteNode)
        self.logger.info(f"Currently city nav to {goal_place}. The remaining route waypoints is {len(self.last_route)}. The estimated time till arrival is {timedelta(seconds=self.last_route.calc_time(pose=self.get_outdoor_pose()))}s")
        # If the estimated arrival time exceeds, regenerate
        if requery:
            estimated_arrival_time = self.curr_time + timedelta(seconds=self.last_route.calc_time(pose=self.get_outdoor_pose()))
            if self.last_estimated_arrival_time < self.curr_time:
                action = {"type": "query_map_tool", "arg1": "query_route", "arg2": goal_place}
                return action, False
        # throw away arrived waypoints
        idx = len(self.last_route)-1
        while idx > 0:
            idx -= 1
            arrived = is_near_goal(cur_trans[0], cur_trans[1], None, self.last_route[idx].location, threshold=5 if idx==len(self.last_route)-1 else 20)
            if arrived:
                for i in range(idx+1):
                    self.last_route.pop(0)
                break
        idx = 0
        # get occ map and check if the route will encounter any sentinels
        builder = self.s_mem.get_sg(place=self.current_place).volume_grid_builder
        occ_map, x_min, y_min, x_max, y_max = builder.get_occ_map() # occ map: 1 for unknow, 2 for obstacle, 3 for open
        while idx < len(self.last_route)-1:
            wp_x_world, wp_y_world = self.last_route[idx].location[0], self.last_route[idx].location[1]
            wp_pos_in_map = [
                builder.align_nav(wp_x_world) - x_min,
                builder.align_nav(wp_y_world) - y_min
            ]
            if 0 <= int(wp_pos_in_map[1]) < y_max - y_min and 0 <= int(wp_pos_in_map[0]) < x_max - x_min and occ_map[int(wp_pos_in_map[1])][int(wp_pos_in_map[0])] in [2, 4]:
                self.last_route.pop(0)
            else:
                break
        # avoid going for wp inside buildings (unknow) 
        idx = 0
        while idx < len(self.last_route):
            wp_x_world, wp_y_world = self.last_route[idx].location[0], self.last_route[idx].location[1]
            wp_pos_in_map = [
                builder.align_nav(wp_x_world) - x_min,
                builder.align_nav(wp_y_world) - y_min
            ]
            if 0 <= int(wp_pos_in_map[1]) < y_max - y_min and 0 <= int(wp_pos_in_map[0]) < x_max - x_min and occ_map[int(wp_pos_in_map[1])][int(wp_pos_in_map[0])] in [3]:
                break
            idx += 1
        if 0 < idx < len(self.last_route):
            self.navigate(self.s_mem.get_sg(), self.last_route[idx].location)
            time_to_reach_next_open_space_wp=len(self.last_path_for_estimation)
            self.navigate(self.s_mem.get_sg(), self.last_route[0].location)
            time_to_reach_next_wp=len(self.last_path_for_estimation)
            if time_to_reach_next_wp > time_to_reach_next_open_space_wp:
                while idx>0:
                    idx -= 1
                    self.last_route.pop(0)
        # If no intermediate waypoints left.
        if len(self.last_route)==1:
            self.last_nav = [self.last_route[0].location]
            return self.goto_place(goal_place)
        # navigate depending on current transit type.
        if self.last_route[0].transit=='walk':
            if self.obs['current_vehicle']=='bus':
                return {'type': 'exit_bus', 'arg1': None}, False
            else:
                return self.llm_navigate(max_retry=0)
        elif self.last_route[0].transit=='bus':
            if self.obs['current_vehicle']!='bus':
                if 'bus' in self.obs['accessible_places']:
                    return {'type': 'enter_bus', 'arg1': None}, False
                else:
                    return {'type': 'wait'}, False
            else:
                return {'type': 'wait'}, False
    
    def llm_navigate(self, max_retry = 3, threshold=200.):
        assert len(self.last_route)>0
        self.logger.debug(f"Current last_nav is {self.last_nav}")
        if not self.last_nav:
            self.generate_navigation_plan(max_retry=max_retry)
            self.last_estimated_move_time = self.curr_time + timedelta(seconds=self.calc_time(waypoints=self.last_nav))
        if self.last_estimated_move_time is None: self.last_estimated_move_time = self.curr_time + timedelta(seconds=self.calc_time(waypoints=self.last_nav))
        # If the estimated arrival time exceeds, regenerate
        estimated_move_time = self.curr_time + timedelta(seconds=self.calc_time(waypoints=self.last_nav))
        if self.last_estimated_move_time + timedelta(seconds=30) < estimated_move_time:
            self.generate_navigation_plan(max_retry)
            self.last_estimated_move_time = self.curr_time + timedelta(seconds=self.calc_time(waypoints=self.last_nav))
        # throw away arrived waypoints
        arrived = True
        curr_goal = None
        cur_trans = np.array(self.pose[:2])
        while arrived:
            if not self.last_nav:
                self.generate_navigation_plan(max_retry=max_retry)
                if not self.last_nav:
                    return {"type": "wait"}, False
                self.last_estimated_move_time = self.curr_time + timedelta(seconds=self.calc_time(waypoints=self.last_nav))
            curr_goal = self.last_nav[0]
            action = self.navigate(self.s_mem.get_sg(), curr_goal)
            arrived = is_near_goal(cur_trans[0], cur_trans[1], None, curr_goal, threshold=10)
            if arrived: self.last_nav.pop(0)
        return action, False
    
    def generate_navigation_plan(self, max_retry=3):
        return self.generate_navigation_plan_without_img(max_retry=0)

    def generate_navigation_plan_without_img(self, max_retry=3):
        assert max_retry >= 0
        if max_retry == 0:
            # Fallback: use first 3 waypoints from last_route
            nav_horizon = 0
            self.last_nav = []
            while nav_horizon < len(self.last_route) and nav_horizon < 3 and self.last_route[nav_horizon].transit == 'walk':
                self.last_nav.append(self.last_route[nav_horizon].location)
                nav_horizon += 1
            return
        # find nearest unexplored point
        builder = self.s_mem.get_sg(place=self.current_place).volume_grid_builder
        occ_map, x_min, y_min, x_max, y_max = builder.get_occ_map() # occ map: 1 for unknown, 2 for obstacle, 3 for open, 4 for warning
        agent_x_world, agent_y_world = self.pose[0], self.pose[1]
        agent_pos_in_map = [
            builder.align_nav(agent_x_world) - x_min,
            builder.align_nav(agent_y_world) - y_min
        ]
        occ_map[int(agent_pos_in_map[1])][int(agent_pos_in_map[0])]=5
        # Define local crop around agent (±30m in world coordinates)
        x_low_w, x_up_w = agent_x_world - 30, agent_x_world + 30
        y_low_w, y_up_w = agent_y_world - 30, agent_y_world + 30
        # Convert to map indices
        x_low = int(max(0, builder.align_nav(x_low_w) - x_min, builder.align_nav(-450) - x_min))
        x_up = int(min(occ_map.shape[1], builder.align_nav(x_up_w) - x_min, builder.align_nav(450) - x_min))
        y_low = int(max(0, builder.align_nav(y_low_w) - y_min, builder.align_nav(-450) - y_min))
        y_up = int(min(occ_map.shape[0], builder.align_nav(y_up_w) - y_min, builder.align_nav(450) - y_min))

        # Crop the occupancy map
        if x_low >= x_up or y_low >= y_up:
            raise ValueError("Invalid crop bounds after alignment.")

        cropped_map = occ_map[y_low:y_up, x_low:x_up]  # Note: y first, then x

        # Function to downscale using mode (most frequent value), handling borders
        def downscale_map(map_array, factor):
            h, w = map_array.shape
            # Trim to make dimensions divisible by factor
            new_h = (h // factor) * factor
            new_w = (w // factor) * factor
            trimmed = map_array[:new_h, :new_w]
            # Reshape into blocks
            reshaped = trimmed.reshape(new_h // factor, factor, new_w // factor, factor)
            # Move block axes to the end
            blocks = reshaped.swapaxes(1, 2).reshape(new_h // factor, new_w // factor, factor * factor)
            # Apply custom priority rule per block
            downscaled = np.zeros((new_h // factor, new_w // factor), dtype=np.int32)
            for i in range(blocks.shape[0]):
                for j in range(blocks.shape[1]):
                    block = blocks[i, j]
                    has_2 = 2 in block
                    has_3 = 3 in block
                    has_5 = 5 in block
                    
                    if has_2 and has_5:
                        raise ValueError(f"Block at ({i}, {j}) contains both 2 and 5, which is not allowed.")
                    elif has_5:
                        downscaled[i, j] = 5
                    elif has_2:
                        downscaled[i, j] = 2
                    elif has_3:
                        downscaled[i, j] = 3
                    else:
                        downscaled[i, j] = 1  # default to free (1)
            return downscaled

        try:
            downscaled_map = downscale_map(cropped_map, factor=4)
        except ValueError as e:
            self.logger.error(f"Downscaling failed: {e}")
            self.generate_navigation_plan(max_retry=max_retry - 1)
            return

        # Convert downscaled map to string representation
        symbol_map = {1: '?', 2: 'X', 3: '.', 5: 'A'}  # . = free, X = obstacle, ? = unknown, A = agent
        map_str_lines = []
        for row in downscaled_map:
            line = ''.join(symbol_map[val] for val in row)
            map_str_lines.append(line)
        map_str = '\n'.join(map_str_lines)

        # Convert last_route (list of [x, y] in world coords) to relative local coordinates (in cropped map, then in downscale grid)
        def world_to_downscaled_local(x_world, y_world):
            # Convert world → map index
            mx = builder.align_nav(x_world) - x_min
            my = builder.align_nav(y_world) - y_min
            # Convert to local in cropped map
            mx_local = mx - x_low
            my_local = my - y_low
            # Downscale by 4
            mx_ds = mx_local // 4
            my_ds = my_local // 4
            if 0 <= mx_ds < downscaled_map.shape[1] and 0 <= my_ds < downscaled_map.shape[0]:
                return [mx_ds, my_ds], True
            return [mx_ds, my_ds], False
        
        # Encode full route and goal
        path_local = []
        path_global = []
        for pt in self.last_route:
            loc, in_map = world_to_downscaled_local(pt[0], pt[1])
            if in_map:
                path_local.append(loc)
            path_global.append(loc)

        route_local_str = " → ".join(f"({x},{y})" for x, y in path_local) if path_local else "None"
        route_global_str = " → ".join(f"({x},{y})" for x, y in path_global) if path_global else "None"
        
        if not path_global:
            goal_grid = "unknown"
        else:
            goal_grid = path_global[-1]  # Last waypoint in grid coords
        route_global_str += " (goal)"

        prompt=open("agents/sentinel_challenge/meeting_prompts/navigation_plan.txt","r").read()
        # Format the prompt
        prompt = prompt.replace(
            "$map$", map_str
        )
        prompt = prompt.replace(
            "$route_local$", route_local_str
        )
        prompt = prompt.replace(
            "$route_global$", route_global_str
        )
        prompt = prompt.replace(
            "$goal_grid$", str(goal_grid) if isinstance(goal_grid, list) else "unknown"
        )
        self.logger.debug(f"navigating_prompt: {prompt}")
        response = self.generator.generate(prompt, img=None, json_mode=False)
        try:
            response = self.generator.generate(prompt, img=None, json_mode=False)
            response_dict = self.parse_json(prompt, response)
            self.logger.debug(f"generated response: {response_dict}")
            waypoints = response_dict.get("waypoints", [])
            if not isinstance(waypoints, list) or len(waypoints) == 0:
                raise ValueError("Waypoints must be a non-empty list.")
            self.last_nav = self.grid_to_world(waypoints, x_low=x_low, y_low=y_low, x_min=x_min, y_min=y_min, resolution_factor=4)
            if self.debug:
                self.visualize_navigation_plan(
                    downscaled_map=downscaled_map,
                    original_waypoints=[world_to_downscaled_local(pt[0], pt[1])[0] for pt in self.last_route] if hasattr(self, 'last_route') and isinstance(self.last_route, list) else [],
                    waypoints=[world_to_downscaled_local(pt[0], pt[1])[0] for pt in self.last_nav] if hasattr(self, 'last_nav') and isinstance(self.last_nav, list) else [],
                    save_path=f"{self.storage_path}/generated_waypoints/navigation_plan_{self.steps:06d}.png"
                )
        except Exception as e:
            self.logger.error(
                f"Error generating navigation plan: {e} with traceback: {traceback.format_exc()}. Response was: {response}"
            )
            self.generate_navigation_plan(max_retry=max_retry - 1)

    def grid_to_world(self, downscaled_waypoints, x_low, y_low, x_min, y_min, resolution_factor=4):
        """
        Convert waypoints from downscaled grid coordinates to real-world world coordinates.

        Args:
            downscaled_waypoints: List of [x_ds, y_ds], where (x_ds, y_ds) are in the 
                                downscaled, cropped map grid (post-4x downscale).
            x_low, y_low: The lower-left corner (in full map index space) of the cropped region.
                        These are the same values used in generate_navigation_plan.
            resolution_factor: The downscale factor (default 4).

        Returns:
            List of [x_world, y_world] in real-world coordinates (meters).
        """
        builder = self.s_mem.get_sg(place=self.current_place).volume_grid_builder

        world_coords = []
        for x_ds, y_ds in downscaled_waypoints:
            # Step 1: Convert from downscaled grid → cropped high-res map coordinates
            x_cropped = x_ds * resolution_factor + resolution_factor // 2  # center of the block
            y_cropped = y_ds * resolution_factor + resolution_factor // 2

            # Step 2: Convert from cropped map → full map coordinates
            x_full = x_cropped + x_low
            y_full = y_cropped + y_low

            # Step 3: Convert from full map index → world coordinates
            # Recall: map_index = (aligned_world_coord - origin)
            x_world = builder.align_nav_inv(x_full + x_min)  # because x_low was in (map_x - x_min)
            y_world = builder.align_nav_inv(y_full + y_min)

            world_coords.append([x_world, y_world])

        return world_coords

    def visualize_navigation_plan(self, downscaled_map, original_waypoints, waypoints, save_path=None):
        """
        Visualize the downscaled occupancy map with navigation waypoints.

        Args:
            downscaled_map: 2D numpy array (H, W) with values 1=free, 2=obstacle, 3=unknown, 4=agent
            waypoints: List of [x, y] grid coordinates (in downscaled map space)
            agent_grid_pos: [x, y] position of agent in the same grid (downscaled)
            save_path: If provided, save the image to this path
        """
        h, w = downscaled_map.shape

        # Create RGB canvas
        vis_map = np.zeros((h, w, 3), dtype=np.uint8)

        # Color mapping
        vis_map[downscaled_map == 1] = [100, 100, 100]  # free space - gray
        vis_map[downscaled_map == 2] = [255, 255, 255]  # obstacle - white
        vis_map[downscaled_map == 3] = [0, 0, 0]      # unknown - black
        vis_map[downscaled_map == 4] = [0, 255, 0]      # agent - green

        # Convert to float32 for OpenCV drawing
        vis_map = vis_map.astype(np.uint8)

        # Draw waypoints and connect them
        if waypoints and len(original_waypoints) > 0:
            pts = []
            for wp in waypoints:
                x, y = int(wp[0]), int(wp[1])
                if 0 <= x < w and 0 <= y < h:
                    pts.append((x, y))
                    cv2.circle(vis_map, (x, y), radius=1, color=(255, 255, 0), thickness=-1)  # cyan dot

            # Draw lines connecting waypoints
            for i in range(len(pts) - 1):
                cv2.line(vis_map, pts[i], pts[i+1], color=(255, 255, 0), thickness=1)
        if waypoints and len(waypoints) > 0:
            pts = []
            for wp in waypoints:
                x, y = int(wp[0]), int(wp[1])
                if 0 <= x < w and 0 <= y < h:
                    pts.append((x, y))
                    cv2.circle(vis_map, (x, y), radius=1, color=(255, 0, 0), thickness=-1)  # blue dot

            # Draw lines connecting waypoints
            for i in range(len(pts) - 1):
                cv2.line(vis_map, pts[i], pts[i+1], color=(255, 0, 0), thickness=1)

        # Optionally add a scale indicator
        # cv2.putText(vis_map, 'Red: Plan', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Resize for better visibility (scale up 4x)
        vis_map = cv2.resize(vis_map, (w * 4, h * 4), interpolation=cv2.INTER_NEAREST)

        if save_path is not None:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            cv2.imwrite(save_path, vis_map)


    def calc_time(self, waypoints):
        '''
        Here waypoints is a list of xy-coordinates.
        '''
        waypoints=waypoints
        if self.current_place is None:
            ret=np.linalg.norm(np.array(waypoints[0][:2])-np.array(self.pose[:2]))
        else:
            ret=0.
        for i in range(1, len(waypoints)):
            ret+=np.linalg.norm(np.array(waypoints[i][:2])-np.array(waypoints[i-1][:2]))
        return ret*2 # for turning
    
    def get_meeting_target(self):
        # use this function to get geometric center
        meeting_target = np.zeros(2, dtype=float)
        for agent in self.obs['agent_pos_dict']:
            meeting_target += np.array(self.obs['agent_pos_dict'][agent]['pose'][:2])
        meeting_target /= len(self.obs['agent_pos_dict'])
        return meeting_target
    
    def get_meeting_place(self):
        place = self.get_nearest_places(self.get_meeting_target())[0][1]
        return place

    def get_nearest_places(self, target):
        place_list = []
        for place in self.s_mem.get_places():
            goal_place_dict = self.s_mem.get_knowledge(place)
            if goal_place_dict is None:
                self.logger.error(f"No knowledge found for {place}.")
                return None, False
            goal_pos = np.array([goal_place_dict["location"][0], goal_place_dict["location"][1]])
            if goal_place_dict["building"] != "open space":
                goal_pos[0], goal_pos[1] = goal_pos[0] - 1000, goal_pos[1] - 1000
            if goal_pos[0] > 500: # also a hack
                goal_pos[0], goal_pos[1] = goal_pos[0] - 1000, goal_pos[1] - 1000
            goal_bbox = goal_place_dict["bounding_box"]
            place_list.append((np.linalg.norm(np.array(target)-goal_pos),place))
        place_list = sorted(place_list)
        place_list = place_list[:15] if len(place_list)>15 else place_list
        return place_list
    
    def parse_json(self, prompt, response, last_call=False):
        json_str = None
        if "```json" in response:
            # Step 1: Extract the JSON part
            start = response.find("```json") + len("```json")
            end = response.find("```", start)
            json_str = response[start:end].strip()
        else:
            self.logger.warning(f"Error parsing JSON, the string was {response}")
            if not last_call:
                chat_history = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ]
                data = self.generator.generate(
                    f"The output format is wrong. Output the formatted json string enclosed in ```json``` only! Do not include any other character in the output!",
                    chat_history=chat_history)
                return self.parse_json(None, data, last_call=True)
            else:
                self.logger.error(f"Error parsing JSON, already last call, the string was {response}")
                return None

        # # Step 2: Clean up the JSON
        # # Replace single quotes with double quotes
        # # Safely evaluate the string to a Python dictionary
        # parsed_dict = ast.literal_eval(json_str)
        # # Convert the dictionary back to a JSON string
        # json_str = json.dumps(parsed_dict)

        # Step 3: Convert to dictionary
        try:
            response = json.loads(json_str)
        except json.JSONDecodeError as e:
            self.logger.warning(f"Error decoding JSON: {e}, the string was {json_str}")
            if not last_call:
                chat_history = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ]
                data = self.generator.generate(
                    f"The output format is wrong. Output the formatted json string enclosed in ```json``` only! Do not include any other character in the output!",
                    chat_history=chat_history)
                return self.parse_json(None, data, last_call=True)
        return response

    def search_nearby(self, source=None):
        '''
            search_range: [x_min, x_max, y_min, y_max]
        '''
        search_range = None
        self.logger.debug(f"Searching {self.search_target}")
        if source is not None:
            if type(source) is dict:
                search_range = [source['region']['x_min'], source['region']['x_max'],
                                source['region']['y_min'], source['region']['y_max']]
            elif type(source) is str and source in self.s_mem.get_places():
                knowledge = self.s_mem.get_knowledge(source)
                bbox = knowledge.get('bounding_box', None)
                if bbox is not None:
                    bbox = bbox_center_to_corners_repr(bbox)
                    bbox = irregular_to_regular_bbox(bbox)
                    search_range = [np.min(bbox[:, 0]), np.max(bbox[:, 0]),
                                    np.min(bbox[:, 1]), np.max(bbox[:, 1])]
        if not self.looking_down:
            self.looking_down = True
            return {'type': 'look_down'}
        reach_target_distance = 2. if self.current_place is None else 1.
        if self.search_target is not None:
            if self.search_target[0] - reach_target_distance < self.pose[0] < self.search_target[0] + reach_target_distance and \
                    self.search_target[1] - reach_target_distance < self.pose[1] < self.search_target[1] + reach_target_distance:
                # search target has been reached
                self.search_target = None
            elif search_range is not None and (not search_range[0] < self.search_target[0] < search_range[1] or not \
                    search_range[2] < self.search_target[1] < search_range[3]):
                # search target is out of the box
                self.search_target = None
            else:
                return self.navigate(self.s_mem.get_sg(self.current_place), self.search_target)

        builder = self.s_mem.get_sg(place=self.current_place).volume_grid_builder

        if self.current_place is None:
            # find nearest unexplored point
            occ_map, x_min, y_min, x_max, y_max = builder.get_occ_map()
            agent_pos_in_map = [builder.align_nav(self.pose[0]) - x_min,
                                builder.align_nav(self.pose[1]) - y_min]
            rows, cols = np.where(occ_map == 1)
            dists = np.sqrt((rows - agent_pos_in_map[0]) ** 2 + (cols - agent_pos_in_map[1]) ** 2)
            xs = [(row + x_min) * builder.conf.nav_grid_size for row in rows]
            ys = [(col + y_min) * builder.conf.nav_grid_size for col in cols]
            order = np.argsort(dists)
            sorted_rows = rows[order]
            sorted_cols = cols[order]

            if search_range is not None:
                mask = []
                bbox_x_min, bbox_x_max, bbox_y_min, bbox_y_max = search_range
                for x, y in zip(xs, ys):
                    mask.append(bbox_x_min <= x <= bbox_x_max and bbox_y_min <= y <= bbox_y_max)
                self.logger.debug(f"Search mask bbox {[bbox_x_min, bbox_x_max, bbox_y_min, bbox_y_max]}")
                self.logger.debug(f"Search mask size {sum(mask)}")
                if sum(mask) > 0:
                    sorted_rows = sorted_rows[mask]
                    sorted_cols = sorted_cols[mask]

            sorted_rows = sorted_rows[:100]
            sorted_cols = sorted_cols[:100]
            positions = list(zip(sorted_rows, sorted_cols))
            chosen_position = None
            if len(positions) > 0:
                row, col = random.choice(positions)
                x = (row + x_min) * builder.conf.nav_grid_size
                y = (col + y_min) * builder.conf.nav_grid_size
                chosen_position = [x, y]
            else:
                self.logger.error("Can not find a search target!")
            self.search_target = chosen_position
        else:
            self.search_target = None
        if self.search_target is not None:
            return self.navigate(self.s_mem.get_sg(self.current_place), self.search_target)
        else:
            return {"type": "turn_left",
                    "arg1": 90}
        
    def get_outdoor_pose(self):
        if self.current_place == None:
            return self.pose[:2]
        goal_place_dict = self.s_mem.get_knowledge(self.current_place)
        if goal_place_dict is None:
            self.logger.error(f"No knowledge found for {self.current_place}.")
            return self.pose[:2]
        goal_pos = np.array([goal_place_dict["location"][0], goal_place_dict["location"][1]])
        if goal_place_dict["building"] != "open space":
            goal_pos[0], goal_pos[1] = goal_pos[0] - 1000, goal_pos[1] - 1000
        return goal_pos[:2]
        
    def get_outdoor_pose_description(self):
        if self.current_place == None:
            return str(self.pose[:2])
        goal_place_dict = self.s_mem.get_knowledge(self.current_place)
        if goal_place_dict is None:
            self.logger.error(f"No knowledge found for {self.current_place}.")
            return ""
        goal_pos = np.array([goal_place_dict["location"][0], goal_place_dict["location"][1]])
        if goal_place_dict["building"] != "open space":
            goal_pos[0], goal_pos[1] = goal_pos[0] - 1000, goal_pos[1] - 1000
        return str(goal_pos[:2])
        
    def update_known_poses(self, new_poses):
        for agent in new_poses:
            if any([type(x) not in [float, int] for x in new_poses[agent]]): continue
            self.known_poses[agent]=new_poses[agent]
        
    def update_known_eta(self, new_eta):
        for place in new_eta:
            place_name = place
            if place_name.startswith("<") and place_name.endswith(">"):
                    place_name = place_name[1:-1]
            if place_name not in self.known_eta:
                self.known_eta[place_name]=dict()
            for agent in new_eta[place]:
                self.known_eta[place_name][agent]=new_eta[place][agent]

    def update_known_sentinel_poses(self, new_sentinel_poses, shared=0):
        for sentinel_pose in new_sentinel_poses:
            sentinel_pose = list(sentinel_pose)
            assert len(sentinel_pose)==3
            sentinel_pose.append(shared)
            flag = True
            for idx, known_sentinel_pose in enumerate(self.known_sentinel_poses):
                if np.linalg.norm(np.array(sentinel_pose[:2]) - np.array(known_sentinel_pose[:2])) < 10.:
                    flag = False
                    if shared == 1:
                        self.known_sentinel_poses[idx][2] = 1
            if flag:
                self.known_sentinel_poses.append(list(sentinel_pose))
                if self.current_place is None:
                    self.s_mem.get_sg('open space').add_danger_zone(list(sentinel_pose[:3]), 1, self.obs['extrinsics']) # use 1 will not affect the effect i guess?

    def get_nearest_places_description(self, target):
        place_list = self.get_nearest_places(target)
        places_description = ""
        for dis, place in place_list:
            goal_place_dict = self.s_mem.get_knowledge(place)
            if goal_place_dict is None:
                self.logger.error(f"No knowledge found for {place}.")
                return None, False
            goal_pos = np.array([goal_place_dict["location"][0], goal_place_dict["location"][1]])
            if goal_place_dict["building"] != "open space":
                goal_pos[0], goal_pos[1] = goal_pos[0] - 1000, goal_pos[1] - 1000
            if goal_pos[0] > 500: # also a hack
                goal_pos[0], goal_pos[1] = goal_pos[0] - 1000, goal_pos[1] - 1000
            goal_bbox = goal_place_dict["bounding_box"]
            places_description += f"<{place}>: location {goal_pos}\n" if "spatial_memory" not in self.ablate else f"<{place}>\n"
        return places_description

    def get_place_outdoor_pose(self, place):
        """return a (2,) list or None"""
        goal_place_dict = self.s_mem.get_knowledge(place)
        if goal_place_dict is None:
            self.logger.error(f"No knowledge found for {place}.")
            return None
        goal_pos = np.array([goal_place_dict["location"][0], goal_place_dict["location"][1]])
        if goal_place_dict["building"] != "open space":
            goal_pos[0], goal_pos[1] = goal_pos[0] - 1000, goal_pos[1] - 1000
        if goal_pos[0] > 500: # also a hack
            goal_pos[0], goal_pos[1] = goal_pos[0] - 1000, goal_pos[1] - 1000
        goal_bbox = goal_place_dict["bounding_box"]
        return goal_pos[:2]

    def get_previous_actions_description(self):
        if len(self.action_history) == 0:
            return "None"
        else:
            action_list = self.action_history[-10:] if len(self.action_history) > 10 else self.action_history
            return "\n".join([action.to_description() for action in action_list])

    def get_known_poses_description(self):
        if len(self.known_poses) == 0:
            return "None"
        else:
            return "\n".join(f"{agent_name} is at {self.known_poses[agent_name][0]:.2f}, {self.known_poses[agent_name][1]:.2f}" for agent_name in self.known_poses)

    def get_known_eta_description(self):
        if len(self.known_eta) == 0:
            return "None"
        else:
            return "\n".join(f"{agent_name}'s ETA to <{place_name}> is {self.known_eta[place_name][agent_name]}." for place_name in self.known_eta for agent_name in self.known_eta[place_name])

    def get_known_sentinel_poses_description(self):
        if len(self.known_sentinel_poses) == 0:
            return "None"
        else:
            return "\n".join(f"{sentinel_pos[0]:.2f}, {sentinel_pos[1]:.2f}" for sentinel_pos in self.known_sentinel_poses)

    def get_eta_history_description(self):
        if len(self.eta_history) == 0:
            return "None"
        else:
            return "\n".join([f"ETA at time {key} is: {self.eta_history[key]}s remains to reach destination. Estimated arrival time is {(datetime.strptime(key, '%H:%M:%S')+timedelta(**dict(zip(['hours', 'minutes', 'seconds'],map(int, self.eta_history[key].split(':')))))).strftime('%H:%M:%S')}." for key in self.eta_history])

    def get_agent_opinions_description(self):
        if len(self.agent_opinions) == 0:
            return "None"
        else:
            return "\n".join([f"{key}: {self.agent_opinions[key]}" for key in self.agent_opinions])
        
    def get_agent_poses_description(self):
        agent_pos_dict=copy.copy(self.obs["agent_pos_dict"])
        agent_pos_description = ""
        for agent in agent_pos_dict:
            if agent_pos_dict[agent]['place'] is not None:
                agent_pos_dict[agent]['pose'][0], agent_pos_dict[agent]['pose'][1] = agent_pos_dict[agent]['pose'][0]-1000, agent_pos_dict[agent]['pose'][1]-1000
            agent_pos_description += f"{agent} is now in {agent_pos_dict[agent]['place'] if agent_pos_dict[agent]['place'] is not None else 'open space'}, with coordinate {agent_pos_dict[agent]['pose']}.\n"
        agent_pos_description.strip("\n")
        return agent_pos_description

    def get_conversation_description(self, limit=20):
        if len(self.conversation_history) == 0:
            return "None"
        conversation_list = self.conversation_history[-limit:] if len(self.conversation_history) > limit else self.conversation_history
        return "\n".join([chat.to_description() for chat in conversation_list])

    def get_past_event_description(self):
        if len(self.event_history) == 0:
            return "None"
        event_list = self.event_history[-20:] if len(self.event_history) > 20 else self.event_history
        return "\n".join([event.to_description() for event in event_list])

    def get_app_message_description(self):
        if len(self.app_message_history) == 0:
            return "None"
        app_message_list = [app_message for app_message in self.app_message_history if app_message.time + timedelta(minutes=2) > self.curr_time]
        return "\n".join([app_message.to_description() for app_message in app_message_list])

    def goto(self, target, force=False):
        if target.startswith("task_"):
            # it is a region
            _, task_idx, source, _ = target.split("_")
            task_idx = int(task_idx)
            if source == "source":
                target = self.task_manager.tasks[task_idx].source[0]
            else:
                target = self.task_manager.tasks[task_idx].destination[0]
            return self.goto_region(target['region'], force=force)
        if type(target) is str:
            # it is a room or agent
            if target in self.get_agent_list():
                return self.goto_agent(target, force=force)
            else:
                return self.goto_place(target, force=force)
        else:
            # it is a region
            return self.goto_region(target['region'], force=force)
    
    def get_agent_list(self):
        return self.s_mem.agents

    def goto_place(self, target_place: str, force=False) -> (dict, bool):
        places = self.s_mem.get_places() + ['open space']
        if target_place is None:
            target_place = 'open space'
            self.logger.debug(
                f"{self.name} at {self.obs['current_place']} is entering open space.")
            self.last_action = {
                'type': 'enter',
                'arg1': 'open space'
            }
            return self.last_action, True
        self.logger.info(f"Currently goto_place {target_place}.")
        if target_place not in places:
            self.logger.error(f"Target place {target_place} is not a valid place.")
            return None, False
        if force:
            self.last_action = {'type': 'force_enter', 'arg1': target_place}
            return self.last_action, True
        goal_place_dict = self.s_mem.get_knowledge(target_place)
        if goal_place_dict is None:
            self.logger.error(f"No knowledge found for {target_place}.")
            return None, False
        goal_pos = np.array([goal_place_dict["location"][0], goal_place_dict["location"][1]])
        if goal_place_dict["building"] != "open space":
            goal_pos[0], goal_pos[1] = goal_pos[0] - 1000, goal_pos[1] - 1000
        if goal_pos[0] > 500: # hack, there is open space place in memory > 10000
            goal_pos[0], goal_pos[1] = goal_pos[0] - 1000, goal_pos[1] - 1000
        goal_bbox = goal_place_dict["bounding_box"]
        self.logger.debug(f"Goal place: {target_place}, goal pos: {goal_pos}, goal bbox: {goal_bbox}")
        self.last_action = {'type': 'wait', 'arg1': None}
        # already at the correct place
        if target_place == self.obs['current_place']:
            self.logger.debug(f"{self.name} arrived at {target_place}.")
            return self.last_action, True
        # can enter the correct place
        if target_place in self.obs['accessible_places']:
            self.logger.debug(f"{self.name} finished navigation to {target_place} at {goal_pos}")
            self.last_action = {
                'type': 'enter',
                'arg1': target_place
            }
            return self.last_action, True
        # at wrong place, need to enter open space
        if self.obs['current_place'] is not None:
            self.logger.debug(
                f"{self.name} at {self.obs['current_place']} is entering open space to move to {target_place} at {goal_pos}.")
            self.last_action = {
                'type': 'enter',
                'arg1': 'open space'
            }
            return self.last_action, False
        # at open space, need to move to the correct place
        cur_trans = np.array(self.pose[:2])
        if is_near_goal(cur_trans[0], cur_trans[1], goal_bbox, goal_pos):
            self.logger.warning(
                f"{self.name} at {self.pose} is near the goal {goal_pos}, but not at the goal {target_place}.")
            self.logger.debug(f"{self.name}'s current accessible places are {self.obs['accessible_places']}")
            return {"type": "debug_action", "arg1": "query_accessible_places", "arg2": target_place}, True
        self.logger.debug(
            f"{self.name} at {tuple(int(p) for p in self.pose)} is moving to {target_place} at {tuple(int(g) for g in goal_pos)}.")
        start = time.time()
        self.last_action = self.navigate(self.s_mem.get_sg(self.current_place), goal_pos, goal_bbox)
        self.logger.debug(f"Navigate time: {start}, {time.time()}")
        return self.last_action, False

    def goto_region(self, target_region: dict, force=False):
        if self.current_place is not None:
            return {
                'type': 'enter',
                'arg1': 'open space'
            }, False
        target_pos = [(target_region['x_min'] + target_region['x_max']) / 2,
                      (target_region['y_min'] + target_region['y_max']) / 2]
        if force:
            return {
                'type': 'teleport',
                'arg1': target_pos
            }, True
        action = self.navigate(self.s_mem.get_sg(self.current_place), target_pos, goal_bbox=None)
        arrived = False
        if target_region['x_min'] < self.pose[0] < target_region['x_max'] and \
                target_region['y_min'] < self.pose[1] < target_region['y_max']:
            arrived = True
        return action, arrived
    
