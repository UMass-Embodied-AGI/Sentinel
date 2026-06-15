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
import math

from vico.agents.agent import Agent
from agents.memory import SemanticMemory
from agents.sg.builder.builder import Builder, BuilderConfig
from vico.tools.utils import *


class BaseSentinelAgent(Agent):
    def __init__(self, name, pose, info, sim_path, no_react=False, debug=False, logger=None,
                 detect_interval=-1, patrol_config=None):
        super().__init__(name, pose, info, sim_path, no_react, debug, logger)
        self.looking_down = False
        self.s_mem = SemanticMemory(os.path.join(self.storage_path, "semantic_memory"), detect_interval=detect_interval, debug=self.debug, logger=self.logger, knowledge_path=os.path.join(self.storage_path, "seed_knowledge.json"))

        self.spot_counter = dict()
        self.visible_agent = dict()
        self.countdown = 0
        self.current_target = None
        self.detection_count = 0

        self.patrol_config = patrol_config
        self.countdown_t = 15
        self.detection_min_pixel_ratio = 0.001
        self.max_speed = 0.2

    def reset(self, name, pose):
        super().reset(name, pose)
        self.curr_time = datetime.strptime(self.scratch['curr_time'], "%B %d, %Y, %H:%M:%S") if self.scratch['curr_time'] is not None else None
        self.s_mem = SemanticMemory(os.path.join(self.storage_path, "semantic_memory"), debug=self.debug, logger=self.logger)

    def _process_obs(self, obs):
        num_new_objects = self.s_mem.update(obs)
        self.curr_time = obs['curr_time']
        self.held_objects = obs['held_objects']
        self.current_place = obs['current_place']
        self.obs = obs
        tmp_arr=set(self.obs['segmentation'].flatten().tolist())
        values, counts = np.unique(self.obs['segmentation'], return_counts=True)
        freq = dict(zip(values, counts))
        self.visible_agent = {}
        for i in freq:
            if i not in self.obs["gt_seg_idxc_to_info"]: continue
            if freq[i] < self.detection_min_pixel_ratio * len(self.obs['segmentation'].flatten()): continue
            e = self.obs["gt_seg_idxc_to_info"][i]
            if 'type' in e and e['type'] == 'avatar': # e[-1] is None
                if 'Sentinel' in e['name']: continue
                if e['name'] not in self.obs['agent_pos_dict']: continue
                self.visible_agent[e['name']] = freq[i]
                if e['name'] not in self.spot_counter:
                    self.spot_counter[e['name']] = 0
                self.spot_counter[e['name']] += 1
                mask = (self.obs['segmentation'] == i)
                selected_depths = self.obs['depth'][mask]
                depth = np.median(selected_depths)
                self.logger.info(f"I see {e['name']}. its position is at {self.obs['agent_pos_dict'][e['name']]}, our distance is {np.linalg.norm(np.array(self.pose[:2]) - np.array(self.obs['agent_pos_dict'][e['name']]['pose'][:2]))} and the median depth is {depth}. The frequency is {freq[i]}")

    def _act(self, obs):
        self.logger.debug(f"current pose is {list(self.pose[:3])}")
        if len(self.visible_agent) == 0:
            self.set_target(None)
            action = self.patrol()
        else:
            if self.current_target is not None and self.current_target in self.visible_agent:
                self.countdown -= self.visible_agent[self.current_target]/len(self.obs['segmentation'].flatten())/self.detection_min_pixel_ratio
                if self.countdown > 0:
                    # action = {"type": "signal", "arg1": f"warning", "arg2": self.current_target}
                    action = self.chase(self.s_mem.get_sg(), goal_pos=self.obs['agent_pos_dict'][self.current_target]["pose"][:2])
                    if action['type']=='turn_left':
                        action['type']='look_after_left'
                    elif action['type']=='turn_right':
                        action['type']='look_after_right'
                    elif action['type']=='move_forward':
                        action['type']='chase'
                else:
                    action = {"type": "signal", "arg1": f"ban", "arg2": self.current_target}
                    self.set_target(None)
            else:
                for agent_name in self.visible_agent:
                    self.set_target(agent_name)
                    action = {"type": "signal", "arg1": f"warning", "arg2": self.current_target}
                    break
        
        self.last_action = action
        if self.current_target is not None:
            self.detection_count += 1
        if action['type'] == 'chase':
            action['arg1'] = min(action['arg1'], self.max_speed)
        self.logger.debug(f"current action is {self.last_action}")
        return self.last_action
    
    def set_target(self, agent_name):
        if agent_name is None:
            self.current_target = None
            self.countdown = 0
        else:
            self.current_target = agent_name
            self.countdown = self.countdown_t
    
    def patrol(self):
        if self.patrol_config is None:
            return None
        if self.patrol_config["type"] == "fixed":
            return {"type": "wait"}
        elif self.patrol_config["type"] == "rotating":
            return {"type": "turn_right", "arg1": 30}
        elif self.patrol_config["type"] == "patrolling":
            num_rep=0
            while is_near_goal(curr_x=self.pose[0], curr_y=self.pose[1], goal_bbox=None, goal_pos=self.patrol_config["route"][self.patrol_config["route_index"]]):
                self.patrol_config["route_index"]=(self.patrol_config["route_index"]+1)%(len(self.patrol_config["route"]))
                num_rep += 1
                if num_rep > len(self.patrol_config["route"]): break
            action = self.navigate(self.s_mem.get_sg(), goal_pos=self.patrol_config["route"][self.patrol_config["route_index"]], goal_bbox=None)
            return action

    def chase(self, sg, goal_pos, goal_bbox=None):
        if goal_pos is None:
            return None
        cur_trans = np.array(self.pose[:2])
        cur_goal = goal_pos
        target_rad = np.arctan2(cur_goal[1] - cur_trans[1], cur_goal[0] - cur_trans[0])
        delta_rad = target_rad - self.pose[-1]
        if delta_rad > np.pi:
            delta_rad -= 2 * np.pi
        elif delta_rad < -np.pi:
            delta_rad += 2 * np.pi
        self.logger.debug(f"Current pose is {list(self.pose)}. Current Goal is {list(cur_goal)}. Target_deg is {np.rad2deg(target_rad)}, while curr_deg is {np.rad2deg(self.pose[-1])}")

        if delta_rad > np.deg2rad(15):
            action = {
                'type': 'turn_left',
                'arg1': np.rad2deg(delta_rad),
            }
        elif delta_rad < -np.deg2rad(15):
            action = {
                'type': 'turn_right',
                'arg1': np.rad2deg(-delta_rad),
            }
        else: action = {
            'type': 'move_forward',
            'arg1': np.linalg.norm(cur_goal - cur_trans),
        }
        return action
    
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