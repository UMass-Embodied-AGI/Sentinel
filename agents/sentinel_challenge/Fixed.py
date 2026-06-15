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

from vico.agents.agent import Agent
from agents.memory import SemanticMemory
from agents.sentinel_challenge.base_nav import *
from agents.sg.builder.builder import Builder, BuilderConfig


class FixedMeetingAgent(BaseNavigationMeetingAgent):
    def __init__(self, name, pose, info, sim_path, no_react=False, debug=False, logger=None,
                 lm_source='openai', lm_id='gpt-4o', max_tokens=4096, temperature=0, top_p=1.0, init_generator=True,
                 detect_interval=-1, num_agents=1, enable_danger_zone=False):
        super().__init__(name, pose, info, sim_path, no_react, debug, logger, lm_source, lm_id, max_tokens, temperature, top_p, init_generator, detect_interval, num_agents, enable_danger_zone)

    def reset(self, name, pose):
        super().reset(name, pose)

    def _process_obs(self, obs):
        super()._process_obs(obs)
        self.process_obs_with_sptial_knowledge(obs)

    def _act(self, obs):
        if self.banned:
            if self.pose[0]>-1000:
                return {"type": "teleport", "arg1": [-1500., -1500.]}
            return {"type": "task_complete"}
        self.meeting_place = "The Greening of Detroit - Lafayette Greens"
        if self.s_mem.get_knowledge(self.meeting_place) is None:
            action = {'type': 'query_map_tool', 'arg1': 'query_place', 'arg2': self.meeting_place}
            self.last_action = action
            return action
        if self.mode == NavAgentState.DISCUSS:
            self.enter_navigation_mode()
        action, arrived = self.city_navigate(self.meeting_place)
        if arrived:
            action = {'type': 'task_complete'}
        self.mode_time_counter += 1
        self.last_action = action
        return action