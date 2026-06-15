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


class ReplayAgent(Agent):
    def __init__(self, name, pose, info, sim_path, no_react=False, debug=False, logger=None,
                 detect_interval=-1, replay_actions=None):
        super().__init__(name, pose, info, sim_path, no_react, debug, logger)
        self.looking_down = False
        self.s_mem = SemanticMemory(os.path.join(self.storage_path, "semantic_memory"), detect_interval=detect_interval, debug=self.debug, logger=self.logger, knowledge_path=os.path.join(self.storage_path, "seed_knowledge.json"))

        self.replay_actions = replay_actions if replay_actions is not None else []  # @ruxi fill this: action list read from the output

    def reset(self, name, pose):
        super().reset(name, pose)
        self.curr_time = datetime.strptime(self.scratch['curr_time'], "%B %d, %Y, %H:%M:%S") if self.scratch['curr_time'] is not None else None
        self.s_mem = SemanticMemory(os.path.join(self.storage_path, "semantic_memory"), debug=self.debug, logger=self.logger)

    def _process_obs(self, obs):
        pass

    def _act(self, obs):
        if self.replay_actions is None or len(self.replay_actions) == 0:
            return {'type': 'task_terminate'}

        if self.steps >= len(self.replay_actions):
            return {'type': 'task_terminate'}

        action = self.replay_actions[self.steps]

        return action
