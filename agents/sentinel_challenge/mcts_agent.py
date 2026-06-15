"""
MCTS-based agent for meeting challenge.
Uses Monte Carlo Tree Search to plan the next place to visit to minimize distance between agents.
"""
import ast
import os
import random
import copy
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import numpy as np
import math
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from vico.agents.agent import Agent
from agents.memory import SemanticMemory, EventInstance
from agents.sentinel_challenge.base_nav import *
from agents.sentinel_challenge.mcts_state import MCTSState
from agents.sentinel_challenge.mcts_planner import MCTSPlanner
from vico.modules.MapTool import Route, RouteNode
from vico.tools.utils import *
from tools.model_manager import global_model_manager


class MCTSMeetingAgent(BaseNavigationMeetingAgent):
    """
    Agent that uses MCTS to plan navigation to minimize distance between agents.
    """
    def __init__(self, name, pose, info, sim_path, no_react=False, debug=False, logger=None,
                 lm_source='openai', lm_id='gpt-4o', max_tokens=4096, temperature=0, top_p=1.0, init_generator=True,
                 detect_interval=-1, num_agents=1, enable_danger_zone=False, ablate="",
                 mcts_simulations=200, mcts_max_depth=3, mcts_exploration=5):
        """
        Initialize MCTS agent.
        
        Args:
            name: Agent name
            pose: Initial pose
            info: Agent info
            sim_path: Simulation path
            no_react: Whether to disable reactions
            debug: Debug mode
            logger: Logger instance
            lm_source: Language model source
            lm_id: Language model ID
            max_tokens: Max tokens for generation
            temperature: Temperature for generation
            top_p: Top-p for generation
            init_generator: Whether to initialize generator
            detect_interval: Detection interval
            num_agents: Number of agents
            enable_danger_zone: Enable danger zone detection
            ablate: Ablation string
            mcts_simulations: Number of MCTS simulations
            mcts_max_depth: Maximum depth for MCTS
            mcts_exploration: Exploration constant for MCTS
        """
        super().__init__(name, pose, info, sim_path, no_react, debug, logger, lm_source, lm_id, 
                        max_tokens, temperature, top_p, init_generator, detect_interval, num_agents, 
                        enable_danger_zone, ablate)
        
        # Initialize MCTS planner
        self.mcts_planner = MCTSPlanner(
            agent_name=name,
            num_simulations=mcts_simulations,
            max_depth=mcts_max_depth,
            exploration_constant=mcts_exploration,
            logger=logger
        )
        
        # Planning state
        self.planned_place = None
        self.planning_interval = 50  # Replan every N steps
        self.known_sentinel = dict()

        self.replan = False
        self.nearby_queried = False
        self.task_complete = False
    
    def reset(self, name, pose):
        """Reset agent state."""
        super().reset(name, pose)
        self.planned_place = None
    
    def _process_obs(self, obs):
        """Process observations."""
        super()._process_obs(obs)
        self.process_obs_with_sptial_knowledge(obs)

        # react to new danger
        self.danger_detected = False
        for sentinel in self.visible_sentinels:
            if sentinel not in self.known_sentinel:
                self.danger_detected = True
                if self.logger:
                    self.logger.info(f"{self.name} detected new sentinel: {sentinel}")
            self.known_sentinel[sentinel] = self.visible_sentinels[sentinel][:2]
    
    def _get_available_places(self) -> Tuple[List[str], Dict[str, List[float]]]:
        """
        Get list of available places and their locations.
        
        Returns:
            Tuple of (list of place names, dictionary mapping place names to locations)
        """
        places = self.s_mem.get_places()
        place_locations = {}
        building_locations = {}
        
        for place in places:
            place_dict = self.s_mem.get_knowledge(place)
            if place_dict is None:
                continue
            
            location = np.array([place_dict["location"][0], place_dict["location"][1]])
            if place_dict["building"] != "open space":
                location[0], location[1] = location[0] - 1000, location[1] - 1000
            
            if location[0] > 500:  # Hack for coordinate system
                location[0], location[1] = location[0] - 1000, location[1] - 1000
            
            if place_dict["building"] is not None:
                if place_dict["building"] not in building_locations:
                    building_locations[place_dict["building"]] = place
                    place_locations[place] = location[:2].tolist()
                if building_locations[place_dict["building"]] > place:
                    place_locations.pop(building_locations[place_dict["building"]], None)
                    building_locations[place_dict["building"]] = place
                    place_locations[place] = location[:2].tolist()
            else:
                place_locations[place] = location[:2].tolist()
        
        return places, place_locations
    
    def _get_agent_positions(self) -> Dict[str, List[float]]:
        """
        Get current positions of all agents.
        
        Returns:
            Dictionary mapping agent names to [x, y] positions
        """
        agent_positions = {}
        
        for agent_name, agent_info in self.obs['agent_pos_dict'].items():
            pose = None
            # Adjust coordinates if needed
            if agent_info.get('place') is not None:
                pose = self.get_place_outdoor_pose(agent_info.get('place'))

            if pose is None:
                pose = agent_info['pose'][:2]
            
            if pose[0] > 500:  # Hack for coordinate system
                pose = [pose[0] - 1000, pose[1] - 1000]
            
            agent_positions[agent_name] = pose
        
        return agent_positions
    
    def _plan_next_place(self) -> Optional[str]:
        """
        Use MCTS to plan the next place to visit.
        
        Returns:
            Place name to visit, or None
        """
        # Get available places and their locations
        available_places, place_locations = self._get_available_places()
        
        if not available_places:
            return None
        
        # Get current agent positions
        agent_positions = self._get_agent_positions()
        
        # Plan using MCTS
        deadline_seconds = 25 * 60
        self.logger.debug("=======================")
        self.logger.debug(f"agent positions: {agent_positions}")
        self.logger.debug("=======================")
        self.logger.debug(f"place locations: {place_locations}")
        self.logger.debug("=======================")
        next_place = self.mcts_planner.plan(
            agent_positions=copy.deepcopy(agent_positions),
            place_locations=place_locations,
            current_time_seconds=self.steps,
            deadline_seconds=deadline_seconds,
            sentinel_positions=self.known_sentinel.values()
        )
        
        if self.logger:
            self.logger.info(f"{self.name} MCTS planned next place: {next_place}")
            if agent_positions:
                max_dist = MCTSState(
                    agent_positions=copy.deepcopy(agent_positions),
                    current_agent=self.name,
                    current_place=self.current_place,
                    time=self.steps,
                    alive_agents={name: True for name in agent_positions},
                    cumulative_distance=0.0,
                    cumulative_detection=0.0,
                    depth=0,
                    logger=self.logger
                ).calculate_max_distance()
                self.logger.info(f"Current max distance between agents: {max_dist:.2f}")
        
        return next_place

    def check_meeting_condition(self):
        # Check meeting condition
        agent_positions = self._get_agent_positions()
        my_pos = np.array(agent_positions.get(self.name, [0, 0])[:2])

        all_close = True
        for agent_name, agent_pos in agent_positions.items():
            if agent_name == self.name:
                continue
            dist = np.linalg.norm(my_pos - np.array(agent_pos[:2]))
            if dist > 20.0:
                all_close = False
                break
        return all_close
    
    def _act(self, obs):
        """
        Receding-horizon MCTS control.
        Allows stopping, replanning, and changing waypoint anytime.
        """

        # -------------------------------------------------
        # 1. If captured
        # -------------------------------------------------
        if self.banned:
            if self.pose[0] > -1000:
                return {"type": "teleport", "arg1": [-1500., -1500.]}
            return {"type": "task_complete"}

        # -------------------------------------------------
        # 2. Check replanning triggers
        # -------------------------------------------------
        if self.replan == False:
            # No current plan
            if self.planned_place is None and self.task_complete == False:
                self.logger.debug(f"replan due to planned_place is None.")
                self.replan = True
                self.nearby_queried = False

            # Periodic replanning
            if self.mode_time_counter % self.planning_interval == 0:
                self.logger.debug(f"replan due to mode_time_counter.")
                self.replan = True
                self.nearby_queried = False

            # Danger zone trigger (if enabled)
            if hasattr(self, "danger_detected") and self.danger_detected:
                self.logger.debug(f"replan due to danger.")
                self.replan = True
                self.nearby_queried = False

            # If other agents stopped at new locations
            agent_positions = self._get_agent_positions()
            # if hasattr(self, "last_agent_positions"): # why adding this?
            #     if agent_positions != self.last_agent_positions:
            #         self.replan = True
            self.last_agent_positions = agent_positions

        # -------------------------------------------------
        # 3. Replan if necessary
        # -------------------------------------------------
        if self.replan:
            self.logger.debug(f"start replan.")
            # First get all available places
            if not self.nearby_queried:
                self.logger.debug(f"start replan: query nearby")
                self.nearby_queried = True
                thres = self.get_nearest_places(self.get_meeting_target())[0][0]
                action = {'type': 'query_map_tool', 'arg1': 'query_nearby', 'arg2': list(self.get_meeting_target()), 'arg3': thres}
                self.mode_time_counter += 1
                self.last_action = action
                return action
            if len(self.places_buffer) > 0:
                while self.places_buffer:
                    place = self.places_buffer.pop(0)
                    place_knowledge = self.s_mem.get_knowledge(place)
                    if place_knowledge is None: break
                if place_knowledge is None:
                    self.logger.debug(f"start replan: query place")
                    action = {'type': 'query_map_tool', 'arg1': 'query_place', 'arg2': place}
                    self.mode_time_counter += 1
                    self.last_action = action
                    return action
            self.replan = False
            # Cancel current navigation
            # self.exit_navigation_mode()

            next_place = self._plan_next_place()

            # if next_place is None:
            #     self.logger.warning(f"Failed to generate plan.")
            #     return {"type": "wait"}

            self.planned_place = next_place
            if self.planned_place is not None:
                self.enter_navigation_mode(goal_place=self.planned_place)
                self.logger.info(f"Plan: {self.planned_place}")

        # -------------------------------------------------
        # 4. Execute 1 navigation step only
        # -------------------------------------------------

        # if all_close
        if self.check_meeting_condition():
            self.meeting_place = self.get_meeting_place()
            self.enter_navigation_mode(goal_place=self.meeting_place)
            action, arrived = self.city_navigate(self.goal_place)
        elif self.planned_place is not None:
            action, arrived = self.city_navigate(self.goal_place)
        else:
            arrived = True

        # -------------------------------------------------
        # 5. If arrived → decide what to do
        # -------------------------------------------------
        self.mode_time_counter += 1
        if arrived:
            self.logger.debug(f"arrived")
            self.planned_place = None
            # self.exit_navigation_mode()

            all_close = self.check_meeting_condition()

            if all_close:
                self.task_complete = True
                return {"type": "task_complete"}

            # Otherwise replan next waypoint next step
            self.task_complete = False
            return {"type": "wait"}

        # -------------------------------------------------
        # 6. Continue moving
        # -------------------------------------------------
        self.last_action = action
        return action

