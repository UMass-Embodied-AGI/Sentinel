"""MAT baseline integration for the meeting challenge.

Unlike the PPO baseline (one independent policy per agent), MAT is
*centralized at decision time*: at each macro step the controller takes all
agents' observations and emits all agents' actions in one autoregressive
forward pass through a Transformer.

To fit the existing ``challenge.py`` loop, the controller is a singleton
shared in-process across all subagents (multi_process=False is forced for
MAT runs). Each ``MATSubAgent`` defers its place-selection to the
controller; navigation and observation processing inherit from
``BaseNavigationMeetingAgent`` unchanged.
"""
from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import torch

from agents.sentinel_challenge.base_nav import *
from agents.sentinel_challenge.mat_policy import (
    K,
    MATConfig,
    MATPolicy,
    featurize_team,
    load_policy,
)


# ---- centralized controller ---------------------------------------------------

class MATController:
    """Singleton shared by all MATSubAgents in a run.

    First subagent to request a decision on macro step *t* triggers the
    centralized forward; subsequent agents at the same step just read the
    cached action.
    """

    def __init__(self, num_agents: int, policy_ckpt: Optional[str] = None,
                 step_limit: int = 1500, planning_interval: int = 50,
                 training_mode: bool = False, save_path: Optional[str] = None,
                 logger=None):
        self.num_agents = num_agents
        self.step_limit = step_limit
        self.planning_interval = planning_interval
        self.training_mode = training_mode
        self.save_path = save_path
        self.logger = logger

        # set_num_threads is safe to call repeatedly. We deliberately do NOT
        # call set_num_interop_threads -- that's a one-shot call that C++
        # aborts (uncatchable from Python) on the second invocation once any
        # parallel torch work has happened. Inside a training loop that's
        # episode 2's MATController constructor, after the PPO update used
        # the interop pool. Since MAT runs multi_process=False, we don't
        # actually need the fork-safety hardening anyway.
        try:
            torch.set_num_threads(1)
        except Exception:
            pass

        if policy_ckpt is not None and os.path.exists(policy_ckpt):
            self.policy = load_policy(policy_ckpt, map_location="cpu")
            if self.logger:
                self.logger.info(f"MAT: loaded policy from {policy_ckpt}")
        else:
            self.policy = MATPolicy(num_agents=num_agents)
            if self.logger:
                self.logger.warning(
                    f"MAT: no checkpoint at {policy_ckpt!r}; random-init policy")
        self.policy.to("cpu")
        if not training_mode:
            self.policy.eval()

        # Registration: subagents add themselves on construction.
        self._agents: List["MATSubAgent"] = []
        # Cached decision: maps `decision_idx` -> {agent_name: place_name}
        self._cache: Dict[int, Dict[str, str]] = {}
        # Training trajectory: one record per macro decision.
        self.trajectory: list = []

    def register(self, agent: "MATSubAgent"):
        self._agents.append(agent)

    def is_decision_step(self, step: int) -> bool:
        return step % self.planning_interval == 0

    def decide(self, requesting_agent: "MATSubAgent") -> Optional[str]:
        """Return the meeting place for ``requesting_agent`` at its current
        global step. Returns ``None`` if no decision is available (e.g.,
        candidates empty)."""
        step = int(getattr(requesting_agent, "steps", 0))
        if step in self._cache:
            return self._cache[step].get(requesting_agent.name)

        # First request this step: run centralized decode.
        # Candidates are deterministic given the team centroid; pull from
        # requesting_agent (any agent would produce the same set).
        centroid = requesting_agent.get_meeting_target()
        candidates = requesting_agent.get_nearest_places(centroid) or []
        if not candidates:
            return None

        gmap, smaps, feats, mask = featurize_team(
            self._agents, candidates, step_limit=self.step_limit)
        with torch.no_grad():
            actions, logprobs, value = self.policy.act(
                gmap, smaps, feats, mask,
                deterministic=(not self.training_mode))

        # Map each agent's action index back to a place name. Out-of-range
        # picks (action >= len(candidates)) fall back to closest.
        cache: Dict[str, str] = {}
        for a, act_t in zip(self._agents, actions):
            ai = int(act_t.item())
            if ai >= len(candidates):
                ai = 0
            cache[a.name] = candidates[ai][1]
        self._cache[step] = cache

        if self.training_mode:
            self.trajectory.append({
                "global_map": gmap.detach().cpu(),
                "self_maps": smaps.detach().cpu(),
                "agent_feats": feats.detach().cpu(),
                "candidate_mask": mask.detach().cpu(),
                "actions": actions.detach().cpu(),
                "logprob": logprobs.detach().cpu(),     # (N,) per-agent
                "value": float(value.item()),
                "step": step,
                "agent_names": [a.name for a in self._agents],
            })
            self._flush()

        if self.logger:
            picks = ", ".join(f"{a.name}={cache[a.name]}" for a in self._agents)
            self.logger.info(f"MAT: step {step} centralized decode -> {picks}")
        return cache.get(requesting_agent.name)

    def _flush(self):
        if not self.training_mode or self.save_path is None:
            return
        try:
            tmp = self.save_path + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(self.trajectory, f)
            os.replace(tmp, self.save_path)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"MAT: failed to flush trajectory: {e}")


# A single module-level holder. challenge.py constructs the controller once
# per run and sets this; subagents read it on construction.
_ACTIVE_CONTROLLER: Optional[MATController] = None


def set_active_controller(ctrl: Optional[MATController]) -> None:
    global _ACTIVE_CONTROLLER
    _ACTIVE_CONTROLLER = ctrl


def get_active_controller() -> Optional[MATController]:
    return _ACTIVE_CONTROLLER


# ---- per-agent subclass -------------------------------------------------------

class MATSubAgent(BaseNavigationMeetingAgent):
    """Per-agent wrapper that defers meeting-place selection to a shared
    ``MATController``. Navigation and observation processing are inherited.

    Constructed without an LLM generator (no LLM dependency).
    """

    def __init__(self, name, pose, info, sim_path, no_react=False, debug=False,
                 logger=None,
                 lm_source='openai', lm_id='gpt-4o', max_tokens=4096,
                 temperature=0, top_p=1.0, init_generator=True,
                 detect_interval=-1, num_agents=1, enable_danger_zone=False,
                 ablate="", planning_interval: int = 50,
                 step_limit: int = 1500):
        super().__init__(name, pose, info, sim_path, no_react, debug, logger,
                         lm_source, lm_id, max_tokens, temperature, top_p,
                         init_generator=init_generator,
                         detect_interval=detect_interval, num_agents=num_agents,
                         enable_danger_zone=enable_danger_zone, ablate=ablate)
        self.planning_interval = planning_interval
        self.step_limit = step_limit
        self.planned_place: Optional[str] = None
        self.nearby_queried: bool = False
        self.task_complete: bool = False
        self.replan = False

        ctrl = get_active_controller()
        if ctrl is None:
            raise RuntimeError(
                "MATSubAgent constructed without an active MATController. "
                "challenge.py should call set_active_controller() first.")
        self.controller = ctrl
        self.controller.register(self)

    def reset(self, name, pose):
        super().reset(name, pose)
        self.planned_place = None
        self.replan = False
        self.nearby_queried = False
        self.task_complete = False

    def _process_obs(self, obs):
        super()._process_obs(obs)
        self.process_obs_with_sptial_knowledge(obs)

    def _check_meeting_condition(self) -> bool:
        positions = {nm: info["pose"][:2]
                     for nm, info in self.obs["agent_pos_dict"].items()}
        my = np.array(positions.get(self.name, [0.0, 0.0])[:2])
        if my[0] > 500:
            my = my - 1000
        for nm, p in positions.items():
            if nm == self.name:
                continue
            other = np.array(p[:2])
            if other[0] > 500:
                other = other - 1000
            if np.linalg.norm(my - other) > 20.0:
                return False
        return True

    def _act(self, obs):
        # Captured -> off-map + task_complete.
        if self.banned:
            if self.pose[0] > -1000:
                return {"type": "teleport", "arg1": [-1500., -1500.]}
            return {"type": "task_complete"}

        # Decision moment: ask the controller for a meeting_place.
        step = int(getattr(self, "steps", 0))
        if (self.planned_place is None and not self.task_complete) or \
           self.controller.is_decision_step(step):
            if not self.nearby_queried:
                self.nearby_queried = True
                target = self.get_meeting_target()
                nearest = self.get_nearest_places(target)
                if nearest:
                    thres = nearest[0][0]
                    self.mode_time_counter += 1
                    action = {
                        "type": "query_map_tool",
                        "arg1": "query_nearby",
                        "arg2": list(target),
                        "arg3": thres,
                    }
                    self.last_action = action
                    return action

            if len(self.places_buffer) > 0:
                place_knowledge = None
                place = None
                while self.places_buffer:
                    place = self.places_buffer.pop(0)
                    place_knowledge = self.s_mem.get_knowledge(place)
                    if place_knowledge is None:
                        break
                if place_knowledge is None and place is not None:
                    action = {"type": "query_map_tool", "arg1": "query_place",
                              "arg2": place}
                    self.mode_time_counter += 1
                    self.last_action = action
                    return action

            picked = self.controller.decide(self)
            if picked is not None:
                self.planned_place = picked
                self.meeting_place = picked
                self.enter_navigation_mode(goal_place=picked)
                self.nearby_queried = False  # allow re-query at next decision

        # Navigation step (delegated to base_nav).
        arrived = True
        action = {"type": "wait"}
        if self._check_meeting_condition():
            self.meeting_place = self.get_meeting_place()
            self.enter_navigation_mode(goal_place=self.meeting_place)
            action, arrived = self.city_navigate(self.goal_place)
        elif self.planned_place is not None:
            action, arrived = self.city_navigate(self.goal_place)

        self.mode_time_counter += 1
        if arrived:
            self.planned_place = None
            if self._check_meeting_condition():
                self.task_complete = True
                return {"type": "task_complete"}
            self.task_complete = False
            return {"type": "wait"}

        self.last_action = action
        return action
