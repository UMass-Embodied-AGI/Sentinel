"""Sentinel-Challenge environment.

Subclasses upstream's `vico.env.VicoEnv` with the additional action types
and state needed by `sentinel_challenge/`. Everything else (scene loading,
agent stepping, observations, traffic manager, map_tool, ...) is
inherited as-is so `vico/env.py` remains the single source of truth.

New action types dispatched in `perform_action`:
    * `remote_converse` — multi-agent speech with priority + range
    * `task_complete`   — agent signals task completion (env no-op)
    * `task_terminate`  — agent signals task abort      (env no-op)
    * `signal`          — broadcast a sentinel-signal event
    * `debug_action`    — accessibility-check diagnostics

Added state (set after `super().__init__()`):
    * `agent_speak_weight` — per-agent priority budget for `remote_converse`
"""

import random

import genesis as gs

from vico.env import VicoEnv
from vico.modules.avatar.utils import AvatarState, ActionStatus
from vico.tools.utils import is_near_goal


class SentinelVicoEnv(VicoEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_speak_weight = [100] * self.num_agents

    def perform_action(self, agent_id, action, is_robot=False):
        if action is None:
            return super().perform_action(agent_id, action, is_robot=is_robot)

        atype = action["type"]

        if atype in ("task_complete", "task_terminate"):
            return

        agent = self.agents[agent_id]

        if atype == "remote_converse":
            self._action_remote_converse(agent_id, agent, action)
            return
        if atype == "signal":
            self._action_signal(agent_id, agent, action)
            return
        if atype == "debug_action":
            self._action_debug(agent_id, agent, action)
            return

        return super().perform_action(agent_id, action, is_robot=is_robot)

    # ------------------------------------------------------------------ #
    # Sentinel-specific action handlers
    # ------------------------------------------------------------------ #

    def _action_remote_converse(self, agent_id, agent, action):
        if agent.robot.base_state == AvatarState.SLEEPING:
            agent.robot.base_state = AvatarState.STANDING
        agent_pos = self.config["agent_poses"][agent_id][:3]
        converse_range = action.get("arg2", 10)
        priority = random.randint(0, self.agent_speak_weight[agent_id])
        self.agent_speak_weight[agent_id] = max(0, self.agent_speak_weight[agent_id] - 30)
        if converse_range > 3200:
            gs.logger.warning(
                f"Agent {self.agent_names[agent_id]} attempted to converse "
                f"with range {converse_range} which is larger than 3200. Ignored."
            )
            self.agents[agent_id].robot.action_status = ActionStatus.FAIL
            return
        deleted = self.events.add(
            type="speech", pos=agent_pos, r=converse_range,
            content=action["arg1"], priority=priority,
            subject=self.agent_names[agent_id], predicate="is", object="talk",
        )
        # If overlapping with another speech event, keep only the highest
        # priority one; others are marked FAIL and given a speak-weight bonus.
        for subject in deleted:
            i = self.agent_names.index(subject)
            self.agents[i].robot.action_status = ActionStatus.FAIL
            self.agent_speak_weight[i] += 60

    def _action_signal(self, agent_id, agent, action):
        if agent.robot.base_state == AvatarState.SLEEPING:
            agent.robot.base_state = AvatarState.STANDING
        agent_pos = self.config["agent_poses"][agent_id][:3]
        deleted = self.events.add(
            type="sentinel signal", pos=agent_pos, r=20, content=action, priority=100,
            subject=self.agent_names[agent_id], predicate="is", object="issue",
        )
        for subject in deleted:
            self.agents[self.agent_names.index(subject)].robot.action_status = ActionStatus.FAIL

    def _action_debug(self, agent_id, agent, action):
        if action["arg1"] != "query_accessible_places":
            return
        assert action["arg2"] in self.place_metadata
        pose = agent.get_global_pose()
        info = self.agent_infos[agent_id]
        gs.logger.info(
            f"debug action triggered: checking {action['arg2']} for accessibility. "
            f"{self.agent_names[agent_id]}'s current place is {info['current_place']}, "
            f"current building is {info['current_building']}, outdoor pose is "
            f"{info['outdoor_pose']}, current pose is {pose}."
        )
        if info["current_place"] is not None:
            assert info["current_building"] != "open space", (
                "The current building must be provided if the agent is not in the open space."
            )
            assert action["arg2"] in [
                place["name"]
                for place in self.building_metadata[info["current_building"]]["places"]
                if place["name"] != info["current_place"]
            ]
        for building, meta in self.building_metadata.items():
            if building == "open space":
                gs.logger.warning("checking openspace")
                for place in meta["places"]:
                    gs.logger.warning(f"checking place {place}")
                    if place == action["arg2"]:
                        gs.logger.warning(
                            f"Checking accessibility for place {action['arg2']} in open space, "
                            f"is_near_goal is "
                            f"{is_near_goal(pose[0], pose[1], None, place['location'][:2], threshold=5)}"
                        )
            if meta["bounding_box"] is None:
                continue
            if action["arg2"] in meta["places"]:
                gs.logger.warning(
                    f"Checking accessibility for place {action['arg2']} in open space, "
                    f"is_near_goal is "
                    f"{is_near_goal(pose[0], pose[1], meta['bounding_box'], None, threshold=5)}"
                )
