"""RL baseline for the meeting challenge.

The policy is a PPO-trained discrete classifier over the top-K candidate
places returned by ``BaseNavigationMeetingAgent.get_nearest_places``. All
navigation, low-level control, and observation processing are inherited
from ``base_nav``; only the meeting-place selection is replaced by a torch
forward pass.

No inter-agent communication: agents converge by parameter sharing on
shared observable state (``agent_pos_dict`` + deterministic candidate
ordering from ``get_nearest_places``).
"""
import copy
import os
import pickle

import numpy as np
import torch

from agents.sentinel_challenge.base_nav import *
from agents.sentinel_challenge.rl_policy import (
    K,
    PlacePolicy,
    PolicyConfig,
    featurize,
    load_policy,
)


class RLMeetingAgent(BaseNavigationMeetingAgent):
    """PPO baseline. Picks meeting_place via a learned policy; delegates
    navigation to ``base_nav.city_navigate``.
    """

    def __init__(self, name, pose, info, sim_path, no_react=False, debug=False, logger=None,
                 lm_source='openai', lm_id='gpt-4o', max_tokens=4096, temperature=0, top_p=1.0,
                 init_generator=True,
                 detect_interval=-1, num_agents=1, enable_danger_zone=False, ablate="",
                 policy_ckpt: str = None, training_mode: bool = False,
                 planning_interval: int = 50, step_limit: int = 1500):
        # Pin torch to a single thread inside the agent subprocess. The parent
        # process initialized CUDA (and the OpenMP/MKL threadpool) via Genesis
        # before forking; multi-threaded BLAS calls in the forked child can
        # deadlock the inherited threadpool. The policy is small enough that
        # single-threaded is more than fast enough.
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        # The RL baseline does not call an LLM; force-disable the generator to
        # avoid spinning up an OpenAI/Azure client at construction time.
        super().__init__(
            name, pose, info, sim_path, no_react, debug, logger,
            lm_source, lm_id, max_tokens, temperature, top_p,
            init_generator=init_generator,
            detect_interval=detect_interval, num_agents=num_agents,
            enable_danger_zone=enable_danger_zone, ablate=ablate,
        )

        self.training_mode = training_mode
        self.planning_interval = planning_interval
        self.step_limit = step_limit

        # Load or freshly initialize the policy. Force CPU device context: the
        # parent process initialized CUDA via Genesis before forking the agent
        # subprocess, and torch refuses to re-init CUDA in a forked child.
        with torch.device("cpu"):
            if policy_ckpt is not None and os.path.exists(policy_ckpt):
                self.policy = load_policy(policy_ckpt, map_location="cpu")
                if self.logger:
                    self.logger.info(f"RL: loaded policy from {policy_ckpt}")
            else:
                self.policy = PlacePolicy()
                if self.logger:
                    self.logger.warning(
                        f"RL: no checkpoint at {policy_ckpt!r}; using random-init policy"
                    )
        self.policy.to("cpu")
        if not training_mode:
            self.policy.eval()

        # Planning bookkeeping (mirrors MCTSMeetingAgent).
        self.planned_place: str = None
        self.replan = False
        self.nearby_queried = False
        self.task_complete = False

        # Trajectory buffer for the trainer (only populated when training_mode=True).
        # Each entry is one decision event:
        #   {"global": tensor, "cand": tensor, "mask": tensor,
        #    "action": int, "logprob": float, "value": float, "step": int}
        self.rl_trajectory: list = []

    def reset(self, name, pose):
        super().reset(name, pose)
        self.planned_place = None
        self.replan = False
        self.nearby_queried = False
        self.task_complete = False
        self.rl_trajectory = []

    def _process_obs(self, obs):
        super()._process_obs(obs)
        self.process_obs_with_sptial_knowledge(obs)

    def _get_agent_positions(self):
        positions = {}
        for nm, info in self.obs["agent_pos_dict"].items():
            xy = np.array(info["pose"][:2], dtype=np.float32)
            if xy[0] > 500:
                xy -= 1000.0
            positions[nm] = xy.tolist()
        return positions

    def _trajectory_path(self) -> str:
        return os.path.join(self.storage_path, "rl_trajectory.pkl")

    def _flush_trajectory(self) -> None:
        """Persist the per-decision trajectory so the trainer can read it back
        across the multiprocessing boundary used by ``AgentProcess``.
        """
        if not self.training_mode:
            return
        try:
            with open(self._trajectory_path(), "wb") as f:
                pickle.dump(self.rl_trajectory, f)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"RL: failed to flush trajectory: {e}")

    def _check_meeting_condition(self) -> bool:
        positions = self._get_agent_positions()
        my = np.array(positions.get(self.name, [0.0, 0.0])[:2])
        for nm, p in positions.items():
            if nm == self.name:
                continue
            if np.linalg.norm(my - np.array(p[:2])) > 20.0:
                return False
        return True

    def _plan_next_place(self) -> str:
        """Run one forward pass of the policy to pick a candidate place."""
        target = self.get_meeting_target()
        candidates = self.get_nearest_places(target)
        if not candidates:
            return None
        g, c, m = featurize(self, candidates, step_limit=self.step_limit)
        with torch.no_grad():
            logits, value_t = self.policy(g, c, m)
        logits = logits.squeeze(0)
        if self.training_mode:
            # softmax in fp64 for numerical stability, then numpy multinomial
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            probs = np.clip(probs, 1e-12, None)
            probs = probs / probs.sum()
            a_int = int(np.random.choice(len(probs), p=probs))
            logprob = float(np.log(probs[a_int]))
        else:
            a_int = int(torch.argmax(logits).item())
            logprob = float(torch.log_softmax(logits, dim=-1)[a_int].item())
        value = float(value_t.item())
        if a_int >= len(candidates):
            # padding row picked due to numerical edge case; fall back to closest
            a_int = 0
        picked = candidates[a_int][1]

        if self.training_mode:
            self.rl_trajectory.append({
                "global": g.detach().cpu(),
                "cand": c.detach().cpu(),
                "mask": m.detach().cpu(),
                "action": a_int,
                "logprob": logprob,
                "value": value,
                "step": getattr(self, "steps", 0),
            })
            self._flush_trajectory()

        if self.logger:
            self.logger.info(
                f"RL: planned meeting_place={picked!r} "
                f"(action={a_int}, value={value:.3f})"
            )
        return picked

    def _act(self, obs):
        # 1. Captured -> teleport off-map and finish.
        if self.banned:
            if self.pose[0] > -1000:
                return {"type": "teleport", "arg1": [-1500., -1500.]}
            return {"type": "task_complete"}

        # 2. Replan triggers.
        if not self.replan:
            if self.planned_place is None and not self.task_complete:
                self.replan = True
                self.nearby_queried = False
            elif self.mode_time_counter % self.planning_interval == 0:
                self.replan = True
                self.nearby_queried = False

        # 3. Replan: optionally query nearby places, then run policy.
        if self.replan:
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

            # Backfill missing knowledge for any place still in places_buffer.
            if len(self.places_buffer) > 0:
                place_knowledge = None
                place = None
                while self.places_buffer:
                    place = self.places_buffer.pop(0)
                    place_knowledge = self.s_mem.get_knowledge(place)
                    if place_knowledge is None:
                        break
                if place_knowledge is None and place is not None:
                    action = {"type": "query_map_tool", "arg1": "query_place", "arg2": place}
                    self.mode_time_counter += 1
                    self.last_action = action
                    return action

            self.replan = False
            next_place = self._plan_next_place()
            self.planned_place = next_place
            if self.planned_place is not None:
                self.meeting_place = self.planned_place
                self.enter_navigation_mode(goal_place=self.planned_place)

        # 4. Step navigation.
        arrived = True
        action = {"type": "wait"}
        if self._check_meeting_condition():
            self.meeting_place = self.get_meeting_place()
            self.enter_navigation_mode(goal_place=self.meeting_place)
            action, arrived = self.city_navigate(self.goal_place)
        elif self.planned_place is not None:
            action, arrived = self.city_navigate(self.goal_place)

        # 5. Arrived handling.
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
