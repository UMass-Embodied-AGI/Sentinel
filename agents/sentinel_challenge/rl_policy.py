"""PPO policy + featurization for the RL meeting-challenge baseline.

Self-contained: no Genesis / sim imports. Importable from both the agent
(`RL.py`) and the trainer (`sentinel_challenge/train_rl.py`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- shape constants ----------------------------------------------------------

K = 15                              # top-K candidate meeting places
MAX_OTHERS = 7                      # padded slots for other agents (excluding self)
PLACE_TYPES = (
    "open space", "restaurant", "shop", "park", "station",
    "office", "home", "school", "hospital", "other",
)                                   # coarse place-type buckets (last one = fallback)
P_DIM = len(PLACE_TYPES)
WARN_RADIUS_CELLS = 8               # window radius for warning_density


def _candidate_feat_dim() -> int:
    # rel_xy_self(2) + rel_xy_centroid(2) + dist_to_centroid(1)
    # + min_dist_to_warning(1) + warning_density(1) + place_type_onehot(P_DIM)
    return 2 + 2 + 1 + 1 + 1 + P_DIM


def _global_feat_dim() -> int:
    # self_xy(2) + self_yaw_sincos(2) + others_xy_mask(MAX_OTHERS*3)
    # + time_remaining(1) + self_banned(1) + current_place_in_top_k_onehot(K+1)
    return 2 + 2 + MAX_OTHERS * 3 + 1 + 1 + (K + 1)


GLOBAL_DIM = _global_feat_dim()
CANDIDATE_DIM = _candidate_feat_dim()


# ---- featurization ------------------------------------------------------------

def _bucket_place_type(building: Optional[str]) -> int:
    if building is None:
        return PLACE_TYPES.index("other")
    b = building.lower()
    for i, t in enumerate(PLACE_TYPES):
        if t in b:
            return i
    return PLACE_TYPES.index("other")


def _normalize_xy(xy: np.ndarray) -> np.ndarray:
    # Scene coords are typically within ~500m; divide by 200 to keep features O(1).
    return xy.astype(np.float32) / 200.0


def _fetch_occ_map(builder):
    """Pull the occupancy map once. Returns
    ``(occ, x_min, y_min, x_max, y_max)`` or ``None`` on any error.

    ``builder.get_occ_map()`` can be expensive (writes a PNG / queries the
    volume grid); we call it once per decision and reuse across all candidate
    places.
    """
    if builder is None:
        return None
    try:
        out = builder.get_occ_map()
    except Exception:
        return None
    if out is None:
        return None
    occ = out[0]
    if occ is None or getattr(occ, "size", 0) == 0:
        return None
    return out


def _occ_warning_summary(occ_tuple, place_xy: np.ndarray,
                         radius_cells: int = WARN_RADIUS_CELLS):
    """Reduce a precomputed occupancy map around `place_xy` to two scalars:
    (min L2 distance to a warning cell, fraction of warning cells in window).

    Returns ``(min_dist_norm, density)`` in [0, 1]. If no occupancy info was
    obtainable (``occ_tuple is None``), returns ``(1.0, 0.0)``.
    """
    if occ_tuple is None:
        return 1.0, 0.0
    occ, x_min, y_min, x_max, y_max = occ_tuple
    H, W = occ.shape[:2]
    # Convert place_xy (world) -> grid indices (best-effort; uniform grid).
    fx = (place_xy[0] - x_min) / max(x_max - x_min, 1e-6)
    fy = (place_xy[1] - y_min) / max(y_max - y_min, 1e-6)
    ci = int(np.clip(fy * (H - 1), 0, H - 1))
    cj = int(np.clip(fx * (W - 1), 0, W - 1))
    i0, i1 = max(ci - radius_cells, 0), min(ci + radius_cells + 1, H)
    j0, j1 = max(cj - radius_cells, 0), min(cj + radius_cells + 1, W)
    window = occ[i0:i1, j0:j1]
    warn_mask = (window == 4)
    if not warn_mask.any():
        return 1.0, 0.0
    density = float(warn_mask.sum()) / float(warn_mask.size)
    yy, xx = np.where(warn_mask)
    dy = (yy + i0) - ci
    dx = (xx + j0) - cj
    min_d = float(np.sqrt(dx * dx + dy * dy).min())
    # Normalize by window size so feature is bounded.
    return min(min_d / (2 * radius_cells + 1), 1.0), min(density, 1.0)


def _shift_coord(xy):
    """Reproduce base_nav's coordinate hack: positions > 500 are wrapped by -1000."""
    arr = np.array(xy[:2], dtype=np.float32)
    if arr[0] > 500:
        arr -= 1000.0
    return arr


def featurize(agent, candidates, step_limit: int = 1500):
    """Build (global_feats, candidate_feats, candidate_mask) for one decision.

    Args:
        agent: an ``RLMeetingAgent`` (or any ``BaseNavigationMeetingAgent``
            subclass that exposes ``obs``, ``pose``, ``s_mem``, ``current_place``,
            ``banned``, ``steps``).
        candidates: list of ``(distance, place_name)`` tuples from
            ``agent.get_nearest_places(target)``. May be shorter than K.
        step_limit: episode length cap from challenge.py (default 1500).

    Returns:
        global_feats: torch.FloatTensor of shape (GLOBAL_DIM,)
        candidate_feats: torch.FloatTensor of shape (K, CANDIDATE_DIM)
        candidate_mask: torch.FloatTensor of shape (K,) with 1.0 for real rows.
    """
    obs = agent.obs
    centroid = agent.get_meeting_target()                # ndarray (2,)
    self_xy = _shift_coord(agent.pose[:2])
    self_yaw = float(agent.pose[-1]) if len(agent.pose) >= 3 else 0.0

    # ---- global features ----
    g = np.zeros(GLOBAL_DIM, dtype=np.float32)
    cur = 0
    g[cur:cur + 2] = _normalize_xy(self_xy); cur += 2
    g[cur:cur + 2] = [math.cos(self_yaw), math.sin(self_yaw)]; cur += 2

    # other agents (excluding self), padded
    others = []
    for nm, info in obs.get("agent_pos_dict", {}).items():
        if nm == agent.name:
            continue
        oxy = _shift_coord(info["pose"][:2])
        others.append(oxy)
    for i in range(MAX_OTHERS):
        if i < len(others):
            g[cur:cur + 2] = _normalize_xy(others[i]); cur += 2
            g[cur] = 1.0; cur += 1                      # mask flag = present
        else:
            cur += 2
            g[cur] = 0.0; cur += 1

    steps_done = getattr(agent, "steps", 0)
    g[cur] = max(0.0, 1.0 - steps_done / float(step_limit)); cur += 1
    g[cur] = 1.0 if getattr(agent, "banned", False) else 0.0; cur += 1

    # ---- per-candidate features + current_place onehot over candidates ----
    cand = np.zeros((K, CANDIDATE_DIM), dtype=np.float32)
    mask = np.zeros(K, dtype=np.float32)
    current_place_idx = K   # default = "not in top-K"

    sg = None
    try:
        sg = agent.s_mem.get_sg(place=agent.current_place)
    except Exception:
        sg = None
    builder = getattr(sg, "volume_grid_builder", None) if sg is not None else None
    occ_tuple = _fetch_occ_map(builder)  # one call per decision, reused below

    for i, (_dist, place_name) in enumerate(candidates[:K]):
        kn = agent.s_mem.get_knowledge(place_name)
        if kn is None:
            continue
        loc = _shift_coord(kn["location"][:2])
        rel_self = loc - self_xy
        rel_cent = loc - np.array(centroid, dtype=np.float32)
        cur2 = 0
        cand[i, cur2:cur2 + 2] = _normalize_xy(rel_self); cur2 += 2
        cand[i, cur2:cur2 + 2] = _normalize_xy(rel_cent); cur2 += 2
        cand[i, cur2] = float(np.linalg.norm(rel_cent)) / 200.0; cur2 += 1

        min_d, density = _occ_warning_summary(occ_tuple, loc)
        cand[i, cur2] = min_d; cur2 += 1
        cand[i, cur2] = density; cur2 += 1

        pt_idx = _bucket_place_type(kn.get("building"))
        cand[i, cur2 + pt_idx] = 1.0

        mask[i] = 1.0
        if place_name == agent.current_place:
            current_place_idx = i

    onehot = np.zeros(K + 1, dtype=np.float32)
    onehot[current_place_idx] = 1.0
    g[cur:cur + K + 1] = onehot

    return (
        torch.from_numpy(g),
        torch.from_numpy(cand),
        torch.from_numpy(mask),
    )


# ---- policy net ---------------------------------------------------------------

class PlacePolicy(nn.Module):
    """Discrete masked policy over top-K candidate places + value head."""

    def __init__(self, global_dim: int = GLOBAL_DIM,
                 candidate_dim: int = CANDIDATE_DIM, hidden: int = 256):
        super().__init__()
        self.global_dim = global_dim
        self.candidate_dim = candidate_dim
        self.trunk = nn.Sequential(
            nn.Linear(global_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # per-candidate scoring head: takes [trunk_out || cand_feat] -> logit
        self.score = nn.Sequential(
            nn.Linear(hidden + candidate_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, global_feats: torch.Tensor,
                candidate_feats: torch.Tensor,
                candidate_mask: torch.Tensor):
        """Returns (logits[B,K], value[B]). Logits are masked to -1e9 where
        candidate_mask==0 so sampling can never pick padded rows.
        """
        if global_feats.dim() == 1:
            global_feats = global_feats.unsqueeze(0)
            candidate_feats = candidate_feats.unsqueeze(0)
            candidate_mask = candidate_mask.unsqueeze(0)
        h = self.trunk(global_feats)                            # (B, H)
        h_exp = h.unsqueeze(1).expand(-1, candidate_feats.shape[1], -1)
        joint = torch.cat([h_exp, candidate_feats], dim=-1)     # (B,K,H+C)
        # Flatten leading dims for the score head: nn.Linear broadcasts over
        # arbitrary leading dims, but the 3D BLAS dispatch can deadlock in a
        # post-fork child. The (B*K, F) 2D path is reliable.
        B, K_, F = joint.shape
        scores_flat = self.score(joint.reshape(B * K_, F).contiguous())
        logits = scores_flat.view(B, K_).masked_fill(candidate_mask < 0.5, -1e9)
        value = self.value_head(h).squeeze(-1)                  # (B,)
        return logits, value

    def act(self, global_feats, candidate_feats, candidate_mask,
            deterministic: bool = False):
        logits, value = self.forward(global_feats, candidate_feats, candidate_mask)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob, value, dist.entropy()


@dataclass
class PolicyConfig:
    global_dim: int = GLOBAL_DIM
    candidate_dim: int = CANDIDATE_DIM
    hidden: int = 256
    k: int = K


def save_policy(policy: PlacePolicy, cfg: PolicyConfig, path: str) -> None:
    torch.save({"state_dict": policy.state_dict(),
                "config": cfg.__dict__}, path)


def load_policy(path: str, map_location: str = "cpu") -> PlacePolicy:
    blob = torch.load(path, map_location=map_location)
    cfg = PolicyConfig(**blob["config"])
    pol = PlacePolicy(cfg.global_dim, cfg.candidate_dim, cfg.hidden)
    pol.load_state_dict(blob["state_dict"])
    pol.eval()
    return pol
