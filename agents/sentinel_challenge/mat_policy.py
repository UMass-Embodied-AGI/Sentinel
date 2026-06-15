"""Multi-Agent Transformer (MAT) policy + featurization for the meeting
challenge baseline. Based on Wen et al. 2022 ("Multi-Agent Transformer:
Solving Multi-Agent Reinforcement Learning Problems via Sequence Models")
with CoELA-style hyperparameters (hidden=64, top-down map input).

This module is self-contained (no Genesis / sim imports). Importable from
the agent (`MAT.py`) and the trainer (`sentinel_challenge/train_mat.py`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- shape constants ----------------------------------------------------------

K = 15                  # top-K candidate meeting places (matches PPO baseline)
MAP_H = 12              # top-down map height (matches CoELA)
MAP_W = 24              # top-down map width
GLOBAL_CH = 7           # channels in the shared map (excluding per-agent self)
AGENT_CH = 1            # extra channel injected per agent (self marker)
PLACE_VOCAB = K + 1     # K candidates + "not in candidates"

# Per-agent feature vector: cos(yaw) + sin(yaw) + banned + step_frac + cp one-hot
AGENT_FEAT_DIM = 2 + 1 + 1 + PLACE_VOCAB

# Crop window in world units around the agent centroid (meters).
CROP_HALF_W = 200.0     # x ranges over [cx - CROP_HALF_W, cx + CROP_HALF_W]
CROP_HALF_H = 100.0     # y ranges over [cy - CROP_HALF_H, cy + CROP_HALF_H]

# Hyperparameters matching CoELA / Wen et al.
HIDDEN = 64
N_HEADS = 4
N_LAYERS = 1            # 1 enc + 1 dec layer; small Transformer per the spec


# ---- map building -------------------------------------------------------------

def _shift_coord_arr(arr: np.ndarray) -> np.ndarray:
    """Reproduce base_nav's > 500 coordinate-wrap hack on (N, 2) array."""
    out = arr.astype(np.float32).copy()
    mask = out[:, 0] > 500.0
    out[mask] -= 1000.0
    return out


def _world_to_grid(x: float, y: float, cx: float, cy: float):
    """Project world (x, y) -> (i, j) cell in a (MAP_H, MAP_W) grid centered
    on (cx, cy). Returns None if outside the crop window.
    """
    fy = (y - (cy - CROP_HALF_H)) / (2.0 * CROP_HALF_H)
    fx = (x - (cx - CROP_HALF_W)) / (2.0 * CROP_HALF_W)
    if not (0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0):
        return None
    j = int(np.clip(fx * (MAP_W - 1), 0, MAP_W - 1))
    i = int(np.clip(fy * (MAP_H - 1), 0, MAP_H - 1))
    return i, j


def _fetch_occ_map(builder):
    """One call per decision moment, shared across all agents."""
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


def build_global_map(agents, candidates: List[Tuple[float, str]],
                     centroid: np.ndarray, builder) -> np.ndarray:
    """Build the shared (GLOBAL_CH, MAP_H, MAP_W) map (excluding per-agent
    self marker, which is added later).

    Channels: 0=free, 1=obstacle, 2=unknown, 3=warning,
              4=candidate places, 5=agents-all, 6=meeting place.
    """
    M = np.zeros((GLOBAL_CH, MAP_H, MAP_W), dtype=np.float32)
    cx, cy = float(centroid[0]), float(centroid[1])

    # --- occupancy channels (0..3) ----
    occ_tuple = _fetch_occ_map(builder)
    if occ_tuple is not None:
        occ, x_min, y_min, x_max, y_max = occ_tuple
        # The occ grid is in builder coordinates which already share frame
        # with agent poses (verified by base_nav usage). Project each cell
        # to our crop. To stay simple & cheap, vectorize via per-pixel target
        # bilinear resample using nearest neighbor.
        ys = np.linspace(cy - CROP_HALF_H, cy + CROP_HALF_H, MAP_H,
                         dtype=np.float32)
        xs = np.linspace(cx - CROP_HALF_W, cx + CROP_HALF_W, MAP_W,
                         dtype=np.float32)
        # Convert each (x, y) target to (oi, oj) inside the occ array.
        fy = (ys - y_min) / max(y_max - y_min, 1e-6)
        fx = (xs - x_min) / max(x_max - x_min, 1e-6)
        oi = np.clip((fy * (occ.shape[0] - 1)).astype(np.int64), 0,
                     occ.shape[0] - 1)
        oj = np.clip((fx * (occ.shape[1] - 1)).astype(np.int64), 0,
                     occ.shape[1] - 1)
        sampled = occ[oi[:, None], oj[None, :]]      # (MAP_H, MAP_W)
        M[0] = (sampled == 3).astype(np.float32)     # free
        M[1] = (sampled == 2).astype(np.float32)     # obstacle
        M[2] = (sampled == 1).astype(np.float32)     # unknown
        M[3] = (sampled == 4).astype(np.float32)     # warning

    # --- candidate places (channel 4) ----
    # We rely on each agent's s_mem for place locations. Use the first
    # agent that has knowledge for the place. Same set of K places shared.
    for _dist, place_name in candidates[:K]:
        loc = None
        for a in agents:
            kn = a.s_mem.get_knowledge(place_name)
            if kn is None:
                continue
            loc = _shift_coord_arr(np.array([[kn["location"][0],
                                              kn["location"][1]]]))[0]
            break
        if loc is None:
            continue
        ij = _world_to_grid(float(loc[0]), float(loc[1]), cx, cy)
        if ij is not None:
            M[4, ij[0], ij[1]] = 1.0

    # --- all agents (channel 5) ----
    for a in agents:
        if getattr(a, "banned", False):
            continue
        xy = _shift_coord_arr(np.array([[a.pose[0], a.pose[1]]]))[0]
        ij = _world_to_grid(float(xy[0]), float(xy[1]), cx, cy)
        if ij is not None:
            M[5, ij[0], ij[1]] = 1.0

    # --- meeting place (channel 6) ----
    for a in agents:
        if getattr(a, "meeting_place", None) is None:
            continue
        kn = a.s_mem.get_knowledge(a.meeting_place)
        if kn is None:
            continue
        loc = _shift_coord_arr(np.array([[kn["location"][0],
                                          kn["location"][1]]]))[0]
        ij = _world_to_grid(float(loc[0]), float(loc[1]), cx, cy)
        if ij is not None:
            M[6, ij[0], ij[1]] = 1.0
        break  # all agents share meeting_place once agreed

    return M


def build_self_marker(agent, centroid: np.ndarray) -> np.ndarray:
    """Single-channel (1, MAP_H, MAP_W) one-hot of agent's grid cell."""
    out = np.zeros((1, MAP_H, MAP_W), dtype=np.float32)
    xy = _shift_coord_arr(np.array([[agent.pose[0], agent.pose[1]]]))[0]
    ij = _world_to_grid(float(xy[0]), float(xy[1]),
                        float(centroid[0]), float(centroid[1]))
    if ij is not None:
        out[0, ij[0], ij[1]] = 1.0
    return out


def build_agent_feats(agent, candidates: List[Tuple[float, str]],
                      step_limit: int) -> np.ndarray:
    """(AGENT_FEAT_DIM,) feature vector for one agent."""
    yaw = float(agent.pose[-1]) if len(agent.pose) >= 3 else 0.0
    v = np.zeros(AGENT_FEAT_DIM, dtype=np.float32)
    cur = 0
    v[cur] = math.cos(yaw); cur += 1
    v[cur] = math.sin(yaw); cur += 1
    v[cur] = 1.0 if getattr(agent, "banned", False) else 0.0; cur += 1
    steps_done = getattr(agent, "steps", 0)
    v[cur] = max(0.0, 1.0 - steps_done / float(step_limit)); cur += 1
    # current_place index in candidate list
    cp_idx = PLACE_VOCAB - 1
    cur_place = getattr(agent, "current_place", None)
    for i, (_dist, name) in enumerate(candidates[:K]):
        if name == cur_place:
            cp_idx = i
            break
    v[cur + cp_idx] = 1.0
    return v


def featurize_team(agents, candidates: List[Tuple[float, str]],
                   step_limit: int = 1500):
    """Build all tensors needed for one MAT decision moment.

    Returns:
        global_map:  (1, GLOBAL_CH,        MAP_H, MAP_W)  shared
        self_maps:   (N, AGENT_CH,         MAP_H, MAP_W)  per-agent
        agent_feats: (N, AGENT_FEAT_DIM)                  per-agent
        candidate_mask: (K,) float, 1 for real candidates
    """
    # centroid for cropping
    poses = np.array([[a.pose[0], a.pose[1]] for a in agents], dtype=np.float32)
    poses = _shift_coord_arr(poses)
    centroid = poses.mean(axis=0)

    # Pick any agent's scene-graph builder; they share the volume grid.
    builder = None
    for a in agents:
        try:
            sg = a.s_mem.get_sg(place=a.current_place)
            builder = getattr(sg, "volume_grid_builder", None)
            if builder is not None:
                break
        except Exception:
            continue

    gmap = build_global_map(agents, candidates, centroid, builder)        # (GC,H,W)
    self_maps = np.stack([build_self_marker(a, centroid) for a in agents],
                         axis=0)                                          # (N,1,H,W)
    feats = np.stack([build_agent_feats(a, candidates, step_limit)
                      for a in agents], axis=0)                           # (N,F)
    mask = np.zeros(K, dtype=np.float32)
    mask[:min(K, len(candidates))] = 1.0
    return (torch.from_numpy(gmap).unsqueeze(0),
            torch.from_numpy(self_maps),
            torch.from_numpy(feats),
            torch.from_numpy(mask))


# ---- policy net ---------------------------------------------------------------

class _MapEncoder(nn.Module):
    """Small CNN: (GLOBAL_CH+AGENT_CH, MAP_H, MAP_W) -> (hidden,)."""
    def __init__(self, in_ch: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),  # H/2, W/2
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2),  # H/4, W/4
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, hidden),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class MATPolicy(nn.Module):
    """Centralized encoder + autoregressive decoder over agents.

    Encoder produces N agent tokens of size `hidden`. Decoder generates N
    discrete actions sequentially, each conditioned on the encoder output
    plus the previously decoded actions (teacher-forced at training time,
    sampled at rollout time).
    """

    def __init__(self, num_agents: int,
                 hidden: int = HIDDEN,
                 n_heads: int = N_HEADS,
                 n_layers: int = N_LAYERS,
                 k: int = K,
                 agent_feat_dim: int = AGENT_FEAT_DIM,
                 map_in_ch: int = GLOBAL_CH + AGENT_CH):
        super().__init__()
        self.num_agents = num_agents
        self.hidden = hidden
        self.k = k

        self.map_encoder = _MapEncoder(map_in_ch, hidden)
        self.feat_encoder = nn.Sequential(
            nn.Linear(agent_feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        # Encoder: 1 layer of self-attention over agent tokens.
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=2 * hidden,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.agent_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Decoder: autoregressive over agents. Memory = encoder output.
        # Input tokens carry previously decoded action embeddings.
        self.action_emb = nn.Embedding(k + 1, hidden)   # +1 for BOS
        self.bos_id = k
        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=2 * hidden,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.agent_decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)

        self.action_head = nn.Linear(hidden, k)
        self.value_head = nn.Linear(hidden, 1)

    # --- helpers ---------------------------------------------------------------

    def _encode(self, global_map, self_maps, agent_feats):
        """global_map: (B, GC, H, W), self_maps: (B, N, 1, H, W),
        agent_feats: (B, N, F). Returns (B, N, hidden)."""
        B, N = self_maps.shape[0], self_maps.shape[1]
        # Tile global_map to per-agent, concat with self channel.
        gm = global_map.unsqueeze(1).expand(-1, N, -1, -1, -1)      # (B,N,GC,H,W)
        maps = torch.cat([gm, self_maps], dim=2)                    # (B,N,GC+1,H,W)
        maps_flat = maps.reshape(B * N, maps.shape[2], MAP_H, MAP_W)
        map_emb = self.map_encoder(maps_flat).view(B, N, self.hidden)
        feat_emb = self.feat_encoder(agent_feats)                   # (B,N,hidden)
        tokens = map_emb + feat_emb                                 # (B,N,hidden)
        return self.agent_encoder(tokens)

    def _causal_mask(self, N, device):
        return torch.triu(torch.ones(N, N, device=device, dtype=torch.bool),
                          diagonal=1)

    # --- training forward ------------------------------------------------------

    def forward(self, global_map, self_maps, agent_feats,
                actions, candidate_mask):
        """Teacher-forced forward pass for PPO update.
        Shapes:
            global_map (B, GC, H, W), self_maps (B, N, 1, H, W),
            agent_feats (B, N, F), actions (B, N) long, mask (B, K) float.
        Returns:
            logits  (B, N, K)
            values  (B,)   single scalar value per state (team value)
        """
        B, N = self_maps.shape[0], self_maps.shape[1]
        enc = self._encode(global_map, self_maps, agent_feats)      # (B,N,H)

        # Build decoder input: BOS, then a_0..a_{N-2}. Shifted right.
        bos = torch.full((B, 1), self.bos_id, dtype=torch.long,
                         device=enc.device)
        shifted = torch.cat([bos, actions[:, :-1]], dim=1)           # (B,N)
        dec_in = self.action_emb(shifted)                            # (B,N,H)
        mask = self._causal_mask(N, enc.device)
        dec_out = self.agent_decoder(dec_in, enc, tgt_mask=mask)     # (B,N,H)

        logits = self.action_head(dec_out)                           # (B,N,K)
        # Mask invalid candidates to -1e9.
        m = candidate_mask.unsqueeze(1).expand(-1, N, -1)            # (B,N,K)
        logits = logits.masked_fill(m < 0.5, -1e9)
        # One team value from encoder summary (mean pool).
        value = self.value_head(enc.mean(dim=1)).squeeze(-1)         # (B,)
        return logits, value

    # --- rollout: sequential sampling -----------------------------------------

    @torch.no_grad()
    def act(self, global_map, self_maps, agent_feats, candidate_mask,
            deterministic: bool = False):
        """One-step rollout. Returns:
            actions (N,) long
            logprob (N,) float  (per-agent log p)
            value   ()  float   (team value)
        Inputs without batch dim are auto-expanded.
        """
        if global_map.dim() == 3:
            global_map = global_map.unsqueeze(0)
        if self_maps.dim() == 4:
            self_maps = self_maps.unsqueeze(0)
        if agent_feats.dim() == 2:
            agent_feats = agent_feats.unsqueeze(0)
        if candidate_mask.dim() == 1:
            candidate_mask = candidate_mask.unsqueeze(0)
        B, N = self_maps.shape[0], self_maps.shape[1]
        enc = self._encode(global_map, self_maps, agent_feats)

        prev = torch.full((B, 1), self.bos_id, dtype=torch.long,
                          device=enc.device)
        actions = []
        logprobs = []
        for i in range(N):
            dec_in = self.action_emb(prev)                          # (B,L,H)
            L = dec_in.shape[1]
            mask = self._causal_mask(L, enc.device)
            dec_out = self.agent_decoder(dec_in, enc, tgt_mask=mask)
            step_logits = self.action_head(dec_out[:, -1])          # (B,K)
            step_logits = step_logits.masked_fill(
                candidate_mask < 0.5, -1e9)
            probs = torch.softmax(step_logits.float(), dim=-1)
            probs = probs.clamp(min=1e-12)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            if deterministic:
                a = probs.argmax(dim=-1)
            else:
                a = torch.multinomial(probs, num_samples=1).squeeze(-1)
            lp = torch.log(probs.gather(-1, a.unsqueeze(-1))).squeeze(-1)
            actions.append(a)
            logprobs.append(lp)
            prev = torch.cat([prev, a.unsqueeze(-1)], dim=1)
        actions = torch.stack(actions, dim=1)                       # (B,N)
        logprobs = torch.stack(logprobs, dim=1)                     # (B,N)
        value = self.value_head(enc.mean(dim=1)).squeeze(-1)        # (B,)
        # Squeeze batch dim if it was implicit
        return (actions.squeeze(0), logprobs.squeeze(0),
                value.squeeze(0))


@dataclass
class MATConfig:
    num_agents: int = 5
    hidden: int = HIDDEN
    n_heads: int = N_HEADS
    n_layers: int = N_LAYERS
    k: int = K
    agent_feat_dim: int = AGENT_FEAT_DIM
    map_in_ch: int = GLOBAL_CH + AGENT_CH


def save_policy(policy: MATPolicy, cfg: MATConfig, path: str) -> None:
    torch.save({"state_dict": policy.state_dict(),
                "config": cfg.__dict__}, path)


def load_policy(path: str, map_location: str = "cpu") -> MATPolicy:
    blob = torch.load(path, map_location=map_location, weights_only=False)
    cfg = MATConfig(**blob["config"])
    pol = MATPolicy(num_agents=cfg.num_agents, hidden=cfg.hidden,
                    n_heads=cfg.n_heads, n_layers=cfg.n_layers,
                    k=cfg.k, agent_feat_dim=cfg.agent_feat_dim,
                    map_in_ch=cfg.map_in_ch)
    pol.load_state_dict(blob["state_dict"])
    pol.eval()
    return pol
