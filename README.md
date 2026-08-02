# Sentinel: Embodied Cooperative Spatial Reasoning and Planning

![Teaser](teaser.png)

This repo contains code for the following paper:

_Xiangye Lin\*, Hongxin Zhang\*, Ruxi Deng, Qinhong Zhou, Chuang Gan_: Sentinel: Embodied Cooperative Spatial Reasoning and Planning

Paper: [https://arxiv.org/pdf/2605.26239](https://arxiv.org/pdf/2605.26239)

Project Website: _coming soon_

We introduce the **Sentinel Challenge**, a benchmark for studying _Cooperative Spatial Intelligence_, in which multiple decentralized embodied agents must communicate in natural language to agree on a mutually safe and convenient meeting point in city-scale outdoor environments, then navigate there while avoiding dynamic sentinels patrolling the scene with only a coarse map tool for spatial guidance. To address this problem, we propose **CoSaR**, a cooperative spatial reasoning and planning framework that bridges the communication and planning strengths of foundation models with classical spatial navigation algorithms. CoSaR maintains a dynamic spatial memory of poses, ETAs, occupancy, and sentinel danger zones, and uses a spatial-aware reasoning module to decide when to communicate, query the map tool, navigate, or wait. Across 14 city-scale scenes with 3–5 agents, CoSaR consistently achieves faster gathering, shorter path lengths, and improved safety over strong baselines.


## Installation

The Sentinel Challenge is built on top of [Virtual Community](https://github.com/UMass-Embodied-AGI/Virtual-Community). The simulator runs on Python 3.11, CUDA 11.7, Ubuntu 24.04. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run the one-shot setup from the repository root:

```bash
bash setup.sh
```

This fetches the submodules, builds the Virtual Community environment with `uv` (installing Genesis at its pinned commit), and layers the release-only dependencies — vLLM and the scene-graph perception backbones — in the order required to avoid known build conflicts. See `setup.sh` for the individual steps if you need to run them manually.

Activate the environment in every new shell:

```bash
source Virtual-Community/.venv/bin/activate
```

### Assets

Virtual Community assets are hosted on HuggingFace at [Virtual-Community-AI/assets](https://huggingface.co/datasets/Virtual-Community-AI/assets) and are **downloaded automatically** the first time you run a scene. The DETROIT scene used by this release is published there, so no manual setup is required.

To pre-fetch all assets in advance:

```bash
python -c "from vico.tools.asset_utils import download_all_assets; download_all_assets()"
```

For indoor scenes from GRUtopia, follow [their instructions](https://github.com/OpenRobotLab/GRUtopia?tab=readme-ov-file#%EF%B8%8F-assets) to download `commercial_scenes.zip` and unzip it under `Virtual-Community/vico/assets/ViCo/scene/`. Otherwise pass `--no_load_indoor_scene` to the runners.

### System Requirements

- **RAM**: 24 GB minimum / 32 GB recommended
- **VRAM**: 10 GB minimum / 16 GB recommended
- **Disk**: 60 GB minimum / 100 GB recommended


## Run Experiments

The main implementation of _CoSaR_ is in `agents/sentinel_challenge/CoSaR.py` (agent), `agents/sentinel_challenge/base_nav.py` (shared navigation base), and `agents/sentinel_challenge/meeting_prompts/cosar_prompts/` (LLM prompts). The challenge entry point is `sentinel_challenge/challenge.py`.

Runners live under `sentinel_challenge/`:

| Folder | Perception |
|---|---|
| `sentinel_challenge/gt_scripts/` | Ground-truth segmentation for all agents (`--enable_gt_segmentation`) |
| `sentinel_challenge/no_gt_scripts/` | Ground-truth segmentation restricted to sentinels (`--gt_only_for_sentinels`); other agents use learned perception |

Each folder contains one script per method (`cosar` = our method, plus baselines `mcts`, `roco`, `coela`, `center`, `center_no_avoidance`, `fixed`, `rl`, `mat`, and CoSaR ablations `_no_analyzer`, `_no_emergency_avoidance`, `_no_refine`, `_no_spatial_memory`, `_qwen`).

For example, to run CoSaR with 3 agents and 1 stationary sentinel in DETROIT under the oracle-perception setting:

```bash
bash sentinel_challenge/gt_scripts/run_cosar.sh DETROIT 3 stationary 1 0
```

Same configuration without oracle perception:

```bash
bash sentinel_challenge/no_gt_scripts/run_cosar.sh DETROIT 3 stationary 1 0
```


## Responsible Use and Ethical Considerations

Sentinel is intended to support research on cooperative embodied intelligence in safety-relevant scenarios, including search-and-rescue, disaster response, and autonomous logistics. It provides a standardized, controlled environment for studying multi-agent planning, communication, and reasoning under partial observability and dynamic risk.

The benchmark is entirely simulation-based and contains only synthetic agents and environments. It does not use or release real-world human data, biometric information, or personally identifiable information. Accordingly, the primary ethical concern is not the benchmark data itself, but the possible transfer of cooperative navigation and reasoning methods to harmful real-world applications.

These capabilities are inherently dual-use. In particular, the following applications are outside the intended scope of this project:

- Mass surveillance
- Biometric identification
- Predictive policing
- Tracking or monitoring identifiable individuals without appropriate authorization

We ask users to apply Sentinel only for lawful, authorized, and ethically reviewed purposes, with safeguards appropriate to their deployment context. Concerns about misuse, documentation, or other responsible-use issues can be reported through the repository's [issue tracker](https://github.com/UMass-Embodied-AGI/Sentinel/issues). We will maintain this guidance through versioned updates as the benchmark and its uses evolve.


## Citation

If you find our work useful, please consider citing:

```bibtex
@misc{lin2026sentinel,
  title  = {Sentinel: Embodied Cooperative Spatial Reasoning and Planning},
  author = {Lin, Xiangye and Zhang, Hongxin and Deng, Ruxi and Zhou, Qinhong and Gan, Chuang},
  year   = {2026},
}
```


## Acknowledgement

The Sentinel Challenge is built on the [Virtual Community](https://github.com/UMass-Embodied-AGI/Virtual-Community) platform; we thank its authors and the underlying open-source projects it depends on, including [Genesis](https://github.com/Genesis-Embodied-AI/Genesis), [GRUtopia](https://github.com/OpenRobotLab/GRUtopia), [Google 3D Tiles](https://3d-tiles.web.app/), [OpenStreetMap](https://www.openstreetmap.org/), and [Blender](https://www.blender.org/). Our baselines build on [CoELA](https://github.com/UMass-Embodied-AGI/CoELA), [RoCo](https://github.com/MandiZhao/robot-collab), and the [Multi-Agent Transformer](https://github.com/PKU-MARL/Multi-Agent-Transformer).
