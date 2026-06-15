import os
import argparse
import json
import shutil
# import utils
import random
import pickle
import numpy as np

def _load_obstacle(scene):
    pkl = f"assets/scenes/{scene}/obstacle_grid.pkl"
    data = pickle.load(open(pkl, 'rb'))
    grid = data["grid"]
    params = data["parameters"]
    return grid, params

def _point_invalid(grid, params, pt):
    i = int((pt[0] - params["min_x"]) / params["resolution"])
    j = int((pt[1] - params["min_y"]) / params["resolution"])
    if i < 0 or i >= params["nx"] or j < 0 or j >= params["ny"]:
        return True
    return grid[i, j] == 1

def _chebyshev_ok(p, q, lo=20.0, hi=100.0):
    dx = abs(p[0] - q[0])
    dy = abs(p[1] - q[1])
    d = max(dx, dy)
    return (d >= lo) and (d <= hi)

def _sample_valid_pair(grid, params, threshold=None):
    while True:
        if threshold is None:
            x = random.uniform(-300, 300)
            y = random.uniform(-300, 300)
        else:
            x = random.uniform(threshold[0], threshold[1])
            y = random.uniform(threshold[2], threshold[3])
        if _point_invalid(grid, params, (x, y)):
            continue
        if threshold is None:
            rx = random.uniform(-300, 300)
            ry = random.uniform(-300, 300)
        else:
            rx = random.uniform(threshold[0], threshold[1])
            ry = random.uniform(threshold[2], threshold[3])
        if _point_invalid(grid, params, (rx, ry)):
            continue
        if not _chebyshev_ok((x, y), (rx, ry)):
            continue
        return (x, y), (rx, ry)
    
def annotate_all_patrol_overwrite(scene, num_agents):
    sentinel_config_path = f"ViCo/assets/scenes/{scene}/sentinel_config/sentinel_config_patrol.json"
    sentinel_config={'agent_names': [], 'agent_infos': [], 'agent_poses': [], 'locator_colors': [], 'locator_colors_rgb': [], 'agent_skins': [], 'patrol_config': []}
    with open(f"ViCo/assets/scenes/{scene}/agents_num_5/config.json", "r") as f:
        config = json.load(f)
        height = config['agent_infos'][0]['outdoor_pose'][2]+10
    grid, params = _load_obstacle(scene)
    for i in range(num_agents):
        (x, y), (rx, ry) = _sample_valid_pair(grid, params)
        sentinel_config['agent_names'].append(f"Sentinel {i}")
        sentinel_config['agent_infos'].append({
                "cash": 1000,
                "held_objects": [
                    None,
                    None
                ],
                "outdoor_pose": [
                    x,
                    y,
                    height,
                    0.0,
                    0.0,
                    0.0
                ],
                "current_building": "open space",
                "current_place": None,
                "current_vehicle": None
            })
        sentinel_config['agent_poses'].append([
                x,
                y,
                height,
                0.0,
                0.0,
                0.0
            ])
        sentinel_config['locator_colors'].append('cyan')
        sentinel_config['locator_colors_rgb'].append([
                0.0,
                1.0,
                1.0
            ])
        sentinel_config['agent_skins'].append("avatars/models/mixamo_sentinel.glb")
        sentinel_config['patrol_config'].append({
                "type": "patrolling",
                "route": [
                    [x, y],
                    [rx, ry]
                ],
                "route_index": 0
            })
    with open(sentinel_config_path, "w") as f:
        json.dump(sentinel_config, f, indent=4)

def annotate_all_rotate(scene, coor_list):
    sentinel_config_path = f"ViCo/assets/scenes/{scene}/sentinel_config/sentinel_config_stationary.json"
    if os.path.exists(sentinel_config_path):
        sentinel_config=json.load(open(sentinel_config_path, "r"))
    else:
        sentinel_config={'agent_names': [], 'agent_infos': [], 'agent_poses': [], 'locator_colors': [], 'locator_colors_rgb': [], 'agent_skins': [], 'patrol_config': []}
    # height_field_path = f"Genesis/genesis/assets/ViCo/..."
    # height_field = utils.load_height_field(height_field_path)
    # height = utils.get_height_at(height_field, args.x, args.y)
    origin_num = len(sentinel_config['agent_names'])
    for i in range(len(coor_list)):
        x, y = coor_list[i][0], coor_list[i][1]
        with open(f"ViCo/assets/scenes/{scene}/agents_num_5/config.json", "r") as f:
            config = json.load(f)
            height = config['agent_infos'][0]['outdoor_pose'][2]+10
        sentinel_config['agent_names'].append(f"Sentinel {i+origin_num}")
        sentinel_config['agent_infos'].append({
                "cash": 1000,
                "held_objects": [
                    None,
                    None
                ],
                "outdoor_pose": [
                    x,
                    y,
                    height,
                    0.0,
                    0.0,
                    0.0
                ],
                "current_building": "open space",
                "current_place": None,
                "current_vehicle": None
            })
        sentinel_config['agent_poses'].append([
                x,
                y,
                height,
                0.0,
                0.0,
                0.0
            ])
        sentinel_config['locator_colors'].append('cyan')
        sentinel_config['locator_colors_rgb'].append([
                0.0,
                1.0,
                1.0
            ])
        sentinel_config['agent_skins'].append("avatars/models/mixamo_sentinel.glb")
        sentinel_config['patrol_config'].append({
                "type": "rotating"
            })
        with open(sentinel_config_path, "w") as f:
            json.dump(sentinel_config, f, indent=4)
        # for x in os.listdir(f"ViCo/assets/scenes/{scene}/agents_num_5/"):
        #     if os.path.exists(f"ViCo/assets/scenes/{scene}/agents_num_5/Sentinel {i+origin_num}"): continue
        #     if os.path.isdir(f"ViCo/assets/scenes/{scene}/agents_num_5/{x}"):
        #         shutil.copytree(f"ViCo/assets/scenes/{scene}/agents_num_5/{x}", f"ViCo/assets/scenes/{scene}/agents_num_5/Sentinel {i+origin_num}")
        #         break

def annotate_all_around_specific_place(scene, sentinel_type, num_agents, center_coords=None):
    sentinel_config_path = f"assets/scenes/{scene}/sentinel_config/sentinel_config_{sentinel_type}.json"
    sentinel_config=json.load(open(sentinel_config_path, "r"))
    with open(f"assets/scenes/{scene}/agents_num_15/config.json", "r") as f:
        config = json.load(f)
        height = config['agent_infos'][0]['outdoor_pose'][2]+10
    grid, params = _load_obstacle(scene)
    for i in range(num_agents):
        if center_coords is None:
            (x, y), (rx, ry) = _sample_valid_pair(grid, params)
        else:
            (x, y), (rx, ry) = _sample_valid_pair(grid, params, threshold=[center_coords[0]-50, center_coords[0]+50, center_coords[1]-50, center_coords[1]+50])
        sentinel_config['agent_infos'][i]={
                "cash": 1000,
                "held_objects": [
                    None,
                    None
                ],
                "outdoor_pose": [
                    x,
                    y,
                    height,
                    0.0,
                    0.0,
                    0.0
                ],
                "current_building": "open space",
                "current_place": None,
                "current_vehicle": None
            }
        sentinel_config['agent_poses'][i]=[
                x,
                y,
                height,
                0.0,
                0.0,
                0.0
            ]
        sentinel_config['patrol_config'][i]={
                "type": "patrolling",
                "route": [
                    [x, y],
                    [rx, ry]
                ],
                "route_index": 0
            } if sentinel_type == 'patrol' else {
                "type": "rotating"
            }
    with open(sentinel_config_path, "w") as f:
        json.dump(sentinel_config, f, indent=4)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", "-s", type=str)
    parser.add_argument("--num", "-n", type=int, default=5)
    args = parser.parse_args()

    annotate_all_around_specific_place(args.scene, 'stationary', args.num)
    annotate_all_around_specific_place(args.scene, 'patrol', args.num)

'''
AMSTERDAM
ClubNl

AUSTIN
Thistle Cafe
[-128.64605331   60.58452034]
BERLIN
Fröbel-Kita im BMG
[ 202.5439682  -199.63046265]
COLUMBUS
Fresh and Snacks Store
[17.34697835793589, 227.3699038467451]
DETROIT
Eatóri Market
[-159.06414795  132.58945847]
EL_PASO
Snacks and DM Store
[-0.2735180356440061, 91.37901884994146]
HAMBURG
Subway
[ 13.24249268 -25.37106371]
HARVARD
Behind VA Shadows Gallery
[ -34.72005653 -202.03935242]
LONDON
Westminster School of Performing Arts
[-202.97782135 -260.19155884]
LONGISLAND
Franklin Square Pharmacy
[ 98.39226532 -40.88065243]
MADISON
306 West Main
[ -19.16016006 -175.14601135]
MADRID
LI-ONNA
[46.78588676 -7.76908445]
'''