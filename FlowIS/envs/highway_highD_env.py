from __future__ import annotations
import joblib 
import gymnasium
import numpy as np
from highway_env.vehicle.behavior import  FlowIS_Follow,IDMVehicle
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

Observation = np.ndarray
def calculate_ttc(scene,lenth):
    """
    计算追尾场景的碰撞时间(TTC)，支持批量处理
    
    参数:
        scene: 形状为(N, 3)的二维numpy数组，每行包含[后车速度, 前车速度, 两车距离]
    
    返回:
        形状为(N, 4)的二维numpy数组，新增的第四列为计算得到的TTC值
    """
    # 提取速度和距离
    speed_rear = scene[:, 0]
    speed_front = scene[:, 1]
    distance = scene[:, 2]-lenth
    
    # 计算速度差
    speed_diff = speed_front - speed_rear
    
    # 处理速度差非负的情况（安全场景）
    safe_mask = speed_diff >= 0
    
    # 初始化TTC数组，安全场景设为999，危险场景计算TTC
    ttc = np.empty_like(speed_diff, dtype=float)
    ttc[safe_mask] = 999
    ttc[~safe_mask] = distance[~safe_mask] / (-speed_diff[~safe_mask])
    
    # 将TTC添加为新列并返回
    return np.column_stack((scene, ttc))

class HighDEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "duration": 10,  # [s]
                "simulation_frequency": 10,
                "policy_frequency": 10,                  
                "observation": {"type": "Kinematics"},
                "action": {"type": "DiscreteMetaAction",},
                # "action": {"type": "ContinuousAction",},
                "lanes_count": 1,
                "vehicles_count": 50,
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "ego_spacing": 2,
                "vehicles_density": 1,
                
                "collision_reward": -10,  # The reward received when colliding with a vehicle.
                "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                # zero for other lanes.
                "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                # lower speeds according to config["reward_speed_range"].
                "lane_change_reward": 0,  # The reward received at each lane change action.
                "reward_speed_range": [20, 30],
                "normalize_reward": True,
                "offroad_terminal": False,
            }
        )
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                self.config["lanes_count"], speed_limit=None
            ),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        self.TDE_highway=joblib.load(r'D:\LocalSyncdisk\加速测试\code\highway版本\模型参数集合\FlowIS_highway.joblib')
        self.NDE_highway=joblib.load(r'D:\LocalSyncdisk\加速测试\code\highway版本\模型参数集合\NDE_highway.joblib')
        while True:
            scene_behavior=self.TDE_highway['scene_behavior'].sample(n_samples=1)
            scene_behavior=self.TDE_highway['fit_scene_behavior'].inverse_transform(scene_behavior)
            ttc=calculate_ttc(scene_behavior,5)[0][4]
            if ttc>0.3:
                break
        self.controlled_vehicles = []   
        # ego_vehicle = self.action_type.vehicle_class(
        #                     self.road,
        #                     position=np.array([100.0,0.0]),
        #                     speed=scene_behavior[0][0],
        #                     heading=0.0
        #                     )     
        ego_vehicle = IDMVehicle(
                        self.road,
                        position=np.array([100.0,0.0]),
                        speed=scene_behavior[0][0],
                        heading=0.0,
                        target_speed=38
                        )  
        self.controlled_vehicles = []        
        self.controlled_vehicles.append(ego_vehicle)
        self.road.vehicles.append(ego_vehicle)


        # 生成背景车
        bv_vehicle=FlowIS_Follow(
            road=self.road,
            position= np.array([100.0+scene_behavior[0][2],0.0]),
            speed   = scene_behavior[0][1],            
            heading = 0.0,
            )
        vehicle_follow = bv_vehicle.create_from(bv_vehicle)
        vehicle_follow.NDE_distribution=self.NDE_highway
        # vehicle_follow.behavior= self.config["behavior"]
        self.road.vehicles.append(vehicle_follow) 

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [
                    self.config["collision_reward"],
                    self.config["high_speed_reward"] + self.config["right_lane_reward"],
                ],
                [0, 1],
            )
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: Action) -> dict[str, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = (
            self.vehicle.target_lane_index[2]
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index[2]
        )
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road),
        }

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (
            self.vehicle.crashed
            or self.config["offroad_terminal"]
            and not self.vehicle.on_road
        )

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]
    
    def _info(self, obs, action = None):
        info=super()._info(obs, action)
        av=self.road.vehicles[0]
        bv=self.road.vehicles[1]
        info['scene_behavior']=[av.speed,bv.speed,bv.position[0]-av.position[0],bv.behavior]
        return info


gymnasium.register(
    id='HighDEnv-v0',
    entry_point='highway_env.envs:HighDEnv',
)