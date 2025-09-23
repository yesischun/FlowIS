from __future__ import annotations

import joblib 
import numpy as np
import gymnasium
from typing import Dict, Tuple
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import AbstractLane, CircularLane, LineType, StraightLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from sklearn.mixture import GaussianMixture
from highway_env.vehicle.behavior import FlowIS_Gostraight,IDMVehicle,FlowIS_TurnLeft
from highway_env.utils_FlowIS import MarginalGaussianMixture,RejectAcceptSampling
from highway_env.utils_FlowIS import NormalProposalDistribution,MixedLaplaceUniform

class InDEnv(AbstractEnv):

    ACTIONS: dict[int, str] = {0: "SLOWER", 1: "IDLE", 2: "FASTER"}
    ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        default_distribution=joblib.load(r'D:\LocalSyncdisk\加速测试\code\highway版本\模型参数集合\FlowIS_intersection.joblib')
        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 15,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "features_range": {
                        "x": [-100, 100],
                        "y": [-100, 100],
                        "vx": [-20, 20],
                        "vy": [-20, 20],
                    },
                    "absolute": True,
                    "flatten": False,
                    "observe_intentions": False,
                },
                "action": {
                    "type": "DiscreteMetaAction",
                    "longitudinal": True,
                    "lateral": True,
                    "target_speeds": [0, 20, 20],
                },
                "duration": 10,
                "simulation_frequency": 10,
                "policy_frequency": 10,
                # "duration": 13,  # [s]
                "destination": "o1",
                "controlled_vehicles": 1,
                "initial_vehicle_count": 10,
                "spawn_probability": 0.6,
                "screen_width": 600,
                "screen_height": 600,
                "centering_position": [0.5, 0.6],
                "scaling": 5.5 * 1.3,
                "collision_reward": -5,
                "high_speed_reward": 1,
                "arrived_reward": 1,
                "reward_speed_range": [7.0, 9.0],
                "normalize_reward": False,
                "offroad_terminal": False,
                "distribution":default_distribution
            }
        )
        return config

    def _reward(self, action: int) -> float:
        """Aggregated reward, for cooperative agents."""
        return sum(
            self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles
        ) / len(self.controlled_vehicles)

    def _rewards(self, action: int) -> dict[str, float]:
        """Multi-objective rewards, for cooperative agents."""
        agents_rewards = [
            self._agent_rewards(action, vehicle) for vehicle in self.controlled_vehicles
        ]
        return {
            name: sum(agent_rewards[name] for agent_rewards in agents_rewards)
            / len(agents_rewards)
            for name in agents_rewards[0].keys()
        }

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        """Per-agent reward signal."""
        rewards = self._agent_rewards(action, vehicle)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        reward = self.config["arrived_reward"] if rewards["arrived_reward"] else reward
        reward *= rewards["on_road_reward"]
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [self.config["collision_reward"], self.config["arrived_reward"]],
                [0, 1],
            )
        return reward

    def _agent_rewards(self, action: int, vehicle: Vehicle) -> dict[str, float]:
        """Per-agent per-objective reward signal."""
        scaled_speed = utils.lmap(
            vehicle.speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": vehicle.crashed,
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "arrived_reward": self.has_arrived(vehicle),
            "on_road_reward": vehicle.on_road,
        }

    def _is_terminated(self) -> bool:
        return (
            any(vehicle.crashed for vehicle in self.controlled_vehicles)
            or all(self.has_arrived(vehicle) for vehicle in self.controlled_vehicles)
            or (self.config["offroad_terminal"] and not self.vehicle.on_road)
        )

    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return vehicle.crashed or self.has_arrived(vehicle)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]

    def _info(self, obs: np.ndarray, action: int) -> dict:
        info = super()._info(obs, action)
        info["agents_rewards"] = tuple(
            self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles
        )
        info["agents_terminated"] = tuple(
            self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles
        )
        ego_vehi=self.road.vehicles[0]
        agent_vehi=self.road.vehicles[1]
        ego_position=ego_vehi.lane.local_coordinates(ego_vehi.position)
        agent_position=agent_vehi.lane.local_coordinates(agent_vehi.position)
        if ego_vehi.lane.speed_limit==41:
            ego_change=0
        elif ego_vehi.lane.speed_limit==42:
            ego_change=100
        else:
            ego_change=100+6.5*np.pi

        if agent_vehi.lane.speed_limit==41:
            agent_change=0
        elif agent_vehi.lane.speed_limit==42:
            agent_change=100
        else:
            agent_change=122
        info["scene_behavior"]=[ego_position[0]+ego_change,ego_vehi.speed,
                                agent_position[0]+agent_change,agent_vehi.speed,
                                agent_vehi.behavior]
        return info

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles(self.config["initial_vehicle_count"])

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = super().step(action)
        # self._clear_vehicles()
        # self._spawn_vehicle(spawn_probability=self.config["spawn_probability"])
        return obs, reward, terminated, truncated, info

    def _make_road(self) -> None:
        """
        Make an 4-way intersection.

        The horizontal road has the right of way. More precisely, the levels of priority are:
            - 3 for horizontal straight lanes and right-turns
            - 1 for vertical straight lanes and right-turns
            - 2 for horizontal left-turns
            - 0 for vertical left-turns

        The code for nodes in the road network is:
        (o:outer | i:inner + [r:right, l:left]) + (0:south | 1:west | 2:north | 3:east)

        :return: the intersection road
        """
        lane_width = AbstractLane.DEFAULT_WIDTH
        right_turn_radius = lane_width + 5  # [m}
        left_turn_radius = right_turn_radius + lane_width  # [m}
        outer_distance = right_turn_radius + lane_width / 2
        access_length = 50 + 50  # [m]

        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        for corner in range(4):
            angle = np.radians(90 * corner)
            is_horizontal = corner % 2
            priority = 3 if is_horizontal else 1
            rotation = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            # Incoming
            start = rotation @ np.array(
                [lane_width / 2, access_length + outer_distance]
            )
            end = rotation @ np.array([lane_width / 2, outer_distance])
            net.add_lane(
                "o" + str(corner),
                "ir" + str(corner),
                StraightLane(
                    start, end, line_types=[s, c], priority=priority, speed_limit=41
                ),
            )
            # Right turn
            r_center = rotation @ (np.array([outer_distance, outer_distance]))
            net.add_lane(
                "ir" + str(corner),
                "il" + str((corner - 1) % 4),
                CircularLane(
                    r_center,
                    right_turn_radius,
                    angle + np.radians(180),
                    angle + np.radians(270),
                    line_types=[n, c],
                    priority=priority,
                    speed_limit=42,
                ),
            )
            # Left turn
            l_center = rotation @ (
                np.array(
                    [
                        -left_turn_radius + lane_width / 2,
                        left_turn_radius - lane_width / 2,
                    ]
                )
            )
            net.add_lane(
                "ir" + str(corner),
                "il" + str((corner + 1) % 4),
                CircularLane(
                    l_center,
                    left_turn_radius,
                    angle + np.radians(0),
                    angle + np.radians(-90),
                    clockwise=False,
                    line_types=[n, n],
                    priority=priority - 1,
                    speed_limit=42,
                ),
            )
            # Straight
            start = rotation @ np.array([lane_width / 2, outer_distance])
            end = rotation @ np.array([lane_width / 2, -outer_distance])
            net.add_lane(
                "ir" + str(corner),
                "il" + str((corner + 2) % 4),
                StraightLane(
                    start, end, line_types=[s, n], priority=priority, speed_limit=42
                ),
            )
            # Exit
            start = rotation @ np.flip(
                [lane_width / 2, access_length + outer_distance], axis=0
            )
            end = rotation @ np.flip([lane_width / 2, outer_distance], axis=0)
            net.add_lane(
                "il" + str((corner - 1) % 4),
                "o" + str((corner - 1) % 4),
                StraightLane(
                    end, start, line_types=[n, c], priority=priority, speed_limit=43
                ),
            )

        road = RegulatedRoad(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def _make_vehicles(self, n_vehicles: int = 10) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane

        :return: the ego-vehicle
        """
        # self.NDE_intersection=joblib.load(self.config["distribution"])
        self.NDE_intersection=joblib.load(r'D:\LocalSyncdisk\加速测试\code\highway版本\模型参数集合\NDE_intersection.joblib')
        self.TDE_intersection=joblib.load(r'D:\LocalSyncdisk\加速测试\code\highway版本\模型参数集合\FlowIS_intersection.joblib')

        # generate initial scene
        while True:
            scene_behavior=self.TDE_intersection['scene_behavior'].sample(n_samples=1)
            scene_behavior=self.TDE_intersection['fit_scene_behavior'].inverse_transform(scene_behavior)[0]
            ttcp_lt = (13 * np.arccos(9/13) + 100  - scene_behavior[0] -2.4) / (scene_behavior[1]+1e-8)
            ttcp_gs = (122 - np.sqrt(13**2 - 9**2) - scene_behavior[2] -2.4) / (scene_behavior[3]+1e-8)
            if  ttcp_lt>0.2 and ttcp_gs>0.2 and 1>np.abs(ttcp_lt-ttcp_gs)>=0.2:
                break

        # generate ego_vehicle            
        if scene_behavior[0]<100:
            ego_lane = self.road.network.get_lane(("o0", "ir0", 0))
            lane_s=scene_behavior[0]
        elif 100<=scene_behavior[0]<100+6.5*np.pi:
            ego_lane = self.road.network.get_lane(("ir0", "il1", 0))   
            lane_s=scene_behavior[0]-100        
        else:
            ego_lane = self.road.network.get_lane(("il1", "o1", 0))   
            lane_s=scene_behavior[0]-100-6.5*np.pi

        self.controlled_vehicles = []        
        # ego_vehicle = self.action_type.vehicle_class(
        #     self.road,
        #     ego_lane.position(lane_s, scene_behavior[1] ),
        #     heading = ego_lane.heading_at(lane_s),
        #     speed   = scene_behavior[2]
        # )
        ego_vehicle = IDMVehicle(
                        self.road,
                        ego_lane.position(lane_s, 0),
                        speed   = scene_behavior[1],
                        heading = ego_lane.heading_at(lane_s),
                        target_speed=15
                        )  
        ego_vehicle.plan_route_to("o1")
        # print(ego_vehicle.route)
        self.controlled_vehicles.append(ego_vehicle)        
        self.road.vehicles.append(ego_vehicle)
        # ego_vehicle=FlowIS_TurnLeft(
        #     road=self.road,
        #     position= ego_lane.position(lane_s, 0),
        #     heading = ego_lane.heading_at(lane_s),
        #     speed   = scene_behavior[2],
        #     )
        # vehicle_left = ego_vehicle.create_from(ego_vehicle)
        # vehicle_left.NDE_intersection= self.AV_intersection
        # vehicle_left.plan_route_to("o1")
        # self.controlled_vehicles.append(vehicle_left)        
        # self.road.vehicles.append(vehicle_left)

        # generate challenge_vehicle   
        if scene_behavior[2]<100:
            lane = self.road.network.get_lane(("o2", "ir2", 0))
            lane_s=scene_behavior[2]
        elif 100<=scene_behavior[2]<122:
            lane = self.road.network.get_lane(("ir2", "il0", 0))
            lane_s=scene_behavior[2]-100
        else:
            lane = self.road.network.get_lane(("il0", "o0", 0))   
            lane_s=scene_behavior[2]-122
        carfollow_is=FlowIS_Gostraight(
            road=self.road,
            position= lane.position(lane_s, 0),
            heading = lane.heading_at(lane_s),
            speed   = scene_behavior[3],
            )
        vehicle_follow = carfollow_is.create_from(carfollow_is)
        vehicle_follow.NDE_intersection= self.TDE_intersection
        vehicle_follow.plan_route_to("o0")
        self.road.vehicles.append(vehicle_follow)

    # def _spawn_vehicle(
    #     self,
    #     longitudinal: float = 0,
    #     position_deviation: float = 1.0,
    #     speed_deviation: float = 1.0,
    #     spawn_probability: float = 0.6,
    #     go_straight: bool = False,
    # ) -> None:
    #     if self.np_random.uniform() > spawn_probability:
    #         return

    #     route = self.np_random.choice(range(4), size=2, replace=False)
    #     route[1] = (route[0] + 2) % 4 if go_straight else route[1]
    #     vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
    #     vehicle = vehicle_type.make_on_lane(
    #         self.road,
    #         ("o" + str(route[0]), "ir" + str(route[0]), 0),
    #         longitudinal=(
    #             longitudinal + 5 + self.np_random.normal() * position_deviation
    #         ),
    #         speed=8 + self.np_random.normal() * speed_deviation,
    #     )
    #     for v in self.road.vehicles:
    #         if np.linalg.norm(v.position - vehicle.position) < 15:
    #             return
    #     vehicle.plan_route_to("o" + str(route[1]))
    #     vehicle.randomize_behavior()
    #     self.road.vehicles.append(vehicle)
    #     return vehicle

    # def _clear_vehicles(self) -> None:
    #     is_leaving = (
    #         lambda vehicle: "il" in vehicle.lane_index[0]
    #         and "o" in vehicle.lane_index[1]
    #         and vehicle.lane.local_coordinates(vehicle.position)[0]
    #         >= vehicle.lane.length - 4 * vehicle.LENGTH
    #     )
    #     self.road.vehicles = [
    #         vehicle
    #         for vehicle in self.road.vehicles
    #         if vehicle in self.controlled_vehicles
    #         or not (is_leaving(vehicle) or vehicle.route is None)
    #     ]

    def has_arrived(self, vehicle: Vehicle, exit_distance: float = 25) -> bool:
        return (
            "il" in vehicle.lane_index[0]
            and "o" in vehicle.lane_index[1]
            and vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance
        )

gymnasium.register(
    id='InDEnv-v0',
    entry_point='highway_env.envs:InDEnv',
)