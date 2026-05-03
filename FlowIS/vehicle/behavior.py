from __future__ import annotations
from typing import Tuple, Union
import numpy as np
import math
from joblib import load
from sklearn.preprocessing import MinMaxScaler
from highway_env import utils
from highway_env.road.road import LaneIndex, Road, Route
from highway_env.utils import Vector
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.sampling import RejectAcceptSampling
from highway_env.vehicle.distributions import MarginalGaussianMixture, NormalProposalDistribution
from highway_env.road.lane import AbstractLane
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

def position_to_lane(ego_vehi,agent_vehi):
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

    return [ego_position[0]+ego_change,ego_vehi.speed,
            agent_position[0]+agent_change,agent_vehi.speed]


def _safe_log_conditional(distribution: dict, scene_behavior_norm: np.ndarray, scene_norm: np.ndarray, dims: list[int]) -> float | None:
    """Compute log p(b|s) from joint and marginal log-pdfs in normalized space."""
    try:
        scene_behavior_norm = np.asarray(scene_behavior_norm).reshape(1, -1)
        scene_norm = np.asarray(scene_norm).reshape(1, -1)
        log_joint = distribution["scene_behavior"].logpdf(scene_behavior_norm)
        log_marginal = distribution["scene_behavior"].logpdf_marginal(scene_norm, dims)
        return float(np.asarray(log_joint - log_marginal).reshape(-1)[0])
    except Exception:
        return None

class IDMVehicle(ControlledVehicle):
    """
    A vehicle using both a longitudinal and a lateral decision policies.

    - Longitudinal: the IDM model computes an acceleration given the preceding vehicle's distance and speed.
    - Lateral: the MOBIL model decides when to change lane by maximizing the acceleration of nearby vehicles.
    """

    # Longitudinal policy parameters
    ACC_MAX = 6.0  # [m/s2]
    """Maximum acceleration."""

    COMFORT_ACC_MAX = 3.0  # [m/s2]
    """Desired maximum acceleration."""

    COMFORT_ACC_MIN = -5.0  # [m/s2]
    """Desired maximum deceleration."""

    DISTANCE_WANTED = 5.0 + ControlledVehicle.LENGTH  # [m]
    """Desired jam distance to the front vehicle."""

    TIME_WANTED = 2  # [s]1.5
    """Desired time gap to the front vehicle."""

    DELTA = 4.0  # []
    """Exponent of the velocity term."""

    DELTA_RANGE = [3.5, 4.5]
    """Range of delta when chosen randomly."""

    # Lateral policy parameters
    POLITENESS = 0.0  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]

    def __init__(
        self,
        road: Road,
        position: Vector,
        heading: float = 0,
        speed: float = 0,
        target_lane_index: int = None,
        target_speed: float = None,
        route: Route = None,
        enable_lane_change: bool = True,
        timer: float = None,
    ):
        super().__init__(
            road, position, heading, speed, target_lane_index, target_speed, route
        )
        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.position) * np.pi) % self.LANE_CHANGE_DELAY

    def randomize_behavior(self):
        self.DELTA = self.road.np_random.uniform(
            low=self.DELTA_RANGE[0], high=self.DELTA_RANGE[1]
        )

    @classmethod
    def create_from(cls, vehicle: ControlledVehicle) -> IDMVehicle:
        """
        Create a new vehicle from an existing one.

        The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(
            vehicle.road,
            vehicle.position,
            heading=vehicle.heading,
            speed=vehicle.speed,
            target_lane_index=vehicle.target_lane_index,
            target_speed=vehicle.target_speed,
            route=vehicle.route,
            timer=getattr(vehicle, "timer", None),
        )
        return v

    def act(self, action: dict | str = None):
        """
        Execute an action.

        For now, no action is supported because the vehicle takes all decisions
        of acceleration and lane changes on its own, based on the IDM and MOBIL models.

        :param action: the action
        """
        if self.crashed:
            return
        action = {}
        # Lateral: MOBIL
        self.follow_road()
        if self.enable_lane_change:
            self.change_lane_policy()
        action["steering"] = self.steering_control(self.target_lane_index)
        action["steering"] = np.clip(
            action["steering"], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE
        )

        # Longitudinal: IDM
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(
            self, self.lane_index
        )
        action["acceleration"] = self.acceleration(
            ego_vehicle=self, front_vehicle=front_vehicle, rear_vehicle=rear_vehicle
        )
        # When changing lane, check both current and target lanes
        if self.lane_index != self.target_lane_index:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(
                self, self.target_lane_index
            )
            target_idm_acceleration = self.acceleration(
                ego_vehicle=self, front_vehicle=front_vehicle, rear_vehicle=rear_vehicle
            )
            action["acceleration"] = min(
                action["acceleration"], target_idm_acceleration
            )
        # action['acceleration'] = self.recover_from_stop(action['acceleration'])
        action["acceleration"] = np.clip(
            action["acceleration"], -self.ACC_MAX, self.ACC_MAX
        )
        # Skip ControlledVehicle.act(), or the command will be overridden.
        Vehicle.act(self, action)

    def step(self, dt: float):
        """
        Step the simulation.

        Increases a timer used for decision policies, and step the vehicle dynamics.

        :param dt: timestep
        """
        self.timer += dt
        super().step(dt)

    def acceleration(
        self,
        ego_vehicle: ControlledVehicle,
        front_vehicle: Vehicle = None,
        rear_vehicle: Vehicle = None,
    ) -> float:
        """
        Compute an acceleration command with the Intelligent Driver Model.

        The acceleration is chosen so as to:
        - reach a target speed;
        - maintain a minimum safety distance (and safety time) w.r.t the front vehicle.

        :param ego_vehicle: the vehicle whose desired acceleration is to be computed. It does not have to be an
                            IDM vehicle, which is why this method is a class method. This allows an IDM vehicle to
                            reason about other vehicles behaviors even though they may not IDMs.
        :param front_vehicle: the vehicle preceding the ego-vehicle
        :param rear_vehicle: the vehicle following the ego-vehicle
        :return: the acceleration command for the ego-vehicle [m/s2]
        """
        if not ego_vehicle or not isinstance(ego_vehicle, Vehicle):
            return 0
        ego_target_speed = getattr(ego_vehicle, "target_speed", 0)
        if ego_vehicle.lane and ego_vehicle.lane.speed_limit is not None:
            ego_target_speed = np.clip(
                ego_target_speed, 0, ego_vehicle.lane.speed_limit
            )
        acceleration = self.COMFORT_ACC_MAX * (
            1
            - np.power(
                max(ego_vehicle.speed, 0) / abs(utils.not_zero(ego_target_speed)),
                self.DELTA,
            )
        )

        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle)
            acceleration -= self.COMFORT_ACC_MAX * np.power(
                self.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2
            )
        return acceleration

    def desired_gap(
        self,
        ego_vehicle: Vehicle,
        front_vehicle: Vehicle = None,
        projected: bool = True,
    ) -> float:
        """
        Compute the desired distance between a vehicle and its leading vehicle.

        :param ego_vehicle: the vehicle being controlled
        :param front_vehicle: its leading vehicle
        :param projected: project 2D velocities in 1D space
        :return: the desired distance between the two [m]
        """
        d0 = self.DISTANCE_WANTED
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = (
            np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction)
            if projected
            else ego_vehicle.speed - front_vehicle.speed
        )
        d_star = (
            d0 + ego_vehicle.speed * tau + ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        )
        return d_star

    def change_lane_policy(self) -> None:
        """
        Decide when to change lane.

        Based on:
        - frequency;
        - closeness of the target lane;
        - MOBIL model.
        """
        # If a lane change is already ongoing
        if self.lane_index != self.target_lane_index:
            # If we are on correct route but bad lane: abort it if someone else is already changing into the same lane
            if self.lane_index[:2] == self.target_lane_index[:2]:
                for v in self.road.vehicles:
                    if (
                        v is not self
                        and v.lane_index != self.target_lane_index
                        and isinstance(v, ControlledVehicle)
                        and v.target_lane_index == self.target_lane_index
                    ):
                        d = self.lane_distance_to(v)
                        d_star = self.desired_gap(self, v)
                        if 0 < d < d_star:
                            self.target_lane_index = self.lane_index
                            break
            return

        # else, at a given frequency,
        if not utils.do_every(self.LANE_CHANGE_DELAY, self.timer):
            return
        self.timer = 0

        # decide to make a lane change
        for lane_index in self.road.network.side_lanes(self.lane_index):
            # Is the candidate lane close enough?
            if not self.road.network.get_lane(lane_index).is_reachable_from(
                self.position
            ):
                continue
            # Only change lane when the vehicle is moving
            if np.abs(self.speed) < 1:
                continue
            # Does the MOBIL model recommend a lane change?
            if self.mobil(lane_index):
                self.target_lane_index = lane_index

    def mobil(self, lane_index: LaneIndex) -> bool:
        """
        MOBIL lane change model: Minimizing Overall Braking Induced by a Lane change

            The vehicle should change lane only if:
            - after changing it (and/or following vehicles) can accelerate more;
            - it doesn't impose an unsafe braking on its new following vehicle.

        :param lane_index: the candidate lane for the change
        :return: whether the lane change should be performed
        """
        # Is the maneuver unsafe for the new following vehicle?
        new_preceding, new_following = self.road.neighbour_vehicles(self, lane_index)
        new_following_a = self.acceleration(
            ego_vehicle=new_following, front_vehicle=new_preceding
        )
        new_following_pred_a = self.acceleration(
            ego_vehicle=new_following, front_vehicle=self
        )
        if new_following_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return False

        # Do I have a planned route for a specific lane which is safe for me to access?
        old_preceding, old_following = self.road.neighbour_vehicles(self)
        self_pred_a = self.acceleration(ego_vehicle=self, front_vehicle=new_preceding)
        if self.route and self.route[0][2] is not None:
            # Wrong direction
            if np.sign(lane_index[2] - self.target_lane_index[2]) != np.sign(
                self.route[0][2] - self.target_lane_index[2]
            ):
                return False
            # Unsafe braking required
            elif self_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
                return False

        # Is there an acceleration advantage for me and/or my followers to change lane?
        else:
            self_a = self.acceleration(ego_vehicle=self, front_vehicle=old_preceding)
            old_following_a = self.acceleration(
                ego_vehicle=old_following, front_vehicle=self
            )
            old_following_pred_a = self.acceleration(
                ego_vehicle=old_following, front_vehicle=old_preceding
            )
            jerk = (
                self_pred_a
                - self_a
                + self.POLITENESS
                * (
                    new_following_pred_a
                    - new_following_a
                    + old_following_pred_a
                    - old_following_a
                )
            )
            if jerk < self.LANE_CHANGE_MIN_ACC_GAIN:
                return False

        # All clear, let's go!
        return True

    def recover_from_stop(self, acceleration: float) -> float:
        """
        If stopped on the wrong lane, try a reversing maneuver.

        :param acceleration: desired acceleration from IDM
        :return: suggested acceleration to recover from being stuck
        """
        stopped_speed = 5
        safe_distance = 200
        # Is the vehicle stopped on the wrong lane?
        if self.target_lane_index != self.lane_index and self.speed < stopped_speed:
            _, rear = self.road.neighbour_vehicles(self)
            _, new_rear = self.road.neighbour_vehicles(
                self, self.road.network.get_lane(self.target_lane_index)
            )
            # Check for free room behind on both lanes
            if (not rear or rear.lane_distance_to(self) > safe_distance) and (
                not new_rear or new_rear.lane_distance_to(self) > safe_distance
            ):
                # Reverse
                return -self.COMFORT_ACC_MAX / 2
        return acceleration

class LinearVehicle(IDMVehicle):
    """A Vehicle whose longitudinal and lateral controllers are linear with respect to parameters."""

    ACCELERATION_PARAMETERS = [0.3, 0.3, 2.0]
    STEERING_PARAMETERS = [
        ControlledVehicle.KP_HEADING,
        ControlledVehicle.KP_HEADING * ControlledVehicle.KP_LATERAL,
    ]

    ACCELERATION_RANGE = np.array(
        [
            0.5 * np.array(ACCELERATION_PARAMETERS),
            1.5 * np.array(ACCELERATION_PARAMETERS),
        ]
    )
    STEERING_RANGE = np.array(
        [
            np.array(STEERING_PARAMETERS) - np.array([0.07, 1.5]),
            np.array(STEERING_PARAMETERS) + np.array([0.07, 1.5]),
        ]
    )

    TIME_WANTED = 2.5

    def __init__(
        self,
        road: Road,
        position: Vector,
        heading: float = 0,
        speed: float = 0,
        target_lane_index: int = None,
        target_speed: float = None,
        route: Route = None,
        enable_lane_change: bool = True,
        timer: float = None,
        data: dict = None,
    ):
        super().__init__(
            road,
            position,
            heading,
            speed,
            target_lane_index,
            target_speed,
            route,
            enable_lane_change,
            timer,
        )
        self.data = data if data is not None else {}
        self.collecting_data = True

    def act(self, action: dict | str = None):
        if self.collecting_data:
            self.collect_data()
        super().act(action)

    def randomize_behavior(self):
        ua = self.road.np_random.uniform(size=np.shape(self.ACCELERATION_PARAMETERS))
        self.ACCELERATION_PARAMETERS = self.ACCELERATION_RANGE[0] + ua * (
            self.ACCELERATION_RANGE[1] - self.ACCELERATION_RANGE[0]
        )
        ub = self.road.np_random.uniform(size=np.shape(self.STEERING_PARAMETERS))
        self.STEERING_PARAMETERS = self.STEERING_RANGE[0] + ub * (
            self.STEERING_RANGE[1] - self.STEERING_RANGE[0]
        )

    def acceleration(
        self,
        ego_vehicle: ControlledVehicle,
        front_vehicle: Vehicle = None,
        rear_vehicle: Vehicle = None,
    ) -> float:
        """
        Compute an acceleration command with a Linear Model.

        The acceleration is chosen so as to:
        - reach a target speed;
        - reach the speed of the leading (resp following) vehicle, if it is lower (resp higher) than ego's;
        - maintain a minimum safety distance w.r.t the leading vehicle.

        :param ego_vehicle: the vehicle whose desired acceleration is to be computed. It does not have to be an
                            Linear vehicle, which is why this method is a class method. This allows a Linear vehicle to
                            reason about other vehicles behaviors even though they may not Linear.
        :param front_vehicle: the vehicle preceding the ego-vehicle
        :param rear_vehicle: the vehicle following the ego-vehicle
        :return: the acceleration command for the ego-vehicle [m/s2]
        """
        return float(
            np.dot(
                self.ACCELERATION_PARAMETERS,
                self.acceleration_features(ego_vehicle, front_vehicle, rear_vehicle),
            )
        )

    def acceleration_features(
        self,
        ego_vehicle: ControlledVehicle,
        front_vehicle: Vehicle = None,
        rear_vehicle: Vehicle = None,
    ) -> np.ndarray:
        vt, dv, dp = 0, 0, 0
        if ego_vehicle:
            vt = (
                getattr(ego_vehicle, "target_speed", ego_vehicle.speed)
                - ego_vehicle.speed
            )
            d_safe = (
                self.DISTANCE_WANTED
                + np.maximum(ego_vehicle.speed, 0) * self.TIME_WANTED
            )
            if front_vehicle:
                d = ego_vehicle.lane_distance_to(front_vehicle)
                dv = min(front_vehicle.speed - ego_vehicle.speed, 0)
                dp = min(d - d_safe, 0)
        return np.array([vt, dv, dp])

    def steering_control(self, target_lane_index: LaneIndex) -> float:
        """
        Linear controller with respect to parameters.

        Overrides the non-linear controller ControlledVehicle.steering_control()

        :param target_lane_index: index of the lane to follow
        :return: a steering wheel angle command [rad]
        """
        return float(
            np.dot(
                np.array(self.STEERING_PARAMETERS),
                self.steering_features(target_lane_index),
            )
        )

    def steering_features(self, target_lane_index: LaneIndex) -> np.ndarray:
        """
        A collection of features used to follow a lane

        :param target_lane_index: index of the lane to follow
        :return: a array of features
        """
        lane = self.road.network.get_lane(target_lane_index)
        lane_coords = lane.local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + self.speed * self.TAU_PURSUIT
        lane_future_heading = lane.heading_at(lane_next_coords)
        features = np.array(
            [
                utils.wrap_to_pi(lane_future_heading - self.heading)
                * self.LENGTH
                / utils.not_zero(self.speed),
                -lane_coords[1] * self.LENGTH / (utils.not_zero(self.speed) ** 2),
            ]
        )
        return features

    def longitudinal_structure(self):
        # Nominal dynamics: integrate speed
        A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        # Target speed dynamics
        phi0 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
        # Front speed control
        phi1 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, -1, 1], [0, 0, 0, 0]])
        # Front position control
        phi2 = np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [-1, 1, -self.TIME_WANTED, 0], [0, 0, 0, 0]]
        )
        # Disable speed control
        front_vehicle, _ = self.road.neighbour_vehicles(self)
        if not front_vehicle or self.speed < front_vehicle.speed:
            phi1 *= 0

        # Disable front position control
        if front_vehicle:
            d = self.lane_distance_to(front_vehicle)
            if d != self.DISTANCE_WANTED + self.TIME_WANTED * self.speed:
                phi2 *= 0
        else:
            phi2 *= 0

        phi = np.array([phi0, phi1, phi2])
        return A, phi

    def lateral_structure(self):
        A = np.array([[0, 1], [0, 0]])
        phi0 = np.array([[0, 0], [0, -1]])
        phi1 = np.array([[0, 0], [-1, 0]])
        phi = np.array([phi0, phi1])
        return A, phi

    def collect_data(self):
        """Store features and outputs for parameter regression."""
        self.add_features(self.data, self.target_lane_index)

    def add_features(self, data, lane_index, output_lane=None):
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
        features = self.acceleration_features(self, front_vehicle, rear_vehicle)
        output = np.dot(self.ACCELERATION_PARAMETERS, features)
        if "longitudinal" not in data:
            data["longitudinal"] = {"features": [], "outputs": []}
        data["longitudinal"]["features"].append(features)
        data["longitudinal"]["outputs"].append(output)

        if output_lane is None:
            output_lane = lane_index
        features = self.steering_features(lane_index)
        out_features = self.steering_features(output_lane)
        output = np.dot(self.STEERING_PARAMETERS, out_features)
        if "lateral" not in data:
            data["lateral"] = {"features": [], "outputs": []}
        data["lateral"]["features"].append(features)
        data["lateral"]["outputs"].append(output)

class AggressiveVehicle(LinearVehicle):
    LANE_CHANGE_MIN_ACC_GAIN = 1.0  # [m/s2]
    MERGE_ACC_GAIN = 0.8
    MERGE_VEL_RATIO = 0.75
    MERGE_TARGET_VEL = 30
    ACCELERATION_PARAMETERS = [
        MERGE_ACC_GAIN / ((1 - MERGE_VEL_RATIO) * MERGE_TARGET_VEL),
        MERGE_ACC_GAIN / (MERGE_VEL_RATIO * MERGE_TARGET_VEL),
        0.5,
    ]

class DefensiveVehicle(LinearVehicle):
    LANE_CHANGE_MIN_ACC_GAIN = 1.0  # [m/s2]
    MERGE_ACC_GAIN = 1.2
    MERGE_VEL_RATIO = 0.75
    MERGE_TARGET_VEL = 30
    ACCELERATION_PARAMETERS = [
        MERGE_ACC_GAIN / ((1 - MERGE_VEL_RATIO) * MERGE_TARGET_VEL),
        MERGE_ACC_GAIN / (MERGE_VEL_RATIO * MERGE_TARGET_VEL),
        2.0,
    ]

class FlowIS_Follow(ControlledVehicle):
    """
    A vehicle using both a longitudinal and a lateral decision policies.

    - Longitudinal: the IDM model computes an acceleration given the preceding vehicle's distance and speed.
    - Lateral: the MOBIL model decides when to change lane by maximizing the acceleration of nearby vehicles.
    """

    # Longitudinal policy parameters
    ACC_MAX = 6.0  # [m/s2]
    """Maximum acceleration."""
    COMFORT_ACC_MAX = 3.0  # [m/s2]
    """Desired maximum acceleration."""
    COMFORT_ACC_MIN = -6.0  # [m/s2]
    """Desired maximum deceleration."""
    DISTANCE_WANTED = 10.0 + ControlledVehicle.LENGTH  # [m]
    """Desired jam distance to the front vehicle."""
    TIME_WANTED = 2  # [s]1.5
    """Desired time gap to the front vehicle."""
    DELTA = 4.0  # []
    """Exponent of the velocity term."""
    DELTA_RANGE = [3.5, 4.5]
    """Range of delta when chosen randomly."""
    # Lateral policy parameters
    POLITENESS = 0.0  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]


    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: int = None,
                 target_speed: float = None,
                 route: Route = None,
                 enable_lane_change: bool = True,
                 timer: float = None,
                 line: AbstractLane = None,
                 NDE_distribution: dict = None
                 ):
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, route)
        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.position)*np.pi) % self.LANE_CHANGE_DELAY
        self.line = line
        self.NDE_distribution = NDE_distribution
        self.proposal_distribution = NDE_distribution
        self.target_distribution = None
        self.logpro = None
        self.behavior = 0.0
        self.behavior_steering = 0.0
        self.log_weight_step = 0.0
        self.log_weight_scene = 0.0
        self.log_weight_episode = 0.0
        self.last_flowis_trace = {}
        # Sparse-injection controls (variance reduction)
        self.flowis_event_mode = "always"  # "always" | "critical"
        self.flowis_ttc_threshold = 4.0
        self.flowis_distance_threshold = 40.0
        self.flowis_delta_v_threshold = 0.5
        self.flowis_decision_interval = 10
        self.flowis_max_weighted_steps = 10
        self.flowis_burst_steps = 1
        self.flowis_weighted_steps = 0
        self.flowis_importance_weighting = True
        self.flowis_use_exact_conditional = True
        self.flowis_clip_actions = False
        self._flowis_step_count = 0
        self._flowis_last_sample_step = -1
        self._flowis_burst_remaining = 0
        self.flowis_scene_context = None
        # Optional cut-in controls (used by HighD cut-in attack mode).
        self.flowis_target_vehicle = None
        self.flowis_cutin_enabled = False
        self.flowis_cutin_trigger_distance = 25.0
        self.flowis_cutin_commit_window = (-10.0, 25.0)
        self.flowis_cutin_committed = False
    @classmethod
    def create_from(cls, vehicle: ControlledVehicle) -> "IDMVehicle":
        """
        Create a new vehicle from an existing one.

        The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.position, heading=vehicle.heading, speed=vehicle.speed,
                target_lane_index=vehicle.target_lane_index, target_speed=vehicle.target_speed,
                route=vehicle.route, timer=getattr(vehicle, 'timer', None))
        return v

    def desired_gap(
        self,
        ego_vehicle: Vehicle,
        front_vehicle: Vehicle = None,
        projected: bool = True,
    ) -> float:
        """
        Compute the desired distance between a vehicle and its leading vehicle.

        :param ego_vehicle: the vehicle being controlled
        :param front_vehicle: its leading vehicle
        :param projected: project 2D velocities in 1D space
        :return: the desired distance between the two [m]
        """
        d0 = self.DISTANCE_WANTED
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = (
            np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction)
            if projected
            else ego_vehicle.speed - front_vehicle.speed
        )
        d_star = (
            d0 + ego_vehicle.speed * tau + ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        )
        return d_star

    def acceleration(
        self,
        ego_vehicle: ControlledVehicle,
        front_vehicle: Vehicle = None,
        rear_vehicle: Vehicle = None,
    ) -> float:
        """
        Compute an acceleration command with the Intelligent Driver Model.

        The acceleration is chosen so as to:
        - reach a target speed;
        - maintain a minimum safety distance (and safety time) w.r.t the front vehicle.

        :param ego_vehicle: the vehicle whose desired acceleration is to be computed. It does not have to be an
                            IDM vehicle, which is why this method is a class method. This allows an IDM vehicle to
                            reason about other vehicles behaviors even though they may not IDMs.
        :param front_vehicle: the vehicle preceding the ego-vehicle
        :param rear_vehicle: the vehicle following the ego-vehicle
        :return: the acceleration command for the ego-vehicle [m/s2]
        """
        if not ego_vehicle or not isinstance(ego_vehicle, Vehicle):
            return 0
        ego_target_speed = getattr(ego_vehicle, "target_speed", 0)
        if ego_vehicle.lane and ego_vehicle.lane.speed_limit is not None:
            ego_target_speed = np.clip(
                ego_target_speed, 0, ego_vehicle.lane.speed_limit
            )
        acceleration = self.COMFORT_ACC_MAX * (
            1
            - np.power(
                max(ego_vehicle.speed, 0) / abs(utils.not_zero(ego_target_speed)),
                self.DELTA,
            )
        )

        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle)
            acceleration -= self.COMFORT_ACC_MAX * np.power(
                self.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2
            )
        return acceleration

    def _scene_raw(self) -> np.ndarray:
        target = self.flowis_target_vehicle
        if target is None:
            target = self.road.vehicles[0]
        return np.array([target.speed, self.speed, self.position[0] - target.position[0]], dtype=float)

    def _scene_metrics(self, scene_raw: np.ndarray) -> tuple[float, float]:
        # closing speed > 0 means rear car is approaching front car.
        delta_v = float(scene_raw[0] - scene_raw[1])
        distance = float(scene_raw[2] - 5.0)
        if delta_v <= 1e-6:
            ttc = float("inf")
        else:
            ttc = distance / delta_v
        return ttc, delta_v

    def _critical_event(self, scene_raw: np.ndarray) -> tuple[bool, float, float]:
        ttc, delta_v = self._scene_metrics(scene_raw)
        distance = float(scene_raw[2])
        is_critical = (
            (ttc > 0.0 and ttc < float(self.flowis_ttc_threshold))
            or (distance < float(self.flowis_distance_threshold))
            or (delta_v > float(self.flowis_delta_v_threshold))
        )
        return is_critical, ttc, delta_v

    def _idm_behavior(self) -> float:
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self, self.lane_index)
        accelerat = self.acceleration(ego_vehicle=self, front_vehicle=front_vehicle, rear_vehicle=rear_vehicle)
        return float(np.clip(accelerat, -5.0, self.ACC_MAX))

    def _scene_dims(self, distribution: dict) -> list[int]:
        dims = distribution.get("scene_dims", None) if isinstance(distribution, dict) else None
        if dims is None:
            return [0, 1, 2]
        return [int(d) for d in dims]

    def _behavior_dims(self, distribution: dict, scene_dims: list[int]) -> list[int]:
        if not isinstance(distribution, dict):
            return [3]
        explicit = distribution.get("behavior_dims", None)
        if explicit is not None:
            return [int(d) for d in explicit]
        total_dim = distribution.get("scene_behavior_dim", None)
        if total_dim is None:
            scaler = distribution.get("fit_scene_behavior", None)
            if scaler is not None and hasattr(scaler, "n_features_in_"):
                total_dim = int(getattr(scaler, "n_features_in_"))
        if total_dim is None:
            model = distribution.get("scene_behavior", None)
            total_dim = int(getattr(model, "n_features_in_", len(scene_dims) + 1))
        return [i for i in range(int(total_dim)) if i not in scene_dims]

    def _build_scene_raw(self, distribution: dict) -> np.ndarray:
        target = self.flowis_target_vehicle
        if target is None:
            target = self.road.vehicles[0]
        base_scene = np.array([target.speed, self.speed, self.position[0] - target.position[0]], dtype=float)

        scene_dims = self._scene_dims(distribution)
        dim_count = len(scene_dims)
        if dim_count <= 3:
            return base_scene[:dim_count]

        if self.flowis_scene_context is not None:
            context = np.asarray(self.flowis_scene_context, dtype=float).reshape(-1)
        else:
            context = np.zeros(dim_count, dtype=float)
        if context.shape[0] < dim_count:
            pad = np.zeros(dim_count - context.shape[0], dtype=float)
            context = np.concatenate([context, pad], axis=0)

        # Keep cut-in scene semantics from distribution (v3,v2,v2y,v1,sx2,sx1,sy,...)
        # instead of overwriting first 3 dims with follow-style features.
        if self.flowis_cutin_enabled:
            return context[:dim_count].copy()

        scene_raw = context[:dim_count].copy()
        scene_raw[:3] = base_scene[:3]
        return scene_raw

    def _scene_support_check(self, distribution: dict, scene_raw: np.ndarray) -> tuple[bool, str]:
        """Check whether the current scene is suitable for conditional sampling."""
        scene_raw = np.asarray(scene_raw, dtype=float).reshape(-1)

        # Cut-in mode uses dedicated scene semantics and can naturally violate
        # follow-style neighborhood assumptions; do not pre-block injection here.
        if self.flowis_cutin_enabled:
            return True, "cutin_bypass"

        # If model uses richer scene dims (e.g., cut-in with rear context), require a rear vehicle.
        if scene_raw.shape[0] > 3:
            try:
                _, rear = self.road.neighbour_vehicles(self, self.lane_index)
                if rear is None:
                    return False, "no_rear_vehicle"
            except Exception:
                # If neighbor query is unavailable, do not hard-fail here.
                pass

        scaler = distribution.get("fit_scene", None) if isinstance(distribution, dict) else None
        if scaler is not None and hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):
            dmin = np.asarray(scaler.data_min_, dtype=float).reshape(-1)
            dmax = np.asarray(scaler.data_max_, dtype=float).reshape(-1)
            d = min(scene_raw.shape[0], dmin.shape[0], dmax.shape[0])
            if d > 0:
                s = scene_raw[:d]
                lo = dmin[:d]
                hi = dmax[:d]
                span = np.maximum(hi - lo, 1e-6)
                # Small slack to absorb floating jitter but still reject OOD states.
                lower = lo - 0.05 * span
                upper = hi + 0.05 * span
                if np.any(s < lower) or np.any(s > upper):
                    return False, "scene_out_of_support"

        return True, "ok"

    def _flowis_sample_step(
        self,
        distribution: dict,
        weighted: bool,
        sampler: str,
        fallback_sampler: str,
    ):
        target = self.target_distribution
        scene_dims = self._scene_dims(distribution)
        behavior_dims = self._behavior_dims(distribution, scene_dims)
        scene_raw = self._build_scene_raw(distribution)
        scene_norm = distribution['fit_scene'].transform([scene_raw])

        use_2d_sampler = (
            len(behavior_dims) >= 2
            and isinstance(distribution, dict)
            and ("proposal_steering_angle" in distribution)
        )
        use_exact_sampler = bool(getattr(self, "flowis_use_exact_conditional", True))
        if use_exact_sampler:
            scene_behavior_norm, log_q_cond, trials = RejectAcceptSampling.conditional_sample_gmm(
                scene_norm,
                distribution,
                scene_dims,
                behavior_dims,
                random_state=None,
            )
        elif use_2d_sampler:
            scene_behavior_norm, log_q_cond, trials = RejectAcceptSampling.rejection_sampling_2D(
                scene_norm,
                distribution,
                dim=scene_dims,
                max_trials=1000,
                random_state=None,
            )
        else:
            scene_behavior_norm, log_q_cond, trials = RejectAcceptSampling.rejection_sampling_1D(
                scene_norm,
                distribution,
                dim=scene_dims,
                max_trials=1000,
                random_state=None,
            )

        if scene_behavior_norm is None:
            self.behavior = self._idm_behavior()
            self.behavior_steering = 0.0
            self.logpro = None
            self.log_weight_step = 0.0
            self.last_flowis_trace = {
                'accepted': False,
                'trials': int(trials or 0),
                'scene_raw': scene_raw.tolist(),
                'behavior': self.behavior,
                'behavior_steering': self.behavior_steering,
                'weighted_step': bool(weighted),
                'log_q_cond': None,
                'log_p_cond': None,
                'log_weight_step': 0.0,
                'sampler': fallback_sampler,
            }
            return self.logpro, trials

        scene_behavior_norm = np.asarray(scene_behavior_norm).reshape(1, -1)
        scene_behavior_raw = distribution['fit_scene_behavior'].inverse_transform(scene_behavior_norm)
        acc_idx = int(behavior_dims[0]) if behavior_dims else int(scene_behavior_raw.shape[1] - 1)
        accelerat = float(scene_behavior_raw[0][acc_idx])
        if bool(getattr(self, "flowis_clip_actions", False)):
            self.behavior = float(np.clip(accelerat, -5.0, self.ACC_MAX))
        else:
            self.behavior = float(accelerat)
        if use_2d_sampler and len(behavior_dims) >= 2:
            steer_idx = int(behavior_dims[1])
            steer_raw = float(scene_behavior_raw[0][steer_idx])
            if bool(getattr(self, "flowis_clip_actions", False)):
                self.behavior_steering = float(np.clip(steer_raw, -0.35, 0.35))
            else:
                self.behavior_steering = float(steer_raw)
        else:
            self.behavior_steering = 0.0
        self.logpro = float(np.asarray(log_q_cond).reshape(-1)[0]) if log_q_cond is not None else None

        if weighted:
            log_p_cond = _safe_log_conditional(target, scene_behavior_norm, scene_norm, scene_dims) if target else None
            if log_p_cond is not None and self.logpro is not None:
                self.log_weight_step = float(log_p_cond - self.logpro)
            else:
                self.log_weight_step = 0.0
        else:
            log_p_cond = None
            self.log_weight_step = 0.0

        self.last_flowis_trace = {
            'accepted': True,
            'trials': int(trials or 0),
            'scene_raw': scene_raw.tolist(),
            'scene_norm': scene_norm.reshape(-1).tolist(),
            'scene_behavior_norm': scene_behavior_norm.reshape(-1).tolist(),
            'scene_behavior_raw': scene_behavior_raw.reshape(-1).tolist(),
            'behavior': self.behavior,
            'behavior_steering': self.behavior_steering,
            'scene_dims': scene_dims,
            'behavior_dims': behavior_dims,
            'weighted_step': bool(weighted),
            'log_q_cond': self.logpro,
            'log_p_cond': log_p_cond,
            'log_weight_step': self.log_weight_step,
            'sampler': 'conditional_gmm_exact' if use_exact_sampler else ('rejection_sampling_2D' if use_2d_sampler else sampler),
        }
        return self.logpro, trials

    def flowis(self):
        # Weighted step: sample behavior from TDE (proposal) and compute log p/q with NDE target.
        proposal = self.proposal_distribution or self.NDE_distribution
        return self._flowis_sample_step(
            proposal,
            weighted=True,
            sampler='rejection_sampling_1D_tde_weighted',
            fallback_sampler='fallback_idm_tde_weighted',
        )

    def flowis_unweighted(self):
        # Non-weighted step: sample behavior from NDE and force unit weight.
        nde_dist = self.target_distribution or self.NDE_distribution or self.proposal_distribution
        return self._flowis_sample_step(
            nde_dist,
            weighted=False,
            sampler='rejection_sampling_1D_nde_unweighted',
            fallback_sampler='fallback_idm_nde_unweighted',
        )

    def act(self, action: Union[dict, str] = None):
        """
        Execute an action.

        For now, no action is supported because the vehicle takes all decisions
        of acceleration and lane changes on its own, based on the IDM and MOBIL models.

        :param action: the action
        """
        if self.crashed:
            return
        self.follow_road()
        self._flowis_step_count += 1
        distribution_for_scene = self.proposal_distribution or self.NDE_distribution or self.target_distribution
        scene_raw = self._build_scene_raw(distribution_for_scene)

        if scene_raw.shape[0] >= 3:
            critical, ttc, delta_v = self._critical_event(scene_raw)
        else:
            # If scene has <3 dims, fallback to always critical for safety.
            critical, ttc, delta_v = True, float("inf"), 0.0

        use_critical_mode = str(self.flowis_event_mode).lower() == "critical"
        interval = max(1, int(self.flowis_decision_interval))
        burst_active = self._flowis_burst_remaining > 0
        should_resample = (
            burst_active
            or
            self._flowis_last_sample_step < 0
            or (self._flowis_step_count - self._flowis_last_sample_step) >= interval
        )

        proposal = self.proposal_distribution or self.NDE_distribution
        target = self.target_distribution
        weighting_enabled = bool(self.flowis_importance_weighting) and (proposal is not None) and (target is not None)
        # Monte Carlo NDE runs should stay unweighted (q == p).
        if proposal is target:
            weighting_enabled = False

        budget_limit = max(0, int(self.flowis_max_weighted_steps))
        budget_ok = self.flowis_weighted_steps < budget_limit
        weighted_step = should_resample and weighting_enabled and budget_ok and ((not use_critical_mode) or critical)
        # During an active burst window, force weighted resampling each step.
        if burst_active and should_resample and weighting_enabled and budget_ok:
            weighted_step = True
        # If burst cannot continue (budget or weighting disabled), stop burst.
        if burst_active and not (weighting_enabled and budget_ok):
            self._flowis_burst_remaining = 0
            burst_active = False
        support_ok, support_reason = self._scene_support_check(distribution_for_scene, scene_raw)

        if should_resample and (not support_ok):
            self.behavior = self._idm_behavior()
            self.behavior_steering = 0.0
            self.logpro = None
            self.log_weight_step = 0.0
            self.last_flowis_trace = {
                'accepted': False,
                'scene_raw': scene_raw.tolist(),
                'behavior': self.behavior,
                'behavior_steering': self.behavior_steering,
                'ttc': float(ttc),
                'delta_v': float(delta_v),
                'critical': bool(critical),
                'event_mode': self.flowis_event_mode,
                'decision_interval': interval,
                'weighted_step': False,
                'weighted_steps': int(self.flowis_weighted_steps),
                'max_weighted_steps': budget_limit,
                'log_q_cond': None,
                'log_p_cond': None,
                'log_weight_step': 0.0,
                'sampler': 'skip_unsupported_scene_idm',
                'support_reason': support_reason,
            }
        elif use_critical_mode and (not critical):
            self.behavior = self._idm_behavior()
            self.behavior_steering = 0.0
            self.logpro = None
            self.log_weight_step = 0.0
            self.last_flowis_trace = {
                'accepted': False,
                'scene_raw': scene_raw.tolist(),
                'behavior': self.behavior,
                'behavior_steering': self.behavior_steering,
                'ttc': float(ttc),
                'delta_v': float(delta_v),
                'critical': False,
                'event_mode': self.flowis_event_mode,
                'decision_interval': interval,
                'weighted_step': False,
                'weighted_steps': int(self.flowis_weighted_steps),
                'max_weighted_steps': budget_limit,
                'log_q_cond': None,
                'log_p_cond': None,
                'log_weight_step': 0.0,
                'sampler': 'skip_noncritical_idm',
            }
        elif should_resample:
            prev_burst_active = burst_active
            burst_step_consumed = False
            if weighted_step:
                self.logpro, _ = self.flowis()
                # If TDE conditional sampling fails under OOD scene, fallback to NDE.
                # This keeps simulation running and sets step-weight to 1 for this step.
                if self.logpro is None:
                    self.logpro, _ = self.flowis_unweighted()
                    weighted_step = False
                else:
                    self.flowis_weighted_steps += 1
                    burst_step_consumed = True
                    # Start a burst after the first successful TDE injection.
                    if (not prev_burst_active) and int(self.flowis_burst_steps) > 1:
                        self._flowis_burst_remaining = int(self.flowis_burst_steps) - 1
            else:
                self.logpro, _ = self.flowis_unweighted()
            # Consume one burst slot per step while burst is active.
            if prev_burst_active and self._flowis_burst_remaining > 0:
                self._flowis_burst_remaining = max(0, self._flowis_burst_remaining - 1)
            self._flowis_last_sample_step = self._flowis_step_count
            self.last_flowis_trace['critical'] = bool(critical)
            self.last_flowis_trace['ttc'] = float(ttc)
            self.last_flowis_trace['delta_v'] = float(delta_v)
            self.last_flowis_trace['event_mode'] = self.flowis_event_mode
            self.last_flowis_trace['decision_interval'] = interval
            self.last_flowis_trace['burst_steps'] = int(self.flowis_burst_steps)
            self.last_flowis_trace['burst_remaining'] = int(self._flowis_burst_remaining)
            self.last_flowis_trace['burst_step_consumed'] = bool(burst_step_consumed)
            self.last_flowis_trace['weighted_step'] = bool(weighted_step)
            self.last_flowis_trace['weighted_steps'] = int(self.flowis_weighted_steps)
            self.last_flowis_trace['max_weighted_steps'] = budget_limit
        else:
            # Hold previously sampled behavior between decision points.
            self.log_weight_step = 0.0
            self.last_flowis_trace = {
                'accepted': True,
                'scene_raw': scene_raw.tolist(),
                'behavior': self.behavior,
                'behavior_steering': self.behavior_steering,
                'ttc': float(ttc),
                'delta_v': float(delta_v),
                'critical': bool(critical),
                'event_mode': self.flowis_event_mode,
                'decision_interval': interval,
                'steps_since_sample': int(self._flowis_step_count - self._flowis_last_sample_step),
                'weighted_step': False,
                'weighted_steps': int(self.flowis_weighted_steps),
                'max_weighted_steps': budget_limit,
                'log_q_cond': None,
                'log_p_cond': None,
                'log_weight_step': 0.0,
                'sampler': 'hold_previous_behavior',
            }

        self.log_weight_episode += self.log_weight_step
        if self.flowis_cutin_enabled and (not self.flowis_cutin_committed):
            target = self.flowis_target_vehicle
            if target is not None and hasattr(target, "lane_index"):
                target_lane = target.lane_index
                dist_x = float(target.position[0] - self.position[0])
                lo, hi = self.flowis_cutin_commit_window
                trigger = abs(dist_x) <= float(self.flowis_cutin_trigger_distance) or (lo <= dist_x <= hi)
                if trigger and target_lane[:2] == self.lane_index[:2]:
                    self.target_lane_index = (target_lane[0], target_lane[1], int(target_lane[2]))
                    self.flowis_cutin_committed = True
                    self.last_flowis_trace["cutin_committed"] = True
                    self.last_flowis_trace["cutin_distance_x"] = dist_x
        action = {}
        base_steering = float(self.steering_control(self.target_lane_index))
        steering = float(np.clip(base_steering + self.behavior_steering, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE))
        action['acceleration'],action['steering']=self.behavior,steering

        Vehicle.act(self, action) 

    def step(self, dt: float):
        """
        Step the simulation.

        Increases a timer used for decision policies, and step the vehicle dynamics.

        :param dt: timestep
        """
        self.timer += dt
        super().step(dt)

class FlowIS_Gostraight(ControlledVehicle):
    """
    A vehicle using both a longitudinal and a lateral decision policies.

    - Longitudinal: the IDM model computes an acceleration given the preceding vehicle's distance and speed.
    - Lateral: the MOBIL model decides when to change lane by maximizing the acceleration of nearby vehicles.
    """

    # Longitudinal policy parameters
    ACC_MAX = 6.0  # [m/s2]
    """Maximum acceleration."""
    COMFORT_ACC_MAX = 3.0  # [m/s2]
    """Desired maximum acceleration."""
    COMFORT_ACC_MIN = -6.0  # [m/s2]
    """Desired maximum deceleration."""
    DISTANCE_WANTED = 10.0 + ControlledVehicle.LENGTH  # [m]
    """Desired jam distance to the front vehicle."""
    TIME_WANTED = 2  # [s]1.5
    """Desired time gap to the front vehicle."""
    DELTA = 4.0  # []
    """Exponent of the velocity term."""
    DELTA_RANGE = [3.5, 4.5]
    """Range of delta when chosen randomly."""
    # Lateral policy parameters
    POLITENESS = 0.0  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]


    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: int = None,
                 target_speed: float = None,
                 route: Route = None,
                 enable_lane_change: bool = True,
                 timer: float = None,
                 line: AbstractLane = None,
                 NDE_intersection: dict = None
                 ):
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, route)
        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.position)*np.pi) % self.LANE_CHANGE_DELAY
        self.line = line
        self.NDE_intersection=NDE_intersection
        self.logpro=None
        self.behavior=None
    @classmethod
    def create_from(cls, vehicle: ControlledVehicle) -> "IDMVehicle":
        """
        Create a new vehicle from an existing one.

        The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.position, heading=vehicle.heading, speed=vehicle.speed,
                target_lane_index=vehicle.target_lane_index, target_speed=vehicle.target_speed,
                route=vehicle.route, timer=getattr(vehicle, 'timer', None))
        return v

    def desired_gap(
        self,
        ego_vehicle: Vehicle,
        front_vehicle: Vehicle = None,
        projected: bool = True,
    ) -> float:
        """
        Compute the desired distance between a vehicle and its leading vehicle.

        :param ego_vehicle: the vehicle being controlled
        :param front_vehicle: its leading vehicle
        :param projected: project 2D velocities in 1D space
        :return: the desired distance between the two [m]
        """
        d0 = self.DISTANCE_WANTED
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = (
            np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction)
            if projected
            else ego_vehicle.speed - front_vehicle.speed
        )
        d_star = (
            d0 + ego_vehicle.speed * tau + ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        )
        return d_star

    def acceleration(
        self,
        ego_vehicle: ControlledVehicle,
        front_vehicle: Vehicle = None,
        rear_vehicle: Vehicle = None,
    ) -> float:
        """
        Compute an acceleration command with the Intelligent Driver Model.

        The acceleration is chosen so as to:
        - reach a target speed;
        - maintain a minimum safety distance (and safety time) w.r.t the front vehicle.

        :param ego_vehicle: the vehicle whose desired acceleration is to be computed. It does not have to be an
                            IDM vehicle, which is why this method is a class method. This allows an IDM vehicle to
                            reason about other vehicles behaviors even though they may not IDMs.
        :param front_vehicle: the vehicle preceding the ego-vehicle
        :param rear_vehicle: the vehicle following the ego-vehicle
        :return: the acceleration command for the ego-vehicle [m/s2]
        """
        if not ego_vehicle or not isinstance(ego_vehicle, Vehicle):
            return 0
        ego_target_speed = getattr(ego_vehicle, "target_speed", 0)
        if ego_vehicle.lane and ego_vehicle.lane.speed_limit is not None:
            ego_target_speed = np.clip(
                ego_target_speed, 0, ego_vehicle.lane.speed_limit
            )
        acceleration = self.COMFORT_ACC_MAX * (
            1
            - np.power(
                max(ego_vehicle.speed, 0) / abs(utils.not_zero(ego_target_speed)),
                self.DELTA,
            )
        )

        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle)
            acceleration -= self.COMFORT_ACC_MAX * np.power(
                self.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2
            )
        return acceleration

    def flowis(self):
        proposal = self.proposal_distribution or self.NDE_distribution
        target = self.target_distribution
        av = self.road.vehicles[0]
        scene_raw = np.array([av.speed, self.speed, self.position[0] - av.position[0]], dtype=float)
        scene_norm = proposal['fit_scene'].transform([scene_raw])

        scene_behavior_norm, log_q_cond, trials = RejectAcceptSampling.rejection_sampling_1D(
            scene_norm,
            proposal,
            dim=[0, 1, 2],
            max_trials=1000,
            random_state=None,
        )

        if scene_behavior_norm is None:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self, self.lane_index)
            accelerat = self.acceleration(ego_vehicle=self, front_vehicle=front_vehicle, rear_vehicle=rear_vehicle)
            self.behavior = float(np.clip(accelerat, -5.0, self.ACC_MAX))
            self.logpro = None
            self.log_weight_step = 0.0
            self.last_flowis_trace = {
                'accepted': False,
                'trials': int(trials or 0),
                'scene_raw': scene_raw.tolist(),
                'behavior': self.behavior,
                'log_q_cond': None,
                'log_p_cond': None,
                'log_weight_step': self.log_weight_step,
                'sampler': 'fallback_idm',
            }
            return self.logpro, trials

        scene_behavior_norm = np.asarray(scene_behavior_norm).reshape(1, -1)
        scene_behavior_raw = proposal['fit_scene_behavior'].inverse_transform(scene_behavior_norm)
        accelerat = float(scene_behavior_raw[0][3])
        self.behavior = float(np.clip(accelerat, -5.0, self.ACC_MAX))
        self.logpro = float(np.asarray(log_q_cond).reshape(-1)[0]) if log_q_cond is not None else None

        log_p_cond = _safe_log_conditional(target, scene_behavior_norm, scene_norm, [0, 1, 2]) if target else None
        if log_p_cond is not None and self.logpro is not None:
            self.log_weight_step = float(log_p_cond - self.logpro)
        else:
            self.log_weight_step = 0.0

        self.last_flowis_trace = {
            'accepted': True,
            'trials': int(trials or 0),
            'scene_raw': scene_raw.tolist(),
            'scene_norm': scene_norm.reshape(-1).tolist(),
            'scene_behavior_norm': scene_behavior_norm.reshape(-1).tolist(),
            'scene_behavior_raw': scene_behavior_raw.reshape(-1).tolist(),
            'behavior': self.behavior,
            'log_q_cond': self.logpro,
            'log_p_cond': log_p_cond,
            'log_weight_step': self.log_weight_step,
            'sampler': 'rejection_sampling_1D',
        }
        return self.logpro, trials

    def act(self, action: Union[dict, str] = None):
        """
        Execute an action.

        For now, no action is supported because the vehicle takes all decisions
        of acceleration and lane changes on its own, based on the IDM and MOBIL models.

        :param action: the action
        """
        if self.crashed:
            return
        self.follow_road()
        self._flowis_step_count += 1
        scene_raw = self._scene_raw()
        critical, ttc, delta_v = self._critical_event(scene_raw)
        use_critical_mode = str(self.flowis_event_mode).lower() == "critical"

        if use_critical_mode and not critical:
            # Non-critical region: no IS update, use nominal IDM.
            self.behavior = self._idm_behavior()
            self.logpro = None
            self.log_weight_step = 0.0
            self.last_flowis_trace = {
                'accepted': False,
                'scene_raw': scene_raw.tolist(),
                'behavior': self.behavior,
                'ttc': float(ttc),
                'delta_v': float(delta_v),
                'critical': False,
                'event_mode': self.flowis_event_mode,
                'log_q_cond': None,
                'log_p_cond': None,
                'log_weight_step': 0.0,
                'sampler': 'skip_noncritical_idm',
            }
        else:
            # Critical region (or always mode): discrete decision interval.
            interval = max(1, int(self.flowis_decision_interval))
            should_resample = (
                self._flowis_last_sample_step < 0
                or (self._flowis_step_count - self._flowis_last_sample_step) >= interval
            )
            if should_resample:
                self.logpro, _ = self.flowis()
                self.log_weight_episode += self.log_weight_step
                self._flowis_last_sample_step = self._flowis_step_count
                self.last_flowis_trace['ttc'] = float(ttc)
                self.last_flowis_trace['delta_v'] = float(delta_v)
                self.last_flowis_trace['critical'] = bool(critical)
                self.last_flowis_trace['event_mode'] = self.flowis_event_mode
                self.last_flowis_trace['decision_interval'] = interval
            else:
                # Hold previously sampled behavior between decision points.
                self.log_weight_step = 0.0
                self.last_flowis_trace = {
                    'accepted': True,
                    'scene_raw': scene_raw.tolist(),
                    'behavior': self.behavior,
                    'ttc': float(ttc),
                    'delta_v': float(delta_v),
                    'critical': bool(critical),
                    'event_mode': self.flowis_event_mode,
                    'decision_interval': interval,
                    'steps_since_sample': int(self._flowis_step_count - self._flowis_last_sample_step),
                    'log_q_cond': None,
                    'log_p_cond': None,
                    'log_weight_step': 0.0,
                    'sampler': 'hold_previous_behavior',
                }
        action = {}
        action['acceleration'],action['steering']=self.behavior,self.steering_control(self.target_lane_index)

        Vehicle.act(self, action) 

    def step(self, dt: float):
        """
        Step the simulation.

        Increases a timer used for decision policies, and step the vehicle dynamics.

        :param dt: timestep
        """
        self.timer += dt
        super().step(dt)

class FlowIS_TurnLeft(ControlledVehicle):
    """
    A vehicle using both a longitudinal and a lateral decision policies.

    - Longitudinal: the IDM model computes an acceleration given the preceding vehicle's distance and speed.
    - Lateral: the MOBIL model decides when to change lane by maximizing the acceleration of nearby vehicles.
    """

    # Longitudinal policy parameters
    ACC_MAX = 6.0  # [m/s2]
    """Maximum acceleration."""
    COMFORT_ACC_MAX = 3.0  # [m/s2]
    """Desired maximum acceleration."""
    COMFORT_ACC_MIN = -6.0  # [m/s2]
    """Desired maximum deceleration."""
    DISTANCE_WANTED = 10.0 + ControlledVehicle.LENGTH  # [m]
    """Desired jam distance to the front vehicle."""
    TIME_WANTED = 2  # [s]1.5
    """Desired time gap to the front vehicle."""
    DELTA = 4.0  # []
    """Exponent of the velocity term."""
    DELTA_RANGE = [3.5, 4.5]
    """Range of delta when chosen randomly."""
    # Lateral policy parameters
    POLITENESS = 0.0  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]


    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: int = None,
                 target_speed: float = None,
                 route: Route = None,
                 enable_lane_change: bool = True,
                 timer: float = None,
                 line: AbstractLane = None,
                 NDE_intersection: dict = None
                 ):
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, route)
        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.position)*np.pi) % self.LANE_CHANGE_DELAY
        self.line = line
        self.NDE_intersection=NDE_intersection
        self.logpro=None
        self.behavior=None
    @classmethod
    def create_from(cls, vehicle: ControlledVehicle) -> "IDMVehicle":
        """
        Create a new vehicle from an existing one.

        The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.position, heading=vehicle.heading, speed=vehicle.speed,
                target_lane_index=vehicle.target_lane_index, target_speed=vehicle.target_speed,
                route=vehicle.route, timer=getattr(vehicle, 'timer', None))
        return v

    def desired_gap(
        self,
        ego_vehicle: Vehicle,
        front_vehicle: Vehicle = None,
        projected: bool = True,
    ) -> float:
        """
        Compute the desired distance between a vehicle and its leading vehicle.

        :param ego_vehicle: the vehicle being controlled
        :param front_vehicle: its leading vehicle
        :param projected: project 2D velocities in 1D space
        :return: the desired distance between the two [m]
        """
        d0 = self.DISTANCE_WANTED
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = (
            np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction)
            if projected
            else ego_vehicle.speed - front_vehicle.speed
        )
        d_star = (
            d0 + ego_vehicle.speed * tau + ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        )
        return d_star

    def acceleration(
        self,
        ego_vehicle: ControlledVehicle,
        front_vehicle: Vehicle = None,
        rear_vehicle: Vehicle = None,
    ) -> float:
        """
        Compute an acceleration command with the Intelligent Driver Model.

        The acceleration is chosen so as to:
        - reach a target speed;
        - maintain a minimum safety distance (and safety time) w.r.t the front vehicle.

        :param ego_vehicle: the vehicle whose desired acceleration is to be computed. It does not have to be an
                            IDM vehicle, which is why this method is a class method. This allows an IDM vehicle to
                            reason about other vehicles behaviors even though they may not IDMs.
        :param front_vehicle: the vehicle preceding the ego-vehicle
        :param rear_vehicle: the vehicle following the ego-vehicle
        :return: the acceleration command for the ego-vehicle [m/s2]
        """
        if not ego_vehicle or not isinstance(ego_vehicle, Vehicle):
            return 0
        ego_target_speed = getattr(ego_vehicle, "target_speed", 0)
        if ego_vehicle.lane and ego_vehicle.lane.speed_limit is not None:
            ego_target_speed = np.clip(
                ego_target_speed, 0, ego_vehicle.lane.speed_limit
            )
        acceleration = self.COMFORT_ACC_MAX * (
            1
            - np.power(
                max(ego_vehicle.speed, 0) / abs(utils.not_zero(ego_target_speed)),
                self.DELTA,
            )
        )

        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle)
            acceleration -= self.COMFORT_ACC_MAX * np.power(
                self.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2
            )
        return acceleration

    def flowis(self):
        proposal = self.proposal_distribution or self.NDE_distribution
        target = self.target_distribution
        av = self.road.vehicles[0]
        scene_raw = np.array([av.speed, self.speed, self.position[0] - av.position[0]], dtype=float)
        scene_norm = proposal['fit_scene'].transform([scene_raw])

        scene_behavior_norm, log_q_cond, trials = RejectAcceptSampling.rejection_sampling_1D(
            scene_norm,
            proposal,
            dim=[0, 1, 2],
            max_trials=1000,
            random_state=None,
        )

        if scene_behavior_norm is None:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self, self.lane_index)
            accelerat = self.acceleration(ego_vehicle=self, front_vehicle=front_vehicle, rear_vehicle=rear_vehicle)
            self.behavior = float(np.clip(accelerat, -5.0, self.ACC_MAX))
            self.logpro = None
            self.log_weight_step = 0.0
            self.last_flowis_trace = {
                'accepted': False,
                'trials': int(trials or 0),
                'scene_raw': scene_raw.tolist(),
                'behavior': self.behavior,
                'log_q_cond': None,
                'log_p_cond': None,
                'log_weight_step': self.log_weight_step,
                'sampler': 'fallback_idm',
            }
            return self.logpro, trials

        scene_behavior_norm = np.asarray(scene_behavior_norm).reshape(1, -1)
        scene_behavior_raw = proposal['fit_scene_behavior'].inverse_transform(scene_behavior_norm)
        accelerat = float(scene_behavior_raw[0][3])
        self.behavior = float(np.clip(accelerat, -5.0, self.ACC_MAX))
        self.logpro = float(np.asarray(log_q_cond).reshape(-1)[0]) if log_q_cond is not None else None

        log_p_cond = _safe_log_conditional(target, scene_behavior_norm, scene_norm, [0, 1, 2]) if target else None
        if log_p_cond is not None and self.logpro is not None:
            self.log_weight_step = float(log_p_cond - self.logpro)
        else:
            self.log_weight_step = 0.0

        self.last_flowis_trace = {
            'accepted': True,
            'trials': int(trials or 0),
            'scene_raw': scene_raw.tolist(),
            'scene_norm': scene_norm.reshape(-1).tolist(),
            'scene_behavior_norm': scene_behavior_norm.reshape(-1).tolist(),
            'scene_behavior_raw': scene_behavior_raw.reshape(-1).tolist(),
            'behavior': self.behavior,
            'log_q_cond': self.logpro,
            'log_p_cond': log_p_cond,
            'log_weight_step': self.log_weight_step,
            'sampler': 'rejection_sampling_1D',
        }
        return self.logpro, trials

    def act(self, action: Union[dict, str] = None):
        """
        Execute an action.

        For now, no action is supported because the vehicle takes all decisions
        of acceleration and lane changes on its own, based on the IDM and MOBIL models.

        :param action: the action
        """
        if self.crashed:
            return
        self.follow_road()
        self._flowis_step_count += 1
        scene_raw = self._scene_raw()
        critical, ttc, delta_v = self._critical_event(scene_raw)
        use_critical_mode = str(self.flowis_event_mode).lower() == "critical"
        interval = max(1, int(self.flowis_decision_interval))

        should_weight = (
            self._flowis_last_sample_step < 0
            or (self._flowis_step_count - self._flowis_last_sample_step) >= interval
        )
        weighted_step = should_weight and ((not use_critical_mode) or critical)

        if weighted_step:
            self.logpro, _ = self.flowis()
            self._flowis_last_sample_step = self._flowis_step_count
        else:
            self.logpro, _ = self.flowis_unweighted()

        self.last_flowis_trace['critical'] = bool(critical)
        self.last_flowis_trace['ttc'] = float(ttc)
        self.last_flowis_trace['delta_v'] = float(delta_v)
        self.last_flowis_trace['event_mode'] = self.flowis_event_mode
        self.last_flowis_trace['decision_interval'] = interval

        self.log_weight_episode += self.log_weight_step
        action = {}
        action['acceleration'],action['steering']=self.behavior,self.steering_control(self.target_lane_index)

        Vehicle.act(self, action) 

    def step(self, dt: float):
        """
        Step the simulation.

        Increases a timer used for decision policies, and step the vehicle dynamics.

        :param dt: timestep
        """
        self.timer += dt
        super().step(dt)



