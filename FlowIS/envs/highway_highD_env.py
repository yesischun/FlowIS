from __future__ import annotations
from pathlib import Path
import joblib 
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
def calculate_ttc(scene, lenth):
    """Calculate TTC for batched rear-end scenes.

    Uses first three columns as [rear_speed, front_speed, distance], and appends
    TTC as a new last column while preserving all original columns.
    """
    scene = np.asarray(scene, dtype=float)
    if scene.ndim != 2 or scene.shape[1] < 3:
        raise ValueError("scene must be a 2D array with at least 3 columns")

    speed_rear = scene[:, 0]
    speed_front = scene[:, 1]
    distance = scene[:, 2] - lenth
    speed_diff = speed_front - speed_rear

    safe_mask = speed_diff >= 0
    ttc = np.empty_like(speed_diff, dtype=float)
    ttc[safe_mask] = 999.0
    ttc[~safe_mask] = distance[~safe_mask] / (-speed_diff[~safe_mask])
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
                "lanes_count": 3,
                "vehicles_count": 50,
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "ego_spacing": 2,
                "vehicles_density": 1,
                # Light background traffic (kept far from AV/BV to reduce interference).
                "flowis_background_vehicles_count": 8,
                "flowis_background_min_distance": 120.0,
                "flowis_background_outer_lanes_only": True,
                # Keep one blocking/companion vehicle on each outer lane near ego.
                "flowis_background_companion_mode": True,
                "flowis_background_companion_offset_max": 8.0,
                "flowis_background_companion_speed_delta": 1.5,
                # Longitudinal spawn range for light background vehicles.
                "flowis_background_spawn_s_min": 140.0,
                "flowis_background_spawn_s_max": 320.0,
                
                "collision_reward": -10,  # The reward received when colliding with a vehicle.
                "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                # zero for other lanes.
                "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                # lower speeds according to config["reward_speed_range"].
                "lane_change_reward": 0,  # The reward received at each lane change action.
                "reward_speed_range": [20, 30],
                "normalize_reward": True,
                "offroad_terminal": False,
                "flowis_terminate_on_any_crash": True,
                "flowis_behavior_proposal": "tde",
                "flowis_scene_source": "tde",  # "tde" | "nde"
                "flowis_attack_mode": "mixed",  # "follow" | "cutin" | "mixed"
                "flowis_follow_probability": 0.9,  # when attack_mode="mixed": P(follow)
                "flowis_trace": True,
                # Keep ego within trained support by default; can be overridden in notebook.
                "flowis_ego_enable_lane_change": False,
                # Keep follow-mode controlled BV stable; cut-in attacker must keep lane-change ability.
                "flowis_follow_enable_lane_change": False,
                "flowis_attacker_enable_lane_change": True,
                "flowis_event_mode": "critical",  # "always" | "critical"
                "flowis_ttc_threshold": 4.0,
                "flowis_distance_threshold": 40.0,
                "flowis_delta_v_threshold": 0.5,
                # Keep scene source unbiased by default: no extra TTC-based truncation.
                "flowis_follow_ttc_rejection": False,
                "flowis_follow_ttc_rejection_threshold": 0.3,
                "flowis_decision_interval": 10,
                "flowis_burst_steps": 3,
                # Per-mode overrides (fallback to shared keys when omitted).
                "flowis_decision_interval_follow": 3,
                "flowis_decision_interval_cutin": 2,
                "flowis_burst_steps_follow": 3,
                "flowis_burst_steps_cutin": 3,
                # Keep weighted injections sparse to reduce IS variance.
                "flowis_max_weighted_steps": 10,
                "flowis_max_weighted_steps_follow": 10,
                "flowis_max_weighted_steps_cutin": 10,
                "flowis_importance_weighting": True,
                # Prefer exact conditional GMM sampling to avoid rejection/fallback bias.
                "flowis_use_exact_conditional": True,
                # Keep weights consistent with sampled actions; avoid post-sample clipping by default.
                "flowis_clip_actions": False,
                "flowis_artifact_dir": str((Path(__file__).resolve().parents[2] / "Artifacts").resolve()),
                "flowis_tde_path": None,
                "flowis_nde_path": None,
                "flowis_tde_follow_path": None,
                "flowis_nde_follow_path": None,
                "flowis_tde_cutin_path": None,
                "flowis_nde_cutin_path": None,
                # Cut-in attack controls
                "flowis_cutin_side": "left",  # "left" | "right"
                "flowis_cutin_trigger_distance": 25.0,
                "flowis_cutin_target": "ego",  # "background" | "ego"
                "flowis_cutin_target_offset": 20.0,  # target BV longitudinal offset from ego
                "flowis_cutin_attacker_offset": -5.0,  # attacker longitudinal offset from target
                "flowis_cutin_relative_clip": 120.0,  # clip sampled relative distance for stable triggering
                "flowis_cutin_enable_front_vehicle": True,
                "flowis_cutin_min_gap_rear_to_cutin": 8.0,
                "flowis_cutin_min_gap_cutin_to_front": 8.0,
                # Optional semantic mapping from sampled cutin scene_behavior -> env initialization.
                # Default feature order follows Experiments/HighD_env.ipynb cutin pipeline:
                # [v3, v2, v2y, v1, sx2, sx1, sy, a2x, a2y]
                "flowis_cutin_feature_names": ["v3", "v2", "v2y", "v1", "sx2", "sx1", "sy", "a2x", "a2y"],
                "flowis_cutin_ego_speed_feature": "v1",
                "flowis_cutin_attacker_speed_feature": "v2",
                "flowis_cutin_relative_to_ego_feature": "sx1",
                "flowis_cutin_relative_to_background_feature": "sx2",
                # Relative sign controls: attacker_s = target_s + sign * rel + attacker_offset.
                "flowis_cutin_rel_sign_to_ego": 1.0,
                "flowis_cutin_rel_sign_to_background": -1.0,
                # Backward-compatible raw index mapping (used when feature names unavailable).
                "flowis_cutin_ego_speed_index": 3,
                "flowis_cutin_attacker_speed_index": 1,
                "flowis_cutin_relative_index": 5,
            }
        )
        return config

    def _cutin_feature_names(self, nde_distribution: dict | None = None, tde_distribution: dict | None = None) -> list[str]:
        for dist in (nde_distribution, tde_distribution):
            if isinstance(dist, dict):
                names = dist.get("feature_names", None)
                if isinstance(names, (list, tuple)) and len(names) > 0:
                    return [str(x) for x in names]
        cfg_names = self.config.get("flowis_cutin_feature_names", None)
        if isinstance(cfg_names, (list, tuple)) and len(cfg_names) > 0:
            return [str(x) for x in cfg_names]
        return []

    def _feature_index(self, feature_names: list[str], feature_key: str, fallback_index: int) -> int:
        if feature_names:
            try:
                return int(feature_names.index(str(feature_key)))
            except ValueError:
                pass
        return int(fallback_index)

    def _resolve_artifact_path(self, config_key: str, default_filename: str) -> Path:
        configured = self.config.get(config_key, None)
        base_dir = self.config.get("flowis_artifact_dir", None)

        candidates: list[Path] = []
        if configured:
            candidates.append(Path(configured))
        if base_dir:
            candidates.append(Path(base_dir) / default_filename)
        candidates.append(Path(default_filename))

        for candidate in candidates:
            if candidate.exists():
                return candidate

        tried = "\n - ".join(str(p) for p in candidates)
        raise FileNotFoundError(
            f"FlowIS artifact not found for '{config_key}'. Tried:\n - {tried}"
        )

    def _mode_config_value(self, key_base: str, mode: str, default):
        mode_key = f"{key_base}_{mode}"
        return self.config.get(mode_key, self.config.get(key_base, default))

    def _load_mode_distributions(self, attack_mode: str):
        mode = str(attack_mode).lower()
        if mode == "cutin":
            tde_candidates = [
                ("flowis_tde_cutin_path", "FlowIS_cutin.joblib"),
                ("flowis_tde_path", "FlowIS_highway.joblib"),
            ]
            nde_candidates = [
                ("flowis_nde_cutin_path", "NDE_cutin.joblib"),
                ("flowis_nde_path", "NDE_highway.joblib"),
            ]
        else:
            tde_candidates = [
                ("flowis_tde_follow_path", "FlowIS_highway.joblib"),
                ("flowis_tde_path", "FlowIS_highway.joblib"),
            ]
            nde_candidates = [
                ("flowis_nde_follow_path", "NDE_highway.joblib"),
                ("flowis_nde_path", "NDE_highway.joblib"),
            ]

        tde_path = None
        nde_path = None
        tde_last_exc = None
        nde_last_exc = None
        for key, filename in tde_candidates:
            try:
                tde_path = self._resolve_artifact_path(key, filename)
                break
            except FileNotFoundError as exc:
                tde_last_exc = exc
        for key, filename in nde_candidates:
            try:
                nde_path = self._resolve_artifact_path(key, filename)
                break
            except FileNotFoundError as exc:
                nde_last_exc = exc

        if tde_path is None:
            raise tde_last_exc
        if nde_path is None:
            raise nde_last_exc

        return joblib.load(tde_path), joblib.load(nde_path)

    @staticmethod
    def _scene_dims_from_dist(distribution: dict) -> list[int]:
        dims = distribution.get("scene_dims", None) if isinstance(distribution, dict) else None
        if dims is not None:
            return [int(i) for i in dims]
        scaler = distribution.get("fit_scene", None) if isinstance(distribution, dict) else None
        if scaler is not None and hasattr(scaler, "n_features_in_"):
            return list(range(int(getattr(scaler, "n_features_in_"))))
        return [0, 1, 2]

    def _extract_scene_raw(self, scene_behavior: np.ndarray, distribution: dict) -> np.ndarray:
        sb = np.asarray(scene_behavior, dtype=float).reshape(1, -1)
        dims = self._scene_dims_from_dist(distribution)
        dims = [d for d in dims if 0 <= d < sb.shape[1]]
        if not dims:
            dims = list(range(min(3, sb.shape[1])))
        return sb[:, dims].reshape(1, -1)

    def _scene_log_prob(self, distribution: dict, scene_raw: np.ndarray) -> float | None:
        try:
            scene_norm = distribution["fit_scene"].transform(np.asarray(scene_raw, dtype=float).reshape(1, -1))
            dims = self._scene_dims_from_dist(distribution)
            logp = distribution["scene_behavior"].logpdf_marginal(scene_norm, dims)
            return float(np.asarray(logp).reshape(-1)[0])
        except Exception:
            return None

    def _scene_log_weight(
        self,
        scene_behavior: np.ndarray,
        scene_source: str,
        nde_distribution: dict,
        tde_distribution: dict,
    ) -> float:
        """
        Scene-level IS correction: log p(s)/q(s).
        - p(s): NDE scene marginal
        - q(s): selected scene source (NDE or TDE)
        """
        src = str(scene_source).lower()
        if src != "tde":
            return 0.0

        scene_raw = self._extract_scene_raw(scene_behavior, nde_distribution)
        log_p_s = self._scene_log_prob(nde_distribution, scene_raw)
        log_q_s = self._scene_log_prob(tde_distribution, scene_raw)
        if log_p_s is None or log_q_s is None:
            return 0.0
        return float(log_p_s - log_q_s)

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
        requested_mode = str(self.config.get("flowis_attack_mode", "mixed")).lower()
        if requested_mode == "mixed":
            follow_prob = float(self.config.get("flowis_follow_probability", 0.9))
            follow_prob = float(np.clip(follow_prob, 0.0, 1.0))
            attack_mode = "follow" if float(self.np_random.random()) < follow_prob else "cutin"
        elif requested_mode in {"follow", "cutin"}:
            attack_mode = requested_mode
        else:
            attack_mode = "follow"
        self._active_attack_mode = attack_mode

        self.TDE_highway, self.NDE_highway = self._load_mode_distributions(attack_mode)
        tde_distribution = self.TDE_highway
        nde_distribution = self.NDE_highway
        scene_source = str(self._mode_config_value("flowis_scene_source", attack_mode, "tde")).lower()
        scene_distribution = nde_distribution if scene_source == "nde" else tde_distribution
        use_follow_ttc_rejection = bool(self.config.get("flowis_follow_ttc_rejection", False))
        ttc_reject_th = float(self.config.get("flowis_follow_ttc_rejection_threshold", 0.3))
        while True:
            sample_out = scene_distribution['scene_behavior'].sample(n_samples=1)
            # Compatibility:
            # - sklearn GaussianMixture.sample returns (X, component_ids)
            # - custom truncated GMM may return only X
            if isinstance(sample_out, tuple):
                sample_out = sample_out[0]
            scene_behavior_norm = np.asarray(sample_out, dtype=float)
            if scene_behavior_norm.ndim == 1:
                scene_behavior_norm = scene_behavior_norm.reshape(1, -1)
            scene_behavior = scene_distribution['fit_scene_behavior'].inverse_transform(scene_behavior_norm)
            # By default, do not truncate follow scenes to preserve p(s)/q(s) correctness.
            if attack_mode == "cutin" or (not use_follow_ttc_rejection):
                break
            ttc = calculate_ttc(scene_behavior, 5)[0][4]
            if ttc > ttc_reject_th:
                break
        self._flowis_last_attack_mode = attack_mode
        self._flowis_last_scene_source = scene_source
        self._flowis_last_scene_behavior = scene_behavior.reshape(-1).tolist()
        scene_log_weight = self._scene_log_weight(
            scene_behavior,
            scene_source,
            nde_distribution,
            tde_distribution,
        )
        self._flowis_last_scene_log_weight = float(scene_log_weight)
        if attack_mode == "cutin":
            self._create_vehicles_cutin(scene_behavior, nde_distribution, tde_distribution, scene_log_weight)
        else:
            self._create_vehicles_follow(scene_behavior, nde_distribution, tde_distribution, scene_log_weight)

    def _create_vehicles_follow(
        self,
        scene_behavior: np.ndarray,
        nde_distribution: dict,
        tde_distribution: dict,
        scene_log_weight: float = 0.0,
    ) -> None:
        lane_count = int(self.config.get("lanes_count", 1))
        center_lane_id = max(0, lane_count // 2)
        center_lane = self.road.network.get_lane(("0", "1", center_lane_id))
        self.controlled_vehicles = []   
        # ego_vehicle = self.action_type.vehicle_class(
        #                     self.road,
        #                     position=np.array([100.0,0.0]),
        #                     speed=scene_behavior[0][0],
        #                     heading=0.0
        #                     )     
        ego_vehicle = IDMVehicle(
                        self.road,
                        position=center_lane.position(100.0, 0.0),
                        speed=scene_behavior[0][0],
                        heading=center_lane.heading_at(100.0),
                        target_speed=38,
                        enable_lane_change=bool(self.config.get("flowis_ego_enable_lane_change", False)),
                        )  
        # Keep AV visually distinct from background vehicles in rendering.
        ego_vehicle.color = (50, 200, 0)
        self.controlled_vehicles = []        
        self.controlled_vehicles.append(ego_vehicle)
        self.road.vehicles.append(ego_vehicle)

        # Background vehicle generated from sampled scene.
        bv_vehicle = FlowIS_Follow(
            road=self.road,
            position=center_lane.position(100.0 + scene_behavior[0][2], 0.0),
            speed   = scene_behavior[0][1],            
            heading = center_lane.heading_at(100.0 + scene_behavior[0][2]),
            )
        vehicle_follow = bv_vehicle.create_from(bv_vehicle)
        vehicle_follow.enable_lane_change = bool(self.config.get("flowis_follow_enable_lane_change", False))
        # Proposal distribution q(b|s): configurable, default uses TDE.
        proposal_mode = str(self._mode_config_value("flowis_behavior_proposal", "follow", "tde")).lower()
        if proposal_mode == "nde":
            vehicle_follow.proposal_distribution = nde_distribution
        else:
            vehicle_follow.proposal_distribution = tde_distribution
        # Target distribution p(b|s): always NDE for importance correction.
        vehicle_follow.target_distribution = nde_distribution
        # Backward compatibility with legacy code paths.
        vehicle_follow.NDE_distribution = vehicle_follow.proposal_distribution
        vehicle_follow.flowis_scene_context = scene_behavior[0].tolist()
        vehicle_follow.flowis_importance_weighting = bool(self.config.get("flowis_importance_weighting", True))
        vehicle_follow.flowis_use_exact_conditional = bool(self.config.get("flowis_use_exact_conditional", True))
        vehicle_follow.flowis_clip_actions = bool(self.config.get("flowis_clip_actions", False))
        vehicle_follow.flowis_max_weighted_steps = int(self._mode_config_value("flowis_max_weighted_steps", "follow", 10))
        vehicle_follow.log_weight_scene = float(scene_log_weight)
        vehicle_follow.log_weight_episode = float(scene_log_weight)
        # Sparse critical-event injection controls.
        vehicle_follow.flowis_event_mode = self.config.get("flowis_event_mode", "critical")
        vehicle_follow.flowis_ttc_threshold = float(self.config.get("flowis_ttc_threshold", 4.0))
        vehicle_follow.flowis_distance_threshold = float(self.config.get("flowis_distance_threshold", 40.0))
        vehicle_follow.flowis_delta_v_threshold = float(self.config.get("flowis_delta_v_threshold", 0.5))
        vehicle_follow.flowis_decision_interval = int(self._mode_config_value("flowis_decision_interval", "follow", 10))
        vehicle_follow.flowis_burst_steps = int(self._mode_config_value("flowis_burst_steps", "follow", 3))
        vehicle_follow.color = (200, 0, 150)
        # vehicle_follow.behavior= self.config["behavior"]
        self.road.vehicles.append(vehicle_follow) 
        self._spawn_light_background_vehicles(ego_vehicle, vehicle_follow)

    def _create_vehicles_cutin(
        self,
        scene_behavior: np.ndarray,
        nde_distribution: dict,
        tde_distribution: dict,
        scene_log_weight: float = 0.0,
    ) -> None:
        lane_count = int(self.config.get("lanes_count", 3))
        center_lane_id = max(0, lane_count // 2)
        center_lane = self.road.network.get_lane(("0", "1", center_lane_id))

        feature_names = self._cutin_feature_names(nde_distribution, tde_distribution)
        ego_idx = self._feature_index(
            feature_names,
            self.config.get("flowis_cutin_ego_speed_feature", "v1"),
            int(self.config.get("flowis_cutin_ego_speed_index", 3)),
        )
        attacker_idx = self._feature_index(
            feature_names,
            self.config.get("flowis_cutin_attacker_speed_feature", "v2"),
            int(self.config.get("flowis_cutin_attacker_speed_index", 1)),
        )
        front_idx = self._feature_index(
            feature_names,
            "v3",
            0,
        )
        rel_ego_idx = self._feature_index(
            feature_names,
            self.config.get("flowis_cutin_relative_to_ego_feature", "sx1"),
            int(self.config.get("flowis_cutin_relative_index", 5)),
        )
        rel_bg_idx = self._feature_index(
            feature_names,
            self.config.get("flowis_cutin_relative_to_background_feature", "sx2"),
            int(self.config.get("flowis_cutin_relative_index", 5)),
        )

        ego_speed = float(scene_behavior[0][ego_idx]) if scene_behavior.shape[1] > ego_idx else 26.0
        attacker_speed = float(scene_behavior[0][attacker_idx]) if scene_behavior.shape[1] > attacker_idx else ego_speed
        front_speed = float(scene_behavior[0][front_idx]) if scene_behavior.shape[1] > front_idx else attacker_speed
        sampled_rel_raw_ego = float(scene_behavior[0][rel_ego_idx]) if scene_behavior.shape[1] > rel_ego_idx else 5.0
        sampled_rel_raw_bg = float(scene_behavior[0][rel_bg_idx]) if scene_behavior.shape[1] > rel_bg_idx else 10.0
        rel_clip = abs(float(self.config.get("flowis_cutin_relative_clip", 12.0)))
        sampled_rel_raw_ego = float(np.clip(sampled_rel_raw_ego, -rel_clip, rel_clip))
        sampled_rel_raw_bg = float(np.clip(sampled_rel_raw_bg, -rel_clip, rel_clip))
        sx1 = abs(sampled_rel_raw_ego)  # rear -> cutin longitudinal gap
        sx2 = abs(sampled_rel_raw_bg)   # cutin -> front longitudinal gap

        ego_vehicle = IDMVehicle(
            self.road,
            position=center_lane.position(100.0, 0.0),
            speed=ego_speed,
            heading=center_lane.heading_at(100.0),
            target_speed=38,
            enable_lane_change=bool(self.config.get("flowis_ego_enable_lane_change", False)),
        )
        ego_vehicle.color = (50, 200, 0)
        self.controlled_vehicles = [ego_vehicle]
        self.road.vehicles.append(ego_vehicle)

        cutin_target_cfg = str(self.config.get("flowis_cutin_target", "ego")).lower()
        target_offset = float(self.config.get("flowis_cutin_target_offset", 20.0))
        target_s = 100.0 + target_offset
        gap_rear_to_cutin = max(
            float(self.config.get("flowis_cutin_min_gap_rear_to_cutin", 8.0)),
            float(abs(sampled_rel_raw_ego)),
        )
        gap_cutin_to_front = max(
            float(self.config.get("flowis_cutin_min_gap_cutin_to_front", 8.0)),
            float(abs(sampled_rel_raw_bg)),
        )
        if cutin_target_cfg == "ego":
            target_vehicle = ego_vehicle
            target_s = float(target_vehicle.position[0])
            sampled_rel = gap_rear_to_cutin
            rel_sign = 1.0
        else:
            # Rear/target vehicle should follow sampled v1 semantics.
            target_vehicle = IDMVehicle(
                self.road,
                position=center_lane.position(target_s, 0.0),
                speed=float(np.clip(ego_speed, 0.0, 40.0)),
                heading=center_lane.heading_at(target_s),
                target_speed=float(np.clip(ego_speed, 0.0, 40.0)),
                enable_lane_change=False,
            )
            target_vehicle.color = (100, 200, 255)
            self.road.vehicles.append(target_vehicle)
            sampled_rel = gap_rear_to_cutin
            rel_sign = 1.0

        side = str(self.config.get("flowis_cutin_side", "left")).lower()
        left_lane = min(lane_count - 1, center_lane_id + 1)
        right_lane = max(0, center_lane_id - 1)
        if side == "right":
            attacker_lane_id = right_lane
        else:
            attacker_lane_id = left_lane
        if attacker_lane_id == center_lane_id:
            attacker_lane_id = 0 if center_lane_id != 0 else lane_count - 1
        attacker_lane = self.road.network.get_lane(("0", "1", attacker_lane_id))

        attacker_offset_cfg = float(self.config.get("flowis_cutin_attacker_offset", 0.0))
        # Enforce geometric order: rear(target) < attacker(cutin) < front(optional).
        attacker_s = float(target_vehicle.position[0] + gap_rear_to_cutin + attacker_offset_cfg)
        attacker_seed = FlowIS_Follow(
            road=self.road,
            position=attacker_lane.position(attacker_s, 0.0),
            speed=attacker_speed,
            heading=attacker_lane.heading_at(attacker_s),
        )
        attacker = attacker_seed.create_from(attacker_seed)
        attacker.enable_lane_change = bool(self.config.get("flowis_attacker_enable_lane_change", True))

        proposal_mode = str(self._mode_config_value("flowis_behavior_proposal", "cutin", "tde")).lower()
        if proposal_mode == "nde":
            attacker.proposal_distribution = nde_distribution
        else:
            attacker.proposal_distribution = tde_distribution
        attacker.target_distribution = nde_distribution
        attacker.NDE_distribution = attacker.proposal_distribution
        attacker.flowis_scene_context = scene_behavior[0].tolist()
        attacker.flowis_importance_weighting = bool(self.config.get("flowis_importance_weighting", True))
        attacker.flowis_use_exact_conditional = bool(self.config.get("flowis_use_exact_conditional", True))
        attacker.flowis_clip_actions = bool(self.config.get("flowis_clip_actions", False))
        attacker.flowis_max_weighted_steps = int(self._mode_config_value("flowis_max_weighted_steps", "cutin", 10))
        attacker.log_weight_scene = float(scene_log_weight)
        attacker.log_weight_episode = float(scene_log_weight)
        attacker.flowis_target_vehicle = target_vehicle
        attacker.flowis_cutin_enabled = True
        base_trigger = float(self.config.get("flowis_cutin_trigger_distance", 25.0))
        attacker.flowis_cutin_trigger_distance = max(base_trigger, sx1 + 5.0)
        attacker.flowis_cutin_commit_window = (-max(10.0, sx1 + 5.0), max(base_trigger, sx1 + 5.0))
        attacker.flowis_event_mode = self.config.get("flowis_event_mode", "critical")
        attacker.flowis_ttc_threshold = float(self.config.get("flowis_ttc_threshold", 4.0))
        attacker.flowis_distance_threshold = float(self.config.get("flowis_distance_threshold", 40.0))
        attacker.flowis_delta_v_threshold = float(self.config.get("flowis_delta_v_threshold", 0.5))
        attacker.flowis_decision_interval = int(self._mode_config_value("flowis_decision_interval", "cutin", 2))
        attacker.flowis_burst_steps = int(self._mode_config_value("flowis_burst_steps", "cutin", 3))
        attacker.target_lane_index = ("0", "1", int(attacker_lane_id))
        attacker.color = (200, 0, 150)

        # Optional front vehicle in target lane to preserve sampled v3/sx2 context.
        if bool(self.config.get("flowis_cutin_enable_front_vehicle", True)):
            front_s = float(attacker_s + gap_cutin_to_front)
            front_vehicle = IDMVehicle(
                self.road,
                position=center_lane.position(front_s, 0.0),
                speed=float(np.clip(front_speed, 0.0, 40.0)),
                heading=center_lane.heading_at(front_s),
                target_speed=float(np.clip(front_speed, 0.0, 40.0)),
                enable_lane_change=False,
            )
            front_vehicle.color = (140, 170, 220)
            self.road.vehicles.append(front_vehicle)
        attacker.last_flowis_trace["cutin_init"] = {
            "feature_names": feature_names,
            "ego_idx": int(ego_idx),
            "attacker_idx": int(attacker_idx),
            "front_idx": int(front_idx),
            "rel_ego_idx": int(rel_ego_idx),
            "rel_bg_idx": int(rel_bg_idx),
            "sampled_rel_raw_ego": float(sampled_rel_raw_ego),
            "sampled_rel_raw_bg": float(sampled_rel_raw_bg),
            "sampled_rel_used": float(sampled_rel),
            "sx1_used": float(sx1),
            "sx2_used": float(sx2),
            "gap_rear_to_cutin": float(gap_rear_to_cutin),
            "gap_cutin_to_front": float(gap_cutin_to_front),
            "cutin_target": cutin_target_cfg,
            "rel_sign": float(rel_sign),
        }
        self.road.vehicles.append(attacker)

        self._spawn_light_background_vehicles(ego_vehicle, attacker)

    def _spawn_light_background_vehicles(self, ego_vehicle: Vehicle, bv_vehicle: Vehicle) -> None:
        count = int(self.config.get("flowis_background_vehicles_count", 0))
        if count <= 0:
            return

        min_distance = float(self.config.get("flowis_background_min_distance", 120.0))
        spawn_s_min = float(self.config.get("flowis_background_spawn_s_min", 140.0))
        spawn_s_max = float(self.config.get("flowis_background_spawn_s_max", 320.0))
        if spawn_s_max <= spawn_s_min:
            spawn_s_max = spawn_s_min + 10.0
        lane_count = int(self.config.get("lanes_count", 1))
        if lane_count <= 1:
            return

        protected = np.array([ego_vehicle.position, bv_vehicle.position], dtype=float)
        lane_positions: dict[int, list[float]] = {i: [] for i in range(lane_count)}
        lane_id_bv = int(getattr(bv_vehicle, "lane_index", ("0", "1", 0))[2])
        outer_only = bool(self.config.get("flowis_background_outer_lanes_only", True))
        companion_mode = bool(self.config.get("flowis_background_companion_mode", True))
        companion_offset_max = float(self.config.get("flowis_background_companion_offset_max", 8.0))
        companion_speed_delta = float(self.config.get("flowis_background_companion_speed_delta", 1.5))
        if outer_only:
            candidate_lanes = [0, lane_count - 1]
            candidate_lanes = sorted(set(candidate_lanes))
            count = max(count, 2)
        else:
            candidate_lanes = list(range(lane_count))
        # Keep BV lane clean if possible.
        if lane_id_bv in candidate_lanes and len(candidate_lanes) > 1:
            candidate_lanes = [lid for lid in candidate_lanes if lid != lane_id_bv]

        base_speed = float(np.clip(ego_vehicle.speed, 18.0, 32.0))

        def spawn_companion_on_lane(lane_id: int) -> bool:
            lane = self.road.network.get_lane(("0", "1", lane_id))
            ego_s = float(getattr(ego_vehicle, "position", np.array([100.0, 0.0]))[0])
            offset = float(self.np_random.uniform(-abs(companion_offset_max), abs(companion_offset_max)))
            longitudinal = float(np.clip(ego_s + offset, 5.0, 1500.0))
            pos = lane.position(longitudinal, 0.0)
            speed = float(np.clip(ego_vehicle.speed + self.np_random.normal(0.0, companion_speed_delta), 16.0, 34.0))
            bg = IDMVehicle(
                self.road,
                position=pos,
                speed=speed,
                heading=lane.heading_at(longitudinal),
                target_speed=speed,
                enable_lane_change=False,
            )
            self.road.vehicles.append(bg)
            lane_positions[lane_id].append(longitudinal)
            return True

        def try_spawn_on_lane(lane_id: int) -> bool:
            lane = self.road.network.get_lane(("0", "1", lane_id))
            for _ in range(40):
                longitudinal = float(self.np_random.uniform(spawn_s_min, spawn_s_max))
                pos = lane.position(longitudinal, 0.0)
                if np.any(np.linalg.norm(protected - pos, axis=1) < min_distance):
                    continue
                if any(abs(longitudinal - s) < 35.0 for s in lane_positions[lane_id]):
                    continue
                speed = float(np.clip(base_speed + self.np_random.normal(0.0, 1.5), 16.0, 34.0))
                bg = IDMVehicle(
                    self.road,
                    position=pos,
                    speed=speed,
                    heading=lane.heading_at(longitudinal),
                    target_speed=speed,
                    enable_lane_change=False,
                )
                self.road.vehicles.append(bg)
                lane_positions[lane_id].append(longitudinal)
                return True
            return False

        # Guarantee at least one background vehicle on each outer lane.
        spawned = 0
        if outer_only and len(candidate_lanes) >= 2:
            for forced_lane in candidate_lanes:
                if spawned >= count:
                    break
                if companion_mode:
                    ok = spawn_companion_on_lane(int(forced_lane))
                else:
                    ok = try_spawn_on_lane(int(forced_lane))
                if ok:
                    spawned += 1

        # Fill remaining quota.
        trials = 0
        max_trials = max(60, count * 20)
        while spawned < count and trials < max_trials:
            trials += 1
            lane_id = int(candidate_lanes[int(self.np_random.integers(0, len(candidate_lanes)))])
            if try_spawn_on_lane(lane_id):
                spawned += 1

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
        """The episode is over when crash/offroad termination conditions are met."""
        any_crash = False
        if bool(self.config.get("flowis_terminate_on_any_crash", True)):
            any_crash = any(bool(getattr(v, "crashed", False)) for v in self.road.vehicles)
        else:
            any_crash = bool(self.vehicle.crashed)

        return any_crash or (
            self.config["offroad_terminal"] and not self.vehicle.on_road
        )

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]
    
    def _info(self, obs, action = None):
        info=super()._info(obs, action)
        av=self.road.vehicles[0]
        flowis_vehicle = None
        for veh in self.road.vehicles[1:]:
            if isinstance(veh, FlowIS_Follow):
                flowis_vehicle = veh
                break
        if flowis_vehicle is None and len(self.road.vehicles) > 1:
            flowis_vehicle = self.road.vehicles[1]
        behavior_val = float(getattr(flowis_vehicle, "behavior", 0.0))
        info['scene_behavior']=[av.speed,flowis_vehicle.speed,flowis_vehicle.position[0]-av.position[0],behavior_val]
        if self.config.get("flowis_trace", True):
            info["flowis"] = {
                "log_weight_step": float(getattr(flowis_vehicle, "log_weight_step", 0.0)),
                "log_weight_scene": float(getattr(flowis_vehicle, "log_weight_scene", 0.0)),
                "log_weight_episode": float(getattr(flowis_vehicle, "log_weight_episode", 0.0)),
                "weighted_steps": int(getattr(flowis_vehicle, "flowis_weighted_steps", 0)),
                "tde_behavior_steps": int(getattr(flowis_vehicle, "flowis_weighted_steps", 0)),
                "max_weighted_steps": int(getattr(flowis_vehicle, "flowis_max_weighted_steps", 0)),
                "attack_mode": str(getattr(self, "_active_attack_mode", self.config.get("flowis_attack_mode", "mixed"))),
                "scene_source": str(getattr(self, "_flowis_last_scene_source", self.config.get("flowis_scene_source", "tde"))),
                "sampled_scene_behavior": list(getattr(self, "_flowis_last_scene_behavior", [])),
                "trace": getattr(flowis_vehicle, "last_flowis_trace", {}),
            }
        return info
