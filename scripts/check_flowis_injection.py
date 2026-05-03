"""Smoke checks for FlowIS injection and log-weight plumbing.

Run:
    conda run -n FlowIS python scripts/check_flowis_injection.py
"""

from __future__ import annotations

import numpy as np

from highway_env.envs.highway_highD_env import HighDEnv
from highway_env.vehicle import behavior as behavior_mod
from highway_env.vehicle.behavior import FlowIS_Follow


class IdentityScaler:
    def transform(self, x):
        return np.asarray(x, dtype=float)

    def inverse_transform(self, x):
        return np.asarray(x, dtype=float)


class ConstantSceneBehavior:
    def __init__(self, log_joint: float, log_marginal: float):
        self.log_joint = float(log_joint)
        self.log_marginal = float(log_marginal)

    def logpdf(self, x):
        _ = x
        return np.array([self.log_joint], dtype=float)

    def logpdf_marginal(self, x, dims):
        _ = (x, dims)
        return np.array([self.log_marginal], dtype=float)


class DummyAV:
    def __init__(self):
        self.speed = 12.0
        self.position = np.array([100.0, 0.0], dtype=float)


class DummyRoad:
    def __init__(self):
        self.vehicles = [DummyAV()]

    def neighbour_vehicles(self, vehicle, lane_index):
        _ = (vehicle, lane_index)
        return None, None


def _make_vehicle_stub():
    v = object.__new__(FlowIS_Follow)
    v.road = DummyRoad()
    v.speed = 11.0
    v.position = np.array([130.0, 0.0], dtype=float)
    v.ACC_MAX = 6.0
    v.behavior = 0.0
    v.logpro = None
    v.log_weight_step = 0.0
    v.log_weight_episode = 0.0
    v.last_flowis_trace = {}
    v.lane_index = ("a", "b", 0)
    v.acceleration = lambda ego_vehicle, front_vehicle, rear_vehicle: -1.2
    return v


def test_accept_path() -> None:
    vehicle = _make_vehicle_stub()

    proposal = {
        "fit_scene": IdentityScaler(),
        "fit_scene_behavior": IdentityScaler(),
        "scene_behavior": ConstantSceneBehavior(log_joint=0.0, log_marginal=0.0),
    }
    target = {
        "scene_behavior": ConstantSceneBehavior(log_joint=1.5, log_marginal=0.5),
    }
    vehicle.proposal_distribution = proposal
    vehicle.NDE_distribution = proposal
    vehicle.target_distribution = target

    original_sampler = behavior_mod.RejectAcceptSampling.rejection_sampling_1D

    def fake_sampler(scene, nde_intersection, dim, max_trials, random_state, type="sample"):
        _ = (scene, nde_intersection, dim, max_trials, random_state, type)
        return np.array([12.0, 11.0, 30.0, 0.2]), 0.7, 3

    behavior_mod.RejectAcceptSampling.rejection_sampling_1D = staticmethod(fake_sampler)
    try:
        log_q, trials = FlowIS_Follow.flowis(vehicle)
    finally:
        behavior_mod.RejectAcceptSampling.rejection_sampling_1D = original_sampler

    assert trials == 3
    assert abs(float(log_q) - 0.7) < 1e-9
    assert abs(float(vehicle.behavior) - 0.2) < 1e-9
    # target log p = 1.5 - 0.5 = 1.0, log q = 0.7 => logw = 0.3
    assert abs(float(vehicle.log_weight_step) - 0.3) < 1e-9
    assert vehicle.last_flowis_trace.get("accepted") is True
    assert vehicle.last_flowis_trace.get("sampler") == "rejection_sampling_1D_tde_weighted"


def test_fallback_path() -> None:
    vehicle = _make_vehicle_stub()
    proposal = {
        "fit_scene": IdentityScaler(),
        "fit_scene_behavior": IdentityScaler(),
        "scene_behavior": ConstantSceneBehavior(log_joint=0.0, log_marginal=0.0),
    }
    vehicle.proposal_distribution = proposal
    vehicle.NDE_distribution = proposal
    vehicle.target_distribution = None

    original_sampler = behavior_mod.RejectAcceptSampling.rejection_sampling_1D

    def fake_sampler(scene, nde_intersection, dim, max_trials, random_state, type="sample"):
        _ = (scene, nde_intersection, dim, max_trials, random_state, type)
        return None, None, 9

    behavior_mod.RejectAcceptSampling.rejection_sampling_1D = staticmethod(fake_sampler)
    try:
        log_q, trials = FlowIS_Follow.flowis(vehicle)
    finally:
        behavior_mod.RejectAcceptSampling.rejection_sampling_1D = original_sampler

    assert log_q is None
    assert trials == 9
    assert vehicle.last_flowis_trace.get("accepted") is False
    assert vehicle.last_flowis_trace.get("sampler") == "fallback_idm_tde_weighted"
    assert abs(float(vehicle.log_weight_step)) < 1e-12


def test_config_keys() -> None:
    config = HighDEnv.default_config()
    assert config["flowis_behavior_proposal"] in {"tde", "nde"}
    assert isinstance(config["flowis_trace"], bool)


def main() -> int:
    test_accept_path()
    print("[OK] accept path")

    test_fallback_path()
    print("[OK] fallback path")

    test_config_keys()
    print("[OK] config keys")

    print("FlowIS injection smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
