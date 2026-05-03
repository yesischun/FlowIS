"""Preflight checks for Experiments/HighD_env.ipynb.

Run:
    python scripts/check_highd_setup.py
"""

from __future__ import annotations

import importlib
import sys


REQUIRED_MODULES = [
    "torch",
    "joblib",
    "gymnasium",
    "highway_env",
    "numpy",
    "pandas",
    "tqdm",
    "matplotlib",
    "sklearn",
    "scipy",
    "IPython",
    "rl_agents.trainer.evaluation",
    "rl_agents.agents.common.factory",
    "highway_env.utils_FlowIS",
    "highway_env.Normalizing_Flow.maf",
    "highway_env.vehicle.behavior",
]


def main() -> int:
    failed: list[tuple[str, str]] = []
    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
            print(f"[OK]   {module_name}")
        except Exception as exc:  # noqa: BLE001
            failed.append((module_name, f"{type(exc).__name__}: {exc}"))
            print(f"[FAIL] {module_name} -> {type(exc).__name__}: {exc}")

    if failed:
        print("\nPreflight failed. Missing/broken modules:")
        for name, err in failed:
            print(f" - {name}: {err}")
        return 1

    print("\nPreflight passed. HighD notebook imports should work.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
