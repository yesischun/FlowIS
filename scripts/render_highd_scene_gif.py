"""Render one HighD scene rollout to GIF.

Usage:
    D:\\conda\\envs\\FlowIS\\python.exe scripts\\render_highd_scene_gif.py ^
      --steps 100 --scene-source nde --proposal nde --outfile Figure\\highd_scene.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
from gymnasium.envs.registration import registry

import highway_env  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--scene-source", choices=["tde", "nde"], default="nde")
    parser.add_argument("--proposal", choices=["tde", "nde"], default="nde")
    parser.add_argument(
        "--tde-path",
        default=r"D:\LocalSyncdisk\加速测试\FlowIS\Artifacts\FlowIS_highway.joblib",
    )
    parser.add_argument(
        "--nde-path",
        default=r"D:\LocalSyncdisk\加速测试\FlowIS\Artifacts\NDE_highway.joblib",
    )
    parser.add_argument("--outfile", default=r"Figure\highd_scene.gif")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if "HighDEnv-v0" not in registry and hasattr(highway_env, "register_highway_envs"):
        highway_env.register_highway_envs()

    out = Path(args.outfile)
    out.parent.mkdir(parents=True, exist_ok=True)

    env = gym.make("HighDEnv-v0", render_mode="rgb_array")
    env.unwrapped.configure(
        {
            "flowis_scene_source": args.scene_source,
            "flowis_behavior_proposal": args.proposal,
            "flowis_tde_path": args.tde_path,
            "flowis_nde_path": args.nde_path,
        }
    )

    frames = []
    try:
        obs, info = env.reset()
        _ = (obs, info)
        frame0 = env.render()
        if frame0 is not None:
            frames.append(frame0)

        for _ in range(args.steps):
            obs, reward, terminated, truncated, info = env.step(0)
            _ = (obs, reward, info)
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            if terminated or truncated:
                break
    finally:
        env.close()

    if not frames:
        print("No frames captured.")
        return 1

    imageio.mimsave(out, frames, fps=args.fps, loop=0)
    print(f"Saved GIF: {out.resolve()}")
    print(f"Frames: {len(frames)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

