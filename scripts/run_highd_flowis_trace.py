"""Run a short HighD FlowIS rollout and export trace diagnostics.

Usage:
    python scripts/run_highd_flowis_trace.py --episodes 2 --max-steps 200 --proposal tde
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd

from gymnasium.envs.registration import registry
import highway_env  # noqa: F401  # needed to register envs
warnings.filterwarnings("ignore", message=".*Overriding environment .*already in registry.*")
if "HighDEnv-v0" not in registry and hasattr(highway_env, "register_highway_envs"):
    highway_env.register_highway_envs()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="HighDEnv-v0")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--proposal", choices=["tde", "nde"], default="tde")
    parser.add_argument("--outdir", default="Figure/flowis_trace")
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--tde-path", default=None)
    parser.add_argument("--nde-path", default=None)
    return parser.parse_args()


def collect_traces(env, episodes: int, max_steps: int) -> pd.DataFrame:
    rows = []
    for ep in range(episodes):
        obs, info = env.reset()
        _ = (obs, info)
        for t in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            _ = obs
            flow = info.get("flowis", {}) if isinstance(info, dict) else {}
            trace = flow.get("trace", {}) if isinstance(flow, dict) else {}
            rows.append(
                {
                    "episode": ep,
                    "step": t,
                    "reward": float(reward),
                    "done": bool(terminated or truncated),
                    "log_weight_step": flow.get("log_weight_step"),
                    "log_weight_episode": flow.get("log_weight_episode"),
                    "accepted": trace.get("accepted"),
                    "trials": trace.get("trials"),
                    "behavior": trace.get("behavior"),
                    "log_q_cond": trace.get("log_q_cond"),
                    "log_p_cond": trace.get("log_p_cond"),
                    "sampler": trace.get("sampler"),
                    "scene_raw": trace.get("scene_raw"),
                    "scene_behavior_raw": trace.get("scene_behavior_raw"),
                }
            )
            if terminated or truncated:
                break
    return pd.DataFrame(rows)


def save_plots(df: pd.DataFrame, outdir: Path) -> None:
    show = df.copy()
    for col in [
        "log_weight_step",
        "log_weight_episode",
        "trials",
        "behavior",
        "log_q_cond",
        "log_p_cond",
    ]:
        show[col] = pd.to_numeric(show[col], errors="coerce")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(show.index, show["log_weight_step"])
    axes[0].set_title("Per-step log weight")
    axes[0].set_xlabel("global step")
    axes[1].plot(show.index, show["log_weight_episode"])
    axes[1].set_title("Cumulative log weight")
    axes[1].set_xlabel("global step")
    axes[2].plot(show.index, show["trials"])
    axes[2].set_title("Rejection trials per step")
    axes[2].set_xlabel("global step")
    fig.tight_layout()
    fig.savefig(outdir / "trace_overview.png", dpi=150)
    plt.close(fig)

    sampler_counts = show["sampler"].value_counts(dropna=False).to_dict()
    (outdir / "sampler_counts.json").write_text(
        json.dumps(sampler_counts, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    env = gym.make(args.env_id)
    config_override = {"flowis_behavior_proposal": args.proposal, "flowis_trace": True}
    if args.artifact_dir:
        config_override["flowis_artifact_dir"] = args.artifact_dir
    if args.tde_path:
        config_override["flowis_tde_path"] = args.tde_path
    if args.nde_path:
        config_override["flowis_nde_path"] = args.nde_path
    try:
        env.unwrapped.configure(config_override)
    except Exception:
        pass

    try:
        df = collect_traces(env, args.episodes, args.max_steps)
    finally:
        env.close()

    if df.empty:
        print("No rollout rows collected.")
        return 1

    df.to_csv(outdir / "trace_rows.csv", index=False)
    save_plots(df, outdir)
    print(f"Saved: {outdir}")
    print(f"Rows: {len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
