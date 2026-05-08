"""
random_baseline.py
NeuroDB - Day 2
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from env.neurodb_env import NeuroDB


def run_random_baseline(n_episodes: int = 50):
    print("=" * 55)
    print("  NeuroDB — Random Agent Baseline")
    print("=" * 55)

    env = NeuroDB()
    obs, _ = env.reset()

    print(f"\nObservation space : {env.observation_space}")
    print(f"Action space      : {env.action_space}")
    print(f"Running {n_episodes} episodes with random actions...\n")

    rewards       = []
    agent_times   = []
    default_times = []
    wins          = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        rewards.append(reward)
        agent_times.append(info["agent_ms"])
        default_times.append(info["default_ms"])
        if info["agent_ms"] < info["default_ms"]:
            wins += 1

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1:3d} | "
                  f"reward={reward:7.4f} | "
                  f"agent={info['agent_ms']:7.1f}ms | "
                  f"default={info['default_ms']:7.1f}ms | "
                  f"order={info['join_order']}")

    env.close()

    print("\n" + "=" * 55)
    print("  BASELINE RESULTS")
    print("=" * 55)
    print(f"  Episodes        : {n_episodes}")
    print(f"  Avg reward      : {np.mean(rewards):.4f}")
    print(f"  Avg agent ms    : {np.mean(agent_times):.2f}")
    print(f"  Avg default ms  : {np.mean(default_times):.2f}")
    print(f"  Random wins     : {wins}/{n_episodes} "
          f"({wins/n_episodes*100:.1f}%)")
    print(f"  Best reward     : {max(rewards):.4f}")
    print(f"  Worst reward    : {min(rewards):.4f}")
    print("\n  Save these numbers — PPO agent must beat them.")
    print("=" * 55)


if __name__ == "__main__":
    run_random_baseline(n_episodes=50)