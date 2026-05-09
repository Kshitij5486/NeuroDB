"""
evaluate.py
NeuroDB - Day 3

Loads the trained PPO model and evaluates it against
PostgreSQL's default optimizer on all training queries.

Usage:
    python agent/evaluate.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO
from env.neurodb_env   import NeuroDB, TRAINING_QUERIES

MODEL_PATH = "models/neurodb_ppo"


def evaluate(n_episodes: int = 100):
    print("=" * 60)
    print("  NeuroDB — PPO Agent Evaluation")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH + ".zip"):
        print(f"\nERROR: No model found at {MODEL_PATH}.zip")
        print("Run python agent/train.py first.")
        return

    print(f"\nLoading model from {MODEL_PATH}.zip ...")
    env   = NeuroDB()
    model = PPO.load(MODEL_PATH, env=env)

    print(f"Evaluating over {n_episodes} episodes...\n")

    rewards       = []
    agent_times   = []
    default_times = []
    wins          = 0
    results_by_query = {i: [] for i in range(len(TRAINING_QUERIES))}

    for ep in range(n_episodes):
        obs, info_reset = env.reset()

        # deterministic=True means agent picks best action, no randomness
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        rewards.append(reward)
        agent_times.append(info["agent_ms"])
        default_times.append(info["default_ms"])
        if info["agent_ms"] < info["default_ms"]:
            wins += 1

    env.close()

    # ── Results ───────────────────────────────────────────────────────────────
    avg_agent   = np.mean(agent_times)
    avg_default = np.mean(default_times)
    improvement = (avg_default - avg_agent) / avg_default * 100
    win_rate    = wins / n_episodes * 100

    print("=" * 60)
    print("  EVALUATION RESULTS  (PPO Agent vs PostgreSQL Default)")
    print("=" * 60)
    print(f"  Episodes evaluated : {n_episodes}")
    print(f"  Avg reward         : {np.mean(rewards):.4f}")
    print()
    print(f"  PPO agent avg ms   : {avg_agent:.3f} ms")
    print(f"  PostgreSQL avg ms  : {avg_default:.3f} ms")
    print(f"  Latency reduction  : {improvement:+.1f}%")
    print()
    print(f"  Agent win rate     : {wins}/{n_episodes} ({win_rate:.1f}%)")
    print(f"  Best reward        : {max(rewards):.4f}")
    print(f"  Worst reward       : {min(rewards):.4f}")
    print()

    # comparison with random baseline
    print("  vs Random Baseline (Day 2):")
    print(f"    Random win rate  : 40.0%")
    print(f"    PPO win rate     : {win_rate:.1f}%")
    delta = win_rate - 40.0
    print(f"    Improvement      : {delta:+.1f} percentage points")
    print("=" * 60)

    if improvement > 0:
        print(f"\n  PPO agent is {improvement:.1f}% faster than PostgreSQL.")
        print("  These are your resume numbers.")
    else:
        print("\n  Agent needs more training. Run train.py with more episodes.")

    return {
        "avg_agent_ms"  : avg_agent,
        "avg_default_ms": avg_default,
        "improvement_%" : improvement,
        "win_rate_%"    : win_rate,
    }


if __name__ == "__main__":
    evaluate(n_episodes=100)