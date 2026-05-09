"""
plot_results.py
NeuroDB - Day 3

Reads the training log and generates the learning curve graph.
This graph goes in your GitHub README and resume portfolio.

Usage:
    python agent/plot_results.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_PATH  = "runs/training_log.json"
PLOT_PATH = "runs/learning_curve.png"


def smooth(values: list, window: int = 50) -> np.ndarray:
    """Rolling average to smooth noisy reward curve."""
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(np.mean(values[start:i+1]))
    return np.array(out)


def plot():
    if not os.path.exists(LOG_PATH):
        print(f"ERROR: {LOG_PATH} not found. Run train.py first.")
        return

    print(f"Loading log from {LOG_PATH}...")
    with open(LOG_PATH) as f:
        log = json.load(f)

    episodes   = np.array(log["episodes"])
    rewards    = np.array(log["rewards"])
    agent_ms   = np.array(log["agent_ms"])
    default_ms = np.array(log["default_ms"])
    wins       = np.array(log["wins"])

    # rolling window metrics
    smooth_reward  = smooth(rewards.tolist(),  window=50)
    smooth_agent   = smooth(agent_ms.tolist(), window=50)
    smooth_default = smooth(default_ms.tolist(), window=50)

    # rolling win rate
    win_rate = []
    window   = 50
    for i in range(len(wins)):
        start = max(0, i - window + 1)
        win_rate.append(np.mean(wins[start:i+1]) * 100)
    win_rate = np.array(win_rate)

    # improvement %
    improvement = (smooth_default - smooth_agent) / smooth_default * 100

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(
        "NeuroDB — PPO Agent Learning Curve",
        fontsize=16, fontweight="bold", y=0.98
    )

    # Panel 1 — Reward
    ax = axes[0]
    ax.plot(episodes, rewards,       alpha=0.15, color="#4C8BF5", linewidth=0.5)
    ax.plot(episodes, smooth_reward, color="#4C8BF5", linewidth=2,
            label="Smoothed reward (window=50)")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylabel("Reward  (−log ms)", fontsize=11)
    ax.set_title("Episode Reward", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(episodes))

    # Panel 2 — Latency comparison
    ax = axes[1]
    ax.plot(episodes, smooth_agent,   color="#34A853", linewidth=2,
            label="PPO agent (ms)")
    ax.plot(episodes, smooth_default, color="#EA4335", linewidth=2,
            linestyle="--", label="PostgreSQL default (ms)")
    ax.fill_between(episodes, smooth_agent, smooth_default,
                    where=(smooth_agent < smooth_default),
                    alpha=0.2, color="#34A853", label="Agent faster")
    ax.fill_between(episodes, smooth_agent, smooth_default,
                    where=(smooth_agent >= smooth_default),
                    alpha=0.2, color="#EA4335", label="Default faster")
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.set_title("Agent Latency vs PostgreSQL Default", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(episodes))

    # Panel 3 — Win rate
    ax = axes[2]
    ax.plot(episodes, win_rate, color="#FBBC04", linewidth=2,
            label="Win rate % (window=50)")
    ax.axhline(y=40, color="gray", linestyle="--", linewidth=1,
               label="Random baseline (40%)")
    ax.axhline(y=50, color="#34A853", linestyle=":",
               linewidth=1, label="50% target")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Win rate (%)", fontsize=11)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_title("Agent Win Rate vs PostgreSQL Default", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(episodes))

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nLearning curve saved → {PLOT_PATH}")
    print("Add this image to your GitHub README.")

    # print final stats
    print("\n── Final 100 episode stats ──")
    print(f"  Avg reward     : {np.mean(rewards[-100:]):.4f}")
    print(f"  Avg agent ms   : {np.mean(agent_ms[-100:]):.3f}")
    print(f"  Avg default ms : {np.mean(default_ms[-100:]):.3f}")
    print(f"  Avg win rate   : {np.mean(win_rate[-100:]):.1f}%")
    print(f"  Avg improvement: {np.mean(improvement[-100:]):+.1f}%")


if __name__ == "__main__":
    plot()