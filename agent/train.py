"""
train.py
NeuroDB - Day 3

Trains a PPO agent on the NeuroDB Gymnasium environment.
Uses Stable Baselines3 PPO with custom logging.

Usage:
    python agent/train.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for Windows
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor   import Monitor

from env.neurodb_env import NeuroDB

# ── Paths ─────────────────────────────────────────────────────────────────────
os.makedirs("runs",   exist_ok=True)
os.makedirs("models", exist_ok=True)

MODEL_PATH   = "models/neurodb_ppo"
LOG_PATH     = "runs/training_log.json"
PLOT_PATH    = "runs/learning_curve.png"


# ── Custom callback — logs every episode ──────────────────────────────────────

class NeuroDBCallback(BaseCallback):
    """
    Fires after every episode step.
    Records reward, latency, and win/loss against PostgreSQL default.
    """

    def __init__(self, log_every: int = 50, verbose: int = 1):
        super().__init__(verbose)
        self.log_every   = log_every
        self.ep_rewards  = []
        self.ep_agent_ms = []
        self.ep_def_ms   = []
        self.ep_wins     = []
        self.start_time  = None

    def _on_training_start(self):
        self.start_time = time.time()
        print("\n" + "=" * 60)
        print("  NeuroDB PPO Training Started")
        print("=" * 60)

    def _on_step(self) -> bool:
        # infos is a list (one per env)
        info = self.locals["infos"][0]

        if "agent_ms" not in info:
            return True

        agent_ms   = info["agent_ms"]
        default_ms = info["default_ms"]
        reward     = self.locals["rewards"][0]

        self.ep_rewards .append(float(reward))
        self.ep_agent_ms.append(float(agent_ms))
        self.ep_def_ms  .append(float(default_ms))
        self.ep_wins    .append(1 if agent_ms < default_ms else 0)

        ep = len(self.ep_rewards)

        if ep % self.log_every == 0:
            recent_r   = self.ep_rewards  [-self.log_every:]
            recent_a   = self.ep_agent_ms [-self.log_every:]
            recent_d   = self.ep_def_ms   [-self.log_every:]
            recent_w   = self.ep_wins     [-self.log_every:]
            elapsed    = time.time() - self.start_time
            win_rate   = sum(recent_w) / len(recent_w) * 100
            avg_imp    = (
                sum((d - a) / d * 100
                    for a, d in zip(recent_a, recent_d))
                / len(recent_a)
            )

            print(f"  Ep {ep:5d} | "
                  f"reward={np.mean(recent_r):7.4f} | "
                  f"agent={np.mean(recent_a):6.2f}ms | "
                  f"default={np.mean(recent_d):6.2f}ms | "
                  f"wins={win_rate:5.1f}% | "
                  f"improve={avg_imp:+6.1f}% | "
                  f"t={elapsed:.0f}s")

        return True

    def _on_training_end(self):
        elapsed = time.time() - self.start_time
        total   = len(self.ep_rewards)
        wins    = sum(self.ep_wins)
        all_imp = [
            (d - a) / d * 100
            for a, d in zip(self.ep_agent_ms, self.ep_def_ms)
        ]

        print("\n" + "=" * 60)
        print("  TRAINING COMPLETE")
        print("=" * 60)
        print(f"  Total episodes   : {total}")
        print(f"  Total time       : {elapsed:.1f}s")
        print(f"  Final avg reward : {np.mean(self.ep_rewards[-100:]):.4f}")
        print(f"  Agent win rate   : {wins/total*100:.1f}%")
        print(f"  Avg improvement  : {np.mean(all_imp):+.1f}%")
        print(f"  Best improvement : {max(all_imp):+.1f}%")

        # save log
        log = {
            "episodes"  : list(range(1, total + 1)),
            "rewards"   : self.ep_rewards,
            "agent_ms"  : self.ep_agent_ms,
            "default_ms": self.ep_def_ms,
            "wins"      : self.ep_wins,
        }
        with open(LOG_PATH, "w") as f:
            json.dump(log, f)
        print(f"\n  Log saved  → {LOG_PATH}")


# ── Training ──────────────────────────────────────────────────────────────────

def train(total_episodes: int = 2000):
    """
    Train PPO agent for `total_episodes` episodes.
    Each episode = 1 step (one query optimization decision).
    So total_timesteps = total_episodes.
    """

    print("Initializing NeuroDB environment...")
    raw_env = NeuroDB()
    env     = Monitor(raw_env)

    print("Building PPO agent...")
    model = PPO(
        policy             = "MlpPolicy",
        env                = env,
        learning_rate      = 3e-4,
        n_steps            = 256,       # collect 64 steps before each update
        batch_size         = 64,
        n_epochs           = 10,       # number of gradient steps per update
        gamma              = 0.99,
        gae_lambda         = 0.95,
        clip_range         = 0.2,
        ent_coef           = 0.01,     # entropy bonus — encourages exploration
        vf_coef            = 0.5,
        max_grad_norm      = 0.5,
        verbose            = 0,        # we handle logging ourselves
        tensorboard_log    = "runs/tb",
    )

    callback = NeuroDBCallback(log_every=50, verbose=1)

    print(f"Training for {total_episodes} episodes...\n")
    model.learn(
        total_timesteps    = total_episodes,
        callback           = callback,
        progress_bar       = False,
    )

    # save model
    model.save(MODEL_PATH)
    print(f"\n  Model saved → {MODEL_PATH}.zip")

    env.close()
    return model, callback


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model, cb = train(total_episodes=10000)
    print("\nTraining done. Run next:")
    print("  python agent/evaluate.py")
    print("  python agent/plot_results.py")