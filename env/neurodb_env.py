"""
neurodb_env.py
NeuroDB - Day 2 (updated Day 3)
Full Gymnasium RL environment for query optimization.
"""

import math
import itertools
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.query_parser import extract_features, FEATURE_DIM
from env.pg_executor  import PGExecutor


TRAINING_QUERIES = [
    {
        "sql": """
            SELECT c.c_name, SUM(o.o_totalprice) AS revenue
            FROM customer c
            JOIN orders o ON c.c_custkey = o.o_custkey
            JOIN nation n ON c.c_nationkey = n.n_nationkey
            WHERE o.o_orderdate >= '1993-01-01'
              AND o.o_orderdate <  '1997-01-01'
            GROUP BY c.c_name
            ORDER BY revenue DESC
            LIMIT 20
        """,
        "tables": ["c", "o", "n"],
    },
    {
        "sql": """
            SELECT o.o_orderkey, SUM(l.l_extendedprice) AS total
            FROM orders o
            JOIN lineitem l ON l.l_orderkey = o.o_orderkey
            JOIN customer c ON o.o_custkey = c.c_custkey
            WHERE l.l_shipdate >= '1994-01-01'
              AND l.l_discount BETWEEN 0.05 AND 0.07
            GROUP BY o.o_orderkey
            ORDER BY total DESC
            LIMIT 20
        """,
        "tables": ["o", "l", "c"],
    },
    {
        "sql": """
            SELECT n.n_name, COUNT(*) AS cnt,
                   AVG(c.c_acctbal) AS avg_bal
            FROM nation n
            JOIN customer c ON c.c_nationkey = n.n_nationkey
            JOIN orders o   ON o.o_custkey   = c.c_custkey
            WHERE o.o_orderstatus = 'F'
            GROUP BY n.n_name
            ORDER BY cnt DESC
        """,
        "tables": ["n", "c", "o"],
    },
    {
        "sql": """
            SELECT s.s_name, SUM(l.l_quantity) AS total_qty
            FROM supplier s
            JOIN lineitem l ON l.l_suppkey   = s.s_suppkey
            JOIN nation n   ON s.s_nationkey = n.n_nationkey
            WHERE n.n_name IN ('GERMANY', 'FRANCE', 'JAPAN')
              AND l.l_shipmode = 'MAIL'
            GROUP BY s.s_name
            ORDER BY total_qty DESC
            LIMIT 20
        """,
        "tables": ["s", "l", "n"],
    },
    {
        "sql": """
            SELECT c.c_mktsegment, o.o_orderpriority,
                   COUNT(*) AS order_count
            FROM customer c
            JOIN orders o ON o.o_custkey   = c.c_custkey
            JOIN nation n ON c.c_nationkey = n.n_nationkey
            WHERE c.c_acctbal > 0
            GROUP BY c.c_mktsegment, o.o_orderpriority
            ORDER BY order_count DESC
        """,
        "tables": ["c", "o", "n"],
    },
]


def _all_permutations(tables: list[str]) -> list[tuple]:
    return list(itertools.permutations(tables))


class NeuroDB(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, render_mode=None):
        super().__init__()

        self.max_actions = 6

        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(FEATURE_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.max_actions)

        self.executor = PGExecutor()
        self.executor.connect()

        self._current_query = None
        self._current_perms = None
        self._obs           = None
        self._episode_count = 0
        self.history        = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        q = TRAINING_QUERIES[
            self.np_random.integers(0, len(TRAINING_QUERIES))
        ]
        self._current_query = q
        self._current_perms = _all_permutations(q["tables"])
        self._episode_count += 1
        self._obs, _ = extract_features(q["sql"])

        return self._obs.copy(), {}

    def step(self, action: int):
        assert self._current_query is not None, "Call reset() first"

        q     = self._current_query
        perms = self._current_perms

        action       = int(action) % len(perms)
        chosen_order = list(perms[action])

        # ── run 3x and take median to reduce OS noise ──────────────────
        hint = self.executor.build_hint(chosen_order)
        try:
            agent_runs = []
            for _ in range(3):
                agent_runs.append(
                    self.executor.run_with_explain(q["sql"], hint)
                )
            agent_ms = float(sorted(agent_runs)[1])
        except RuntimeError:
            agent_ms = 9999.0

        try:
            default_runs = []
            for _ in range(3):
                default_runs.append(
                    self.executor.run_default(q["sql"])
                )
            default_ms = float(sorted(default_runs)[1])
        except RuntimeError:
            default_ms = agent_ms

        # ── shaped reward ───────────────────────────────────────────────
        base_reward = -math.log(max(agent_ms, 0.01))
        if agent_ms < default_ms:
            improvement = (default_ms - agent_ms) / default_ms
            reward = base_reward + improvement * 2.0
        else:
            penalty = (agent_ms - default_ms) / default_ms
            reward = base_reward - penalty * 2.0

        self.history.append({
            "episode"    : self._episode_count,
            "action"     : action,
            "join_order" : chosen_order,
            "agent_ms"   : round(agent_ms,   3),
            "default_ms" : round(default_ms, 3),
            "reward"     : round(reward,     4),
            "agent_wins" : agent_ms < default_ms,
        })

        terminated = True
        truncated  = False
        info = {
            "agent_ms"  : agent_ms,
            "default_ms": default_ms,
            "join_order": chosen_order,
            "hint"      : hint,
        }

        return self._obs.copy(), reward, terminated, truncated, info

    def close(self):
        self.executor.close()

    def print_summary(self, last_n: int = 20):
        if not self.history:
            print("No history yet.")
            return
        recent  = self.history[-last_n:]
        wins    = sum(1 for h in recent if h["agent_wins"])
        avg_imp = sum(
            (h["default_ms"] - h["agent_ms"]) / h["default_ms"] * 100
            for h in recent
        ) / len(recent)
        print(f"\n── Last {len(recent)} episodes ──")
        print(f"  Agent wins     : {wins}/{len(recent)}")
        print(f"  Avg improvement: {avg_imp:.1f}%")
        print(f"  Last reward    : {recent[-1]['reward']:.4f}")
        print(f"  Last join order: {recent[-1]['join_order']}")
        print(f"  Agent ms       : {recent[-1]['agent_ms']}")
        print(f"  Default ms     : {recent[-1]['default_ms']}")