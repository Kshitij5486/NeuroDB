"""
pg_executor.py
NeuroDB - PostgreSQL runner + EXPLAIN ANALYZE + hint injection
"""

import re
import os
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

DEFAULT_CONFIG = {
    "host"    : os.getenv("PG_HOST",     "localhost"),
    "port"    : int(os.getenv("PG_PORT", "5432")),
    "dbname"  : os.getenv("PG_DB",       "tpch"),
    "user"    : os.getenv("PG_USER",     "postgres"),
    "password": os.getenv("PG_PASSWORD", ""),
}


class PGExecutor:

    def __init__(self, config: dict = None):
        self.config = config or DEFAULT_CONFIG
        self.conn   = None
        self.cur    = None

    def connect(self):
        self.conn = psycopg2.connect(**self.config)
        self.conn.autocommit = True
        self.cur  = self.conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor
        )
        print(f"[PGExecutor] Connected → "
              f"{self.config['dbname']}@{self.config['host']}")

    def close(self):
        if self.cur:  self.cur.close()
        if self.conn: self.conn.close()

    def __enter__(self):
        self.connect(); return self

    def __exit__(self, *_):
        self.close()

    # ── core ──────────────────────────────────────────────────────────────

    def run_with_explain(self, sql: str, hint: str = None) -> float:
        assert self.cur, "Call connect() first"
        query      = f"/*+ {hint} */\n{sql}" if hint else sql
        explain    = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) {query}"
        try:
            self.cur.execute(explain)
            rows = self.cur.fetchall()
        except psycopg2.Error as e:
            raise RuntimeError(f"Query failed: {e}") from e
        plan = "\n".join(row[list(row.keys())[0]] for row in rows)
        return self._parse_ms(plan)

    @staticmethod
    def _parse_ms(plan: str) -> float:
        m = re.search(r"Execution Time:\s*([\d.]+)\s*ms", plan)
        if not m:
            raise RuntimeError(f"Cannot parse time from:\n{plan[:300]}")
        return float(m.group(1))

    def run_default(self, sql: str) -> float:
        return self.run_with_explain(sql)

    # ── hint builders ──────────────────────────────────────────────────────

    @staticmethod
    def build_hint(join_order: list[str],
                   join_type: str = "HashJoin") -> str:
        parts = []
        for i in range(2, len(join_order) + 1):
            parts.append(f"{join_type}({' '.join(join_order[:i])})")
        parts.append(f"Leading({' '.join(join_order)})")
        return " ".join(parts)

    # ── compare ────────────────────────────────────────────────────────────

    def compare_plans(self, sql: str, join_order: list[str],
                      join_type: str = "HashJoin",
                      runs: int = 3) -> dict:
        hint         = self.build_hint(join_order, join_type)
        default_times = sorted(
            [self.run_default(sql) for _ in range(runs)]
        )
        agent_times  = sorted(
            [self.run_with_explain(sql, hint) for _ in range(runs)]
        )
        d_ms = default_times[runs // 2]
        a_ms = agent_times [runs // 2]
        return {
            "default_ms"   : round(d_ms, 3),
            "agent_ms"     : round(a_ms, 3),
            "improvement_%": round((d_ms - a_ms) / d_ms * 100, 2),
            "agent_wins"   : a_ms < d_ms,
            "hint"         : hint,
        }


# ── offline tests (no DB needed) ──────────────────────────────────────────

if __name__ == "__main__":
    print("=== Hint builder ===")
    for order in [["c","o"], ["o","l","c"], ["c","o","l","s","n"]]:
        print(f"  {order}  →  {PGExecutor.build_hint(order)}")

    print("\n=== EXPLAIN parser ===")
    mock = "Seq Scan...\nPlanning Time: 0.3 ms\nExecution Time: 42.130 ms\n"
    ms   = PGExecutor._parse_ms(mock)
    assert ms == 42.13
    print(f"  Parsed {ms} ms  ✓")
    print("\nAll offline tests passed.")