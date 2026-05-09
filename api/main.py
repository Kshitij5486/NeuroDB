"""
main.py
NeuroDB - Day 4

FastAPI server that:
  - Accepts SQL queries via POST /optimize
  - Runs the trained PPO agent to pick best join order
  - Executes both agent plan and PostgreSQL default
  - Returns latency comparison in real time
  - Serves a live dashboard at GET /
  - Exposes Prometheus metrics at GET /metrics
"""

import os
import sys
import json
import time
import collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from stable_baselines3 import PPO
from env.neurodb_env   import NeuroDB, TRAINING_QUERIES
from env.query_parser  import extract_features
from env.pg_executor   import PGExecutor
from api.metrics import (
    QUERIES_TOTAL, AGENT_WINS, AGENT_LOSSES,
    AGENT_LATENCY, DEFAULT_LATENCY, IMPROVEMENT,
    WIN_RATE, AVG_IMPROVEMENT
)

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "NeuroDB",
    description = "Reinforcement Learning Query Optimizer",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Global state ──────────────────────────────────────────────────────────────

MODEL_PATH = "models/neurodb_ppo"
model      = None
executor   = None
env        = None

# rolling history for dashboard
history         = collections.deque(maxlen=100)
total_queries   = 0
total_wins      = 0


def load_model():
    global model, executor, env
    print("[NeuroDB API] Loading PPO model...")
    env      = NeuroDB()
    model    = PPO.load(MODEL_PATH, env=env)
    executor = PGExecutor()
    executor.connect()
    print("[NeuroDB API] Model loaded. Server ready.")


@app.on_event("startup")
async def startup():
    load_model()


@app.on_event("shutdown")
async def shutdown():
    if executor:
        executor.close()


# ── Request / Response schemas ────────────────────────────────────────────────

class OptimizeRequest(BaseModel):
    sql      : str
    tables   : list[str]          # table aliases e.g. ["c", "o", "n"]
    runs     : int   = 1          # how many times to run each plan


class OptimizeResponse(BaseModel):
    join_order      : list[str]
    hint            : str
    agent_ms        : float
    default_ms      : float
    improvement_pct : float
    agent_wins      : bool
    reward          : float
    action          : int


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the live dashboard."""
    html_path = os.path.join(
        os.path.dirname(__file__), "dashboard.html"
    )
    if os.path.exists(html_path):
        with open(html_path) as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Dashboard not found</h1>")


@app.post("/optimize", response_model=OptimizeResponse)
async def optimize(req: OptimizeRequest):
    """
    Optimize a SQL query using the trained PPO agent.

    The agent picks the best join order from all permutations
    of the provided table aliases. Returns latency comparison
    between the agent's plan and PostgreSQL's default plan.
    """
    global total_queries, total_wins

    if model is None:
        raise HTTPException(503, "Model not loaded")

    # extract feature vector
    try:
        obs, tables_found = extract_features(req.sql)
    except ValueError as e:
        raise HTTPException(400, f"SQL parse error: {e}")

    # agent picks action
    action, _ = model.predict(obs, deterministic=True)
    action    = int(action)

    # build join order from action
    import itertools
    perms        = list(itertools.permutations(req.tables))
    action       = action % len(perms)
    join_order   = list(perms[action])

    # build hint
    hint = executor.build_hint(join_order)

    # run both plans
    try:
        if req.runs > 1:
            agent_times   = sorted([
                executor.run_with_explain(req.sql, hint)
                for _ in range(req.runs)
            ])
            default_times = sorted([
                executor.run_default(req.sql)
                for _ in range(req.runs)
            ])
            agent_ms   = agent_times  [req.runs // 2]
            default_ms = default_times[req.runs // 2]
        else:
            agent_ms   = executor.run_with_explain(req.sql, hint)
            default_ms = executor.run_default(req.sql)
    except RuntimeError as e:
        raise HTTPException(500, f"Query execution failed: {e}")

    # compute metrics
    improvement = (default_ms - agent_ms) / default_ms * 100
    agent_wins  = agent_ms < default_ms
    import math
    reward      = -math.log(max(agent_ms, 0.01))

    # update counters
    total_queries += 1
    if agent_wins:
        total_wins += 1
        AGENT_WINS.inc()
    else:
        AGENT_LOSSES.inc()

    QUERIES_TOTAL.inc()
    AGENT_LATENCY.observe(agent_ms)
    DEFAULT_LATENCY.observe(default_ms)
    IMPROVEMENT.observe(improvement)

    win_rate = total_wins / total_queries * 100
    WIN_RATE.set(win_rate)
    AVG_IMPROVEMENT.set(improvement)

    # record in history
    record = {
        "id"            : total_queries,
        "timestamp"     : time.strftime("%H:%M:%S"),
        "join_order"    : join_order,
        "agent_ms"      : round(agent_ms,   3),
        "default_ms"    : round(default_ms, 3),
        "improvement"   : round(improvement, 2),
        "agent_wins"    : agent_wins,
        "win_rate"      : round(win_rate, 1),
    }
    history.append(record)

    return OptimizeResponse(
        join_order      = join_order,
        hint            = hint,
        agent_ms        = round(agent_ms,    3),
        default_ms      = round(default_ms,  3),
        improvement_pct = round(improvement, 2),
        agent_wins      = agent_wins,
        reward          = round(reward,      4),
        action          = action,
    )


@app.get("/history")
async def get_history():
    """Return last 100 optimization results for dashboard."""
    return JSONResponse(content=list(history))


@app.get("/stats")
async def get_stats():
    """Return overall statistics."""
    if total_queries == 0:
        return {"total_queries": 0, "win_rate": 0, "message": "No queries yet"}

    recent = list(history)
    avg_imp = (
        sum(h["improvement"] for h in recent) / len(recent)
        if recent else 0
    )

    return {
        "total_queries"     : total_queries,
        "total_wins"        : total_wins,
        "win_rate_%"        : round(total_wins / total_queries * 100, 1),
        "avg_improvement_%" : round(avg_imp, 2),
        "model_path"        : MODEL_PATH,
    }


@app.get("/queries")
async def list_queries():
    """Return all training queries available for testing."""
    return [
        {
            "id"     : i,
            "tables" : q["tables"],
            "sql"    : q["sql"].strip(),
        }
        for i, q in enumerate(TRAINING_QUERIES)
    ]


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content      = generate_latest(),
        media_type   = CONTENT_TYPE_LATEST,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}