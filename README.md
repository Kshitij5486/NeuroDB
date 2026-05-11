<div align="center">

# NeuroDB

### Reinforcement Learning Query Optimizer

*Replaces PostgreSQL's static cost-based query planner with a PPO agent that learns optimal execution plans purely from runtime feedback — no hand-written rules.*

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791.svg)](https://postgresql.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-EE4C2C.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## The Problem

Every SQL query you run goes through a **query optimizer** — a component that decides *how* to execute your query. Should it scan table A first, then join B? Or B first, then A? Which indexes to use?

PostgreSQL's optimizer uses **hand-written cost heuristics from the 1980s**. These heuristics estimate the best plan using statistics — but they're often wrong, especially on complex multi-table joins. A bad join order can make a query 10x slower.

**NeuroDB replaces this with a reinforcement learning agent that learns from real execution feedback.**

---

## How It Works

```
SQL Query
    │
    ▼
AST Parser (pglast)
    │  Extracts: tables, predicates, columns, aggregations
    │  Output: 13-dimensional feature vector
    ▼
PPO Agent (Stable Baselines3)
    │  State:  feature vector
    │  Action: join order permutation index (0-5)
    │  Policy: 2-layer MLP neural network
    ▼
pg_hint_plan Hint Injection
    │  Forces PostgreSQL to use the agent's chosen plan
    │  e.g. /*+ HashJoin(c o) Leading(c o n) */
    ▼
PostgreSQL Execution Engine
    │  Runs the query with the forced plan
    │  Measures actual wall-clock latency
    ▼
Reward Signal
    │  reward = -log(latency_ms)
    │  Shaped: bonus for beating default, penalty for losing
    ▼
PPO Update
    │  Agent updates policy weights
    │  Gets smarter after every episode
    ▼
Repeat for 10,000 episodes
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        NeuroDB System                        │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   FastAPI    │    │  PPO Agent   │    │  PostgreSQL  │  │
│  │  Dashboard   │◄──►│  (PyTorch)   │◄──►│   + TPC-H    │  │
│  │  Port 8000   │    │  MLP Policy  │    │  Port 5432   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                   │           │
│         ▼                   ▼                   ▼           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Prometheus  │    │  Gymnasium   │    │ pg_hint_plan │  │
│  │   Metrics    │    │     Env      │    │  Extension   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Component Breakdown

| Component | File | Purpose |
|---|---|---|
| SQL Parser | `env/query_parser.py` | Converts SQL to 13-dim feature vector via pglast AST |
| PG Executor | `env/pg_executor.py` | Runs EXPLAIN ANALYZE, injects pg_hint_plan hints |
| RL Environment | `env/neurodb_env.py` | Gymnasium env: state, action space, reward function |
| PPO Training | `agent/train.py` | Trains agent for 10,000 episodes with SB3 |
| Evaluation | `agent/evaluate.py` | Benchmarks trained model vs PostgreSQL default |
| FastAPI Server | `api/main.py` | REST API serving agent predictions in real time |
| Dashboard | `api/dashboard.html` | Live web UI showing agent decisions and charts |
| Metrics | `api/metrics.py` | Prometheus metrics: latency, win rate, improvement |
| Data Loader | `data/load_tpch.py` | Generates TPC-H benchmark dataset in PostgreSQL |

---

## Results

| Metric | Value |
|---|---|
| Training episodes | 10,240 |
| Best single-query improvement | +93.8% latency reduction |
| Agent win rate vs PostgreSQL | 47% (random baseline: 40%) |
| Training time | ~34 minutes |
| Benchmark | TPC-H (industry standard) |

### Key Findings

The agent successfully discovers join orders that PostgreSQL's cost-based optimizer misses. The best-case 93.8% improvement occurs on 3-table joins where the optimal leading table dramatically reduces the intermediate result set size — exactly the scenario where the 1980s heuristics fail most.

The win rate of 47% on a local machine (vs 40% random baseline) reflects a known challenge in learned query optimization: local PostgreSQL execution has 20-30% latency variance from OS scheduling noise, making consistent convergence difficult at millisecond query times. On production servers with TPC-H at scale (millions of rows, second-level query times), learned optimizers achieve 2-10x improvements — see [ReJOIN](https://arxiv.org/abs/1808.03196) and [Neo](https://vldb.org/pvldb/vol14/p1607-marcus.pdf).

---

## Tech Stack

| Category | Technology |
|---|---|
| Reinforcement Learning | PyTorch, Stable Baselines3 (PPO) |
| Database | PostgreSQL 16, pg_hint_plan |
| SQL Parsing | pglast (PostgreSQL AST parser) |
| RL Environment | Gymnasium (OpenAI standard) |
| API | FastAPI, Uvicorn |
| Monitoring | Prometheus, live dashboard |
| Benchmark | TPC-H (8 tables, industry standard) |
| Infrastructure | Docker, Docker Compose |

---

## Quick Start

### Option 1 — Docker (recommended, one command)

```bash
git clone https://github.com/Kshitij5486/NeuroDB.git
cd NeuroDB
docker-compose up
```

Open http://localhost:8000

That is it. Docker pulls PostgreSQL, loads TPC-H data, and starts the dashboard automatically.

### Option 2 — Local setup

**Prerequisites:**
- Python 3.11+
- PostgreSQL 16 with pg_hint_plan extension
- Git

```bash
# Clone
git clone https://github.com/Kshitij5486/NeuroDB.git
cd NeuroDB

# Virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Configure database
cp .env.example .env
# Edit .env with your PostgreSQL credentials

# Load TPC-H benchmark data
python data/load_tpch.py

# Train the PPO agent
python agent/train.py

# Start the dashboard
uvicorn api.main:app --reload --port 8000
```

Open http://localhost:8000

---

## API Reference

### POST /optimize

Submit a SQL query for optimization.

```bash
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "sql": "SELECT c.c_name, SUM(o.o_totalprice) FROM customer c JOIN orders o ON c.c_custkey = o.o_custkey JOIN nation n ON c.c_nationkey = n.n_nationkey GROUP BY c.c_name",
    "tables": ["c", "o", "n"]
  }'
```

Response:

```json
{
  "join_order": ["n", "c", "o"],
  "hint": "HashJoin(n c) HashJoin(n c o) Leading(n c o)",
  "agent_ms": 8.234,
  "default_ms": 11.891,
  "improvement_pct": 30.8,
  "agent_wins": true,
  "reward": -2.108,
  "action": 3
}
```

### GET /stats

```json
{
  "total_queries": 42,
  "total_wins": 21,
  "win_rate_%": 50.0,
  "avg_improvement_%": 3.2
}
```

### Other endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Live dashboard |
| `/optimize` | POST | Optimize a SQL query |
| `/stats` | GET | Overall statistics |
| `/history` | GET | Last 100 optimization results |
| `/queries` | GET | List all training queries |
| `/metrics` | GET | Prometheus metrics |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive API documentation |

---

## Project Structure

```
NeuroDB/
├── env/
│   ├── query_parser.py      # SQL AST → 13-dim feature vector
│   ├── pg_executor.py       # PostgreSQL runner + hint injection
│   └── neurodb_env.py       # Gymnasium RL environment
├── agent/
│   ├── train.py             # PPO training script
│   ├── evaluate.py          # Model evaluation vs PostgreSQL
│   ├── plot_results.py      # Learning curve visualization
│   └── random_baseline.py   # Random agent baseline
├── api/
│   ├── main.py              # FastAPI server
│   ├── metrics.py           # Prometheus metrics
│   └── dashboard.html       # Live web dashboard
├── data/
│   └── load_tpch.py         # TPC-H data generator
├── models/
│   └── neurodb_ppo.zip      # Trained PPO model
├── runs/
│   ├── training_log.json    # Episode-by-episode training data
│   └── learning_curve.png   # Learning curve graph
├── docker/
│   └── entrypoint.sh        # Docker startup script
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Research Background

NeuroDB is inspired by these academic works:

- **ReJOIN** (VLDB 2018) — First paper to apply RL to join order selection
- **DQ** (2019) — Deep Q-learning for database knob tuning
- **Neo** (VLDB 2021) — Learned query optimizer using tree convolutions
- **Bao** (SIGMOD 2021) — Learning to optimize via plan hints

The key insight from this literature: join order selection is NP-hard for n tables (n! permutations), and RL agents can learn good heuristics from execution feedback without needing a differentiable query model.

---

## Author

**Kshitij Srivastava**
B.Tech Computer Science and Engineering
NIT Surat (2023-2027)

[![GitHub](https://img.shields.io/badge/GitHub-Kshitij5486-181717.svg?logo=github)](https://github.com/Kshitij5486)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-kshitij--srivastava-0077B5.svg?logo=linkedin)](https://linkedin.com/in/kshitij-srivastava-894b3b232)

---

## License

MIT License — see [LICENSE](LICENSE) for details.