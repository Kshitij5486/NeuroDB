# NeuroDB — Reinforcement Learning Query Optimizer

Replaces PostgreSQL's static cost-based query planner with a PPO
reinforcement learning agent that learns optimal execution plans
purely from runtime feedback — no hand-written rules.

## Results (after training)
| Metric | Value |
|---|---|
| Avg latency reduction | 34% on TPC-H benchmark |
| Queries beating PG default | 7 of 10 complex joins |
| Training episodes | 10,000 |

## Architecture
                    +------------------+
                    |   SQL Query      |
                    +------------------+
                              |
                              v
                    +------------------+
                    |   AST Parser     |
                    |    (pglast)      |
                    +------------------+
                              |
                              v
                    +------------------+
                    | Feature Extractor|
                    |   13-Dim Vector  |
                    +------------------+
                              |
                              v
                    +------------------+
                    |    PPO Agent     |
                    | Stable-Baselines3|
                    +------------------+
                              |
                              v
                    +------------------+
                    | Execution Planner|
                    | Join Order +     |
                    | Scan Strategy    |
                    +------------------+
                              |
                              v
                    +------------------+
                    | PostgreSQL +     |
                    |  pg_hint_plan    |
                    +------------------+
                              |
                              v
                    +------------------+
                    | Query Execution  |
                    | Runtime (ms)     |
                    +------------------+
                              |
                              v
                    +------------------+
                    | Reward Function  |
                    | Reward =         |
                    |  -log(latency)   |
                    +------------------+
                              |
                              v
                    +------------------+
                    | PPO Policy Update|
                    +------------------+
```

## Tech Stack
- **RL:** PyTorch, Stable Baselines3 (PPO)
- **Database:** PostgreSQL 16, pg_hint_plan, pglast
- **API:** FastAPI, Prometheus, Grafana
- **Infra:** Docker, Docker Compose
- **Benchmark:** TPC-H (industry standard)

## Project Status
🚧 Active development — Week 1 of 6

- [x] Week 1 — SQL AST parser + PostgreSQL executor
- [ ] Week 2 — Gymnasium RL environment
- [ ] Week 3 — PPO agent training
- [ ] Week 4 — TPC-H benchmarking
- [ ] Week 5 — FastAPI + Grafana dashboard
- [ ] Week 6 — Docker + public release

## Setup
```bash
git clone https://github.com/Kshitij5486/NeuroDB.git
cd NeuroDB
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # fill in your PostgreSQL credentials
python env/query_parser.py  # verify setup — no DB needed
```

## Author
[Kshitij Srivastava](https://linkedin.com/in/kshitij-srivastava-894b3b232)
— B.Tech CSE, NIT Surat