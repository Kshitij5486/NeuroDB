"""
metrics.py
NeuroDB - Day 4
Prometheus metrics for the FastAPI server.
"""

from prometheus_client import Counter, Histogram, Gauge

# ── Counters ──────────────────────────────────────────────────────────────────
QUERIES_TOTAL = Counter(
    "neurodb_queries_total",
    "Total number of queries optimized"
)

AGENT_WINS = Counter(
    "neurodb_agent_wins_total",
    "Number of times agent beat PostgreSQL default"
)

AGENT_LOSSES = Counter(
    "neurodb_agent_losses_total",
    "Number of times PostgreSQL default was faster"
)

# ── Histograms ────────────────────────────────────────────────────────────────
AGENT_LATENCY = Histogram(
    "neurodb_agent_latency_ms",
    "Agent plan execution latency in ms",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500]
)

DEFAULT_LATENCY = Histogram(
    "neurodb_default_latency_ms",
    "PostgreSQL default plan execution latency in ms",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500]
)

IMPROVEMENT = Histogram(
    "neurodb_improvement_percent",
    "Latency improvement % over PostgreSQL default",
    buckets=[-50, -25, -10, 0, 10, 25, 50, 75, 100]
)

# ── Gauges ────────────────────────────────────────────────────────────────────
WIN_RATE = Gauge(
    "neurodb_win_rate",
    "Rolling win rate of agent vs PostgreSQL"
)

AVG_IMPROVEMENT = Gauge(
    "neurodb_avg_improvement_percent",
    "Rolling average improvement % over PostgreSQL"
)