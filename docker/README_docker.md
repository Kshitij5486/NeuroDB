# NeuroDB Docker Setup

## Requirements
- Docker Desktop installed and running
- 4GB free disk space
- 4GB RAM available for Docker

## Run the entire system with one command

```bash
docker-compose up
```

That's it. Docker will:
1. Pull PostgreSQL 16
2. Build the NeuroDB image
3. Load TPC-H benchmark data automatically
4. Start the API server with the trained PPO model

## Access

| URL | Description |
|---|---|
| http://localhost:8000 | Live dashboard |
| http://localhost:8000/docs | Interactive API docs |
| http://localhost:8000/metrics | Prometheus metrics |
| http://localhost:8000/health | Health check |

## Stop

```bash
docker-compose down
```

## Reset everything (including database)

```bash
docker-compose down -v
```

## Rebuild after code changes

```bash
docker-compose up --build
```