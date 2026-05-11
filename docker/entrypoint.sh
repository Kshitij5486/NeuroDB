#!/bin/sh
# NeuroDB entrypoint script
# Waits for PostgreSQL, loads data if needed, starts the API server

set -e

echo "========================================"
echo "  NeuroDB - Starting up"
echo "========================================"

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL at $PG_HOST:$PG_PORT..."
until pg_isready -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$PG_DB" > /dev/null 2>&1; do
    echo "  PostgreSQL not ready yet, retrying in 2s..."
    sleep 2
done
echo "PostgreSQL is ready."

# Create tpch database if it doesn't exist
echo "Setting up tpch database..."
python -c "
import psycopg2, os
conn = psycopg2.connect(
    host=os.getenv('PG_HOST','postgres'),
    port=os.getenv('PG_PORT','5432'),
    dbname='postgres',
    user=os.getenv('PG_USER','postgres'),
    password=os.getenv('PG_PASSWORD','postgres123')
)
conn.autocommit = True
cur = conn.cursor()
cur.execute(\"SELECT 1 FROM pg_database WHERE datname='tpch'\")
if not cur.fetchone():
    cur.execute('CREATE DATABASE tpch')
    print('Created tpch database.')
else:
    print('tpch database already exists.')
cur.close()
conn.close()
"

# Check if data is already loaded
echo "Checking if TPC-H data is loaded..."
DATA_EXISTS=$(python -c "
import psycopg2, os
try:
    conn = psycopg2.connect(
        host=os.getenv('PG_HOST','postgres'),
        port=os.getenv('PG_PORT','5432'),
        dbname='tpch',
        user=os.getenv('PG_USER','postgres'),
        password=os.getenv('PG_PASSWORD','postgres123')
    )
    cur = conn.cursor()
    cur.execute(\"SELECT COUNT(*) FROM information_schema.tables WHERE table_name='lineitem'\")
    count = cur.fetchone()[0]
    print('yes' if count > 0 else 'no')
    cur.close()
    conn.close()
except:
    print('no')
")

if [ "$DATA_EXISTS" = "no" ]; then
    echo "Loading TPC-H data (this takes 2-3 minutes on first run)..."
    python data/load_tpch.py
    echo "TPC-H data loaded successfully."
else
    echo "TPC-H data already loaded. Skipping."
fi

# Check if model exists
if [ ! -f "models/neurodb_ppo.zip" ]; then
    echo "No trained model found. Training a quick model..."
    python agent/train.py
    echo "Model trained."
else
    echo "Trained model found."
fi

echo ""
echo "========================================"
echo "  Starting NeuroDB API server"
echo "  Dashboard: http://localhost:8000"
echo "  API Docs:  http://localhost:8000/docs"
echo "========================================"
echo ""

# Start FastAPI server
exec uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1