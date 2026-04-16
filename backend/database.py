"""
database.py — Supabase + pgvector connection and schema management.

All SQL runs against Supabase's Postgres via the supabase-py client (REST)
or directly through psycopg2 for DDL that needs raw SQL (CREATE EXTENSION, etc.).
"""

import os
import psycopg2
import psycopg2.extras
from psycopg2.extras import RealDictCursor, Json as PgJson
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
DATABASE_URL = os.environ["DATABASE_URL"]  # postgres://... direct connection


def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def get_pg_conn():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)


# ---------------------------------------------------------------------------
# Schema bootstrap — run once on first deploy
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Raw sales facts
CREATE TABLE IF NOT EXISTS sales (
    id             BIGSERIAL PRIMARY KEY,
    store_id       TEXT NOT NULL,
    store_name     TEXT NOT NULL,
    region         TEXT NOT NULL,
    date           DATE NOT NULL,
    product_category TEXT NOT NULL,
    units_sold     INTEGER NOT NULL,
    revenue        NUMERIC(12,2) NOT NULL,
    abv            NUMERIC(10,2) NOT NULL,
    created_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sales_store_date ON sales (store_id, date);
CREATE INDEX IF NOT EXISTS idx_sales_region      ON sales (region);
CREATE INDEX IF NOT EXISTS idx_sales_category    ON sales (product_category);

-- Chunked text summaries + embeddings for RAG
CREATE TABLE IF NOT EXISTS sales_chunks (
    id          BIGSERIAL PRIMARY KEY,
    chunk_text  TEXT NOT NULL,
    metadata    JSONB DEFAULT '{}',
    embedding   vector(1536),
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON sales_chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 50);
"""

MATCH_CHUNKS_FUNCTION = """
CREATE OR REPLACE FUNCTION match_sales_chunks(
    query_embedding vector(1536),
    match_threshold FLOAT DEFAULT 0.70,
    match_count     INT   DEFAULT 8
)
RETURNS TABLE (
    id          BIGINT,
    chunk_text  TEXT,
    metadata    JSONB,
    similarity  FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        sc.id,
        sc.chunk_text,
        sc.metadata,
        1 - (sc.embedding <=> query_embedding) AS similarity
    FROM sales_chunks sc
    WHERE 1 - (sc.embedding <=> query_embedding) > match_threshold
    ORDER BY sc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
"""


def bootstrap_schema():
    """Create tables and functions if they don't already exist."""
    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
            cur.execute(MATCH_CHUNKS_FUNCTION)
        conn.commit()
        print("Schema bootstrap complete.")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Data helpers used by rag_pipeline.py
# ---------------------------------------------------------------------------

def upsert_sales_rows(rows: list[dict]):
    """Bulk-insert sales rows, skipping exact duplicates."""
    if not rows:
        return
    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO sales
                    (store_id, store_name, region, date,
                     product_category, units_sold, revenue, abv)
                VALUES
                    (%(store_id)s, %(store_name)s, %(region)s, %(date)s,
                     %(product_category)s, %(units_sold)s, %(revenue)s, %(abv)s)
                ON CONFLICT DO NOTHING
                """,
                rows,
            )
        conn.commit()
    finally:
        conn.close()


def upsert_chunk(chunk_text: str, metadata: dict, embedding: list[float]):
    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO sales_chunks (chunk_text, metadata, embedding)
                VALUES (%s, %s, %s)
                """,
                (chunk_text, PgJson(metadata), embedding),
            )
        conn.commit()
    finally:
        conn.close()


def similarity_search(query_embedding: list[float], match_count: int = 8) -> list[dict]:
    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM match_sales_chunks(%s, 0.65, %s)",
                (query_embedding, match_count),
            )
            return cur.fetchall()
    finally:
        conn.close()


def run_sql_query(sql: str, params=None) -> list[dict]:
    """Execute an arbitrary read-only SQL query and return rows as dicts."""
    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    finally:
        conn.close()
