"""
rag_pipeline.py — ETL from CSV → Postgres, chunk generation, and embedding.

Two responsibilities:
  1. ingest_csv(path)  — load sales CSV into `sales` table and build chunks+embeddings
  2. retrieve(query)   — embed the query, run similarity search, return context chunks
"""

import csv
import os
from itertools import groupby

import voyageai
from dotenv import load_dotenv

from database import (
    similarity_search,
    upsert_chunk,
    upsert_sales_rows,
)

load_dotenv()

# Voyage AI client — uses VOYAGE_API_KEY env var by default,
# but Anthropic re-exports Voyage under ANTHROPIC_API_KEY.
# voyageai>=0.3 also accepts an explicit api_key kwarg.
_voyage = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY") or os.environ["ANTHROPIC_API_KEY"])

EMBED_MODEL = "voyage-3"   # 1536 dimensions
CHUNK_SIZE_WEEKS = 1       # one week per chunk per store per category


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def embed_text(text: str) -> list[float]:
    """Return a 1536-d embedding using Voyage AI's voyage-3 model."""
    result = _voyage.embed([text], model=EMBED_MODEL, input_type="document")
    return result.embeddings[0]


# ---------------------------------------------------------------------------
# CSV ingestion
# ---------------------------------------------------------------------------

def ingest_csv(path: str):
    """
    Parse the sales CSV, insert rows into `sales`, then build and embed
    textual summaries (chunks) stored in `sales_chunks`.
    """
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "store_id":        r["store_id"],
                "store_name":      r["store_name"],
                "region":          r["region"],
                "date":            r["date"],
                "product_category": r["product_category"],
                "units_sold":      int(r["units_sold"]),
                "revenue":         float(r["revenue"]),
                "abv":             float(r["abv"]),
            })

    upsert_sales_rows(rows)
    print(f"Inserted {len(rows)} sales rows.")

    # Build chunks grouped by (store_id, product_category, date)
    keyfn = lambda r: (r["store_id"], r["product_category"], r["date"])
    sorted_rows = sorted(rows, key=keyfn)

    chunks_created = 0
    for (store_id, category, week_date), group in groupby(sorted_rows, key=keyfn):
        items = list(group)
        row = items[0]
        chunk_text = (
            f"Store: {row['store_name']} ({store_id}), Region: {row['region']}\n"
            f"Week of: {week_date}\n"
            f"Category: {category}\n"
            f"Units sold: {row['units_sold']}, "
            f"Revenue: ₹{row['revenue']:,.0f}, "
            f"ABV: ₹{row['abv']:,.1f}"
        )
        metadata = {
            "store_id": store_id,
            "store_name": row["store_name"],
            "region": row["region"],
            "date": week_date,
            "category": category,
        }
        embedding = embed_text(chunk_text)
        upsert_chunk(chunk_text, metadata, embedding)
        chunks_created += 1

    # Also create store-level weekly rollup chunks
    store_week_key = lambda r: (r["store_id"], r["date"])
    sorted_rows2 = sorted(rows, key=store_week_key)
    for (store_id, week_date), group in groupby(sorted_rows2, key=store_week_key):
        items = list(group)
        row = items[0]
        total_rev = sum(i["revenue"] for i in items)
        total_units = sum(i["units_sold"] for i in items)
        avg_abv = total_rev / total_units if total_units else 0
        categories = ", ".join(i["product_category"] for i in items)
        chunk_text = (
            f"Weekly summary — Store: {row['store_name']} ({store_id}), "
            f"Region: {row['region']}, Week: {week_date}\n"
            f"Total revenue: ₹{total_rev:,.0f}, "
            f"Total units: {total_units}, "
            f"Avg ABV: ₹{avg_abv:,.1f}\n"
            f"Categories active: {categories}"
        )
        metadata = {
            "store_id": store_id,
            "store_name": row["store_name"],
            "region": row["region"],
            "date": week_date,
            "type": "weekly_summary",
        }
        embedding = embed_text(chunk_text)
        upsert_chunk(chunk_text, metadata, embedding)
        chunks_created += 1

    print(f"Created {chunks_created} embedded chunks.")


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_context(query: str, k: int = 8) -> list[dict]:
    """Embed the query and return top-k similar chunks from pgvector."""
    result = _voyage.embed([query], model=EMBED_MODEL, input_type="query")
    q_embedding = result.embeddings[0]
    results = similarity_search(q_embedding, match_count=k)
    return [dict(r) for r in results]


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a single context string for Claude."""
    if not chunks:
        return "No relevant sales data chunks found."
    parts = []
    for i, c in enumerate(chunks, 1):
        sim = c.get("similarity", 0)
        parts.append(f"[Chunk {i} | similarity={sim:.2f}]\n{c['chunk_text']}")
    return "\n\n".join(parts)


if __name__ == "__main__":
    import sys
    from database import bootstrap_schema

    bootstrap_schema()
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "../data/sample_sales_data.csv"
    ingest_csv(csv_path)
