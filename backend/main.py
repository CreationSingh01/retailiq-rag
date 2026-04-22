"""
main.py — FastAPI application for RetailIQ.

Endpoints:
  POST /query        — natural-language question → answer (RAG + Claude tool_use)
  POST /ingest       — trigger CSV ingestion + embedding pipeline
  GET  /health       — liveness probe
  GET  /stores       — list all stores
  GET  /summary      — aggregate revenue/units by region
"""

import json
import os
import re
from decimal import Decimal
from typing import Any, Optional

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from database import bootstrap_schema, run_sql_query
from rag_pipeline import format_context, ingest_csv, retrieve_context

load_dotenv()

# In the container: /app/frontend/ (COPY frontend/ ./frontend/ in Dockerfile)
# In dev: ../frontend relative to backend/
_here = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(_here, "frontend") if os.path.isdir(os.path.join(_here, "frontend")) \
    else os.path.join(_here, "..", "frontend")

app = FastAPI(title="RetailIQ", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
MODEL = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# Tool definitions (used by Claude's tool_use)
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "query_sales_db",
        "description": (
            "Run a read-only SQL query against the `sales` table in Postgres. "
            "Columns: store_id TEXT, store_name TEXT, region TEXT, date DATE, "
            "product_category TEXT, units_sold INT, revenue NUMERIC, abv NUMERIC. "
            "Always use GROUP BY and aggregates when summarising. "
            "Never use DELETE, UPDATE, INSERT, or DROP."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "A valid read-only PostgreSQL SELECT statement.",
                }
            },
            "required": ["sql"],
        },
    },
    {
        "name": "get_rag_context",
        "description": (
            "Search the vector store for semantically relevant sales data chunks. "
            "Use this to get qualitative context, trends, or when you need examples "
            "before forming an exact SQL query."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The natural-language question to search for.",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of chunks to retrieve (default 8).",
                    "default": 8,
                },
            },
            "required": ["query"],
        },
    },
]

SYSTEM_PROMPT = """You are RetailIQ, an expert AI retail sales analyst for a pharmacy/retail chain in Hyderabad.

You have access to two tools:
1. `query_sales_db` — executes SQL against the sales database.
2. `get_rag_context` — semantic search over pre-embedded sales summaries.

Workflow:
- For factual questions (revenue, units, ABV comparisons), use `query_sales_db`.
- For trend questions or when you need context first, use `get_rag_context` then refine with SQL.
- You may call tools multiple times in one turn to build up a complete answer.

Data notes:
- Dates are stored as DATE (e.g. '2026-01-08').
- `abv` = average basket value in ₹.
- Stores are across three regions: North, South, West.
- Product categories: Medicines, FMCG, Wellness, Surgical, Baby Care.
- Data covers Jan–Mar 2026, weekly granularity.

Always present numbers with ₹ symbol and commas. When comparing periods, state the % change.
Be concise but thorough. If a store is declining, say so plainly and give the numbers.
"""


# ---------------------------------------------------------------------------
# Tool execution router
# ---------------------------------------------------------------------------

def _is_safe_sql(sql: str) -> bool:
    """Reject any SQL that mutates data."""
    banned = re.compile(
        r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|GRANT|REVOKE)\b",
        re.IGNORECASE,
    )
    return not banned.search(sql)


def execute_tool(tool_name: str, tool_input: dict) -> Any:
    if tool_name == "query_sales_db":
        sql = tool_input.get("sql", "")
        if not _is_safe_sql(sql):
            return {"error": "Unsafe SQL detected. Only SELECT statements are allowed."}
        try:
            rows = run_sql_query(sql)
            return {"rows": [dict(r) for r in rows], "count": len(rows)}
        except Exception as exc:
            return {"error": str(exc)}

    elif tool_name == "get_rag_context":
        query = tool_input.get("query", "")
        k = tool_input.get("k", 8)
        chunks = retrieve_context(query, k=k)
        return {"context": format_context(chunks)}

    return {"error": f"Unknown tool: {tool_name}"}


# ---------------------------------------------------------------------------
# Agentic query loop
# ---------------------------------------------------------------------------

def run_agent(user_question: str) -> dict:
    """
    Run the Claude tool-use agentic loop.
    Returns {"answer": str, "tool_calls": list}.
    """
    messages = [{"role": "user", "content": user_question}]
    tool_calls_log = []

    for _iteration in range(6):  # max 6 tool-use rounds
        response = _client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Append assistant turn
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            # Extract final text answer
            text = next(
                (b.text for b in response.content if hasattr(b, "text")), ""
            )
            return {"answer": text, "tool_calls": tool_calls_log}

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result = execute_tool(block.name, block.input)
                tool_calls_log.append({
                    "tool": block.name,
                    "input": block.input,
                    "result_preview": str(result)[:300],
                })
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, default=lambda o: float(o) if isinstance(o, Decimal) else str(o)),
                })
            messages.append({"role": "user", "content": tool_results})
            continue

        # Unexpected stop reason
        break

    return {"answer": "I was unable to produce a complete answer. Please try rephrasing.", "tool_calls": tool_calls_log}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str


class IngestRequest(BaseModel):
    csv_path: str = "/workspaces/retailiq-rag/data/sample_sales_data.csv"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    bootstrap_schema()


@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL}


@app.post("/query")
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    result = run_agent(req.question)
    return result


@app.post("/ingest")
def ingest(req: Optional[IngestRequest] = None):
    if req is None:
        req = IngestRequest()
    try:
        ingest_csv(req.csv_path)
        return {"status": "success", "csv_path": req.csv_path}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"CSV not found: {req.csv_path}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/stores")
def list_stores():
    rows = run_sql_query(
        "SELECT DISTINCT store_id, store_name, region FROM sales ORDER BY store_id"
    )
    return {"stores": [dict(r) for r in rows]}


@app.get("/summary")
def summary():
    rows = run_sql_query(
        """
        SELECT
            region,
            SUM(revenue)    AS total_revenue,
            SUM(units_sold) AS total_units,
            AVG(abv)        AS avg_abv
        FROM sales
        GROUP BY region
        ORDER BY total_revenue DESC
        """
    )
    return {"summary": [dict(r) for r in rows]}
