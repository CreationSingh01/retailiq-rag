# RetailIQ — AI-Powered Retail Sales Analyst

Ask plain-English questions about your store's sales data and get instant, data-backed answers.

**Stack:** FastAPI · Supabase (pgvector) · Anthropic Claude (tool_use) · React (CDN, no build)

---

## Project Layout

```
retailiq-rag/
├── backend/
│   ├── main.py            # FastAPI app + agentic query loop
│   ├── rag_pipeline.py    # CSV ingestion, embedding, retrieval
│   ├── database.py        # Supabase/Postgres helpers + schema bootstrap
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   └── index.html         # Single-file React app (no build step)
├── data/
│   └── sample_sales_data.csv   # 10 stores · 3 regions · 12 weeks · 5 categories
├── DECISIONS.md           # Architecture decision record
├── railway.json           # Railway deployment config
└── .env.example           # Environment variable template
```

---

## Quick Start (local)

### 1. Prerequisites

- Python 3.12+
- A [Supabase](https://supabase.com) project with the `vector` extension enabled
- An [Anthropic API key](https://console.anthropic.com)

### 2. Configure environment

```bash
cp .env.example .env
# Fill in ANTHROPIC_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, DATABASE_URL
```

### 3. Install backend dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 4. Bootstrap the database schema

```bash
python -c "from database import bootstrap_schema; bootstrap_schema()"
```

### 5. Ingest sample data + generate embeddings

```bash
python rag_pipeline.py ../data/sample_sales_data.csv
```

This creates the `sales` and `sales_chunks` tables, loads ~600 sales rows, and generates
~720 embedded text chunks via voyage-3.

### 6. Start the backend

```bash
uvicorn main:app --reload
# API available at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### 7. Open the frontend

```bash
open ../frontend/index.html
# Or serve it: python -m http.server 3000 --directory ../frontend
```

The frontend defaults to `http://localhost:8000` as the API base. To override, set
`window.RETAILIQ_API` in the HTML or via a script tag before the app loads.

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe — returns `{"status":"ok"}` |
| `POST` | `/query` | Ask a natural-language question |
| `POST` | `/ingest` | Trigger CSV ingestion + embedding |
| `GET` | `/stores` | List all stores |
| `GET` | `/summary` | Revenue/units/ABV by region |

### POST /query

```json
{ "question": "Which stores have declining ABV this month?" }
```

Response:
```json
{
  "answer": "Two stores show declining ABV in March 2026: ...",
  "tool_calls": [
    { "tool": "query_sales_db", "input": { "sql": "..." }, "result_preview": "..." }
  ]
}
```

---

## Sample Questions

- "Which stores have declining ABV this week?"
- "Compare North vs South region revenue in February"
- "Top 3 stores by Wellness category sales in Q1"
- "Show stores where Medicines revenue dropped week-over-week"
- "Which store had the highest ABV in January?"
- "Are there any stores showing growth across all categories?"

---

## Data: Sample Sales Dataset

**10 stores** across **3 regions** (North, South, West) in Hyderabad.
Weekly data from **Jan 1 – Mar 12, 2026** across **5 categories**.

| Store | Region | Trend |
|-------|--------|-------|
| Ameerpet Central (S01) | North | Steady growth |
| Kukatpally Hub (S02) | North | Declining — ABV and revenue falling each week |
| Secunderabad East (S03) | North | Stable growth |
| Banjara Hills Premium (S04) | South | Strong growth, highest ABV |
| Jubilee Hills Select (S05) | South | Growing Jan, declining Feb–Mar |
| Himayatnagar Clinic (S06) | South | Flat/steady |
| Gachibowli Tech (S07) | West | Fastest growing store |
| HITEC City Express (S08) | West | Steady growth |
| Madhapur Wellness (S09) | West | High Wellness share, consistent growth |
| Kondapur General (S10) | West | Sharp decline — worst performer |

---

## Deploy to Railway

1. Push this repo to GitHub.
2. Create a new Railway project and select "Deploy from GitHub repo".
3. Set the root directory to `/backend` and point Railway to `railway.json`.
4. Add the environment variables from `.env.example` in Railway's variable store.
5. Railway will build the Dockerfile and expose the service on a public URL.

To serve the frontend, either:
- Add `StaticFiles` mounting in `main.py` (one line), or
- Deploy `frontend/index.html` to Vercel / Netlify / Railway static hosting and set
  `window.RETAILIQ_API` to your Railway backend URL.

---

## Architecture

See [DECISIONS.md](./DECISIONS.md) for full reasoning on every architectural choice.

```
User question
     │
     ▼
 Frontend (React CDN)
     │  POST /query
     ▼
 FastAPI  ──────────────────────────────────────┐
     │                                          │
     │  messages[]                              │
     ▼                                          │
 Claude claude-sonnet-4-6 (tool_use loop)       │
     │                                          │
     ├── query_sales_db ──► Postgres (sales)    │
     │                                          │
     └── get_rag_context ──► pgvector           │
              (sales_chunks, voyage-3 embeddings)│
                                                │
     ◄──────────────── final answer ────────────┘
```
