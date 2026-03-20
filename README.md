# DecisionGPT

Evidence-based answers to **decision questions** using **RAG** (retrieval), **CAG** (cache), and **KAG** (knowledge graph), plus an LLM for structured recommendations.

**Stack:** FastAPI + Python · React + Tailwind + shadcn/ui · Redis · Neo4j · FAISS

---

## Repo layout

| Folder | What |
|--------|------|
| `backend/` | FastAPI API, ingestion, embeddings, pipeline (`README` inside has full detail) |
| `frontend/insight-navigator/` | React UI (query, pipeline trace, graph, documents) |
| `data/` | Optional: drop raw files here; backend also uses `backend/data/` when run from `backend/` |
| `shared/` | Shared assets / future shared code |

---

## Prerequisites

- **Python 3.11+**
- **Node.js** (for the UI)
- **Docker Desktop** (Redis + Neo4j)
- **OpenAI API key** (embeddings + chat)

---

## Backend (API)

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip setuptools wheel
python -m pip install .
Copy-Item .env.example .env   # edit OPENAI_API_KEY
```

Start Redis + Neo4j (from repo root):

```powershell
cd backend\docker
docker compose up -d
```

Run the API (from `backend/`):

```powershell
$env:PYTHONPATH = "."
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Health: `http://localhost:8000/health`  
- Docs: `http://localhost:8000/docs`

Put documents under **`backend/data/raw/`**, then ingest + build index/graph (see `backend/README.md`).

---

## Frontend (UI)

```text
cd frontend/insight-navigator
npm install
npm run dev
```

Open the URL Vite prints (often `http://localhost:5173` or `8080`).

---

## Security

- **Never commit** `backend/.env` or real API keys.  
- Use `.env.example` as a template only.

---

## License

Add your license here if you use one.
