# DecisionGPT

Made with love for **AI** and **knowledge** — and for anyone who wants decisions explained with real evidence, not vibes.

---

## What it does

**DecisionGPT** helps you answer *decision-style questions* — the kind where you need more than a one-line answer. You ask something like *“Should we switch our fleet from diesel to electric?”* and the system:

- **Pulls in proof** from your documents and data (RAG — retrieval over PDFs, CSVs, reports).
- **Remembers similar questions** so you’re not burning tokens on repeat asks (CAG — cache).
- **Connects the dots** between concepts — costs, emissions, policies, trade-offs — using a knowledge graph (KAG).
- **Explains the call** in a structured way: evidence, trade-offs, confidence, and a clear recommendation with citations.

So it’s not just chat: it’s **retrieve → reason → decide**, with sources you can point to.

**Stack:** FastAPI + Python · React + Tailwind + shadcn/ui · Redis · Neo4j · FAISS

---

## Repo layout

| Folder | What |
|--------|------|
| `backend/` | FastAPI API, ingestion, embeddings, full pipeline (see `backend/README.md` for the deep dive) |
| `frontend/insight-navigator/` | React UI — query, pipeline trace, graph view, documents |
| `data/` | Optional place for raw files; when you run the API from `backend/`, it also uses `backend/data/` |
| `shared/` | Shared bits / room to grow |

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
Copy-Item .env.example .env   # then set OPENAI_API_KEY
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

Drop your files under **`backend/data/raw/`**, then run ingest + index + graph (steps in `backend/README.md`).

---

## Frontend (UI)

```text
cd frontend/insight-navigator
npm install
npm run dev
```

Open whatever URL Vite prints (often `http://localhost:5173` or `8080`).

---

## Security

Don’t commit **`backend/.env`** or real API keys — `.env.example` is the template only.

---

## License

This project is licensed under the **MIT License** — see [`LICENSE`](LICENSE).
