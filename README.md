"# Local PDF QA — Fully Local RAG over Your PDFs

Ask questions about 10–15 PDF documents and get answers **grounded exclusively
in those documents**, with citations to filename and page number. Everything
runs on your machine:

- **Ollama** runs the language model (`qwen3:8b`) and the embedding model
  (`embeddinggemma`) locally.
- **SQLite** (with FTS5) stores all text, metadata, and embeddings in a single
  file: `data/knowledge.db`.
- **NumPy** performs cosine similarity in-process — no vector database.
- **Streamlit** provides a browser UI served on `http://127.0.0.1:8501` only.

No cloud APIs, no external databases, no paid services, no telemetry, no API
keys, no Docker.

---

## 1. Architecture

```
                 ┌────────────────────────────────────────────┐
 PDF files ────► │ ingestion.py                               │
 (pdfs/, upload) │  pdf_processor.py  → page-wise text (PyMuPDF)
                 │  chunker.py        → ~1100-char chunks     │
                 │  embeddings.py     → Ollama /api/embed     │
                 └──────────────┬─────────────────────────────┘
                                ▼
                    data/knowledge.db (SQLite)
        documents | chunks (+ float32 embeddings) | chunks_fts (FTS5)
                                ▲
        ┌───────────────────────┴────────────────────────────┐
        │ retriever.py                                       │
        │  semantic leg: NumPy cosine over stored vectors    │
        │  keyword leg:  SQLite FTS5 / BM25                  │
        │  fusion: Reciprocal Rank Fusion + relevance gate   │
        └───────────────────────┬────────────────────────────┘
                                ▼
                   answer_generator.py
        evidence gate → abstain OR strict JSON from qwen3:8b
        citation validation against actually-retrieved chunks
                                ▼
                          app.py (Streamlit)
```

Key grounding properties:

1. **Pre-LLM evidence gate** — if no retrieved chunk clears the relevance
   thresholds, the app answers *"I could not find enough information in the
   provided PDFs."* **without calling the language model at all**.
2. **Citation validation** — every citation the model emits (structured JSON
   or inline `[file.pdf, page N]`) is checked against the retrieved evidence
   set. Citations that reference anything else are **dropped, never repaired
   or fabricated**.
3. **No chat history as evidence** — each question triggers fresh retrieval;
   previous turns are never sent to the model.
4. **Prompt-injection resistance** — excerpt text is labeled untrusted data
   in the system prompt.

> **Honest limitation:** these mechanisms strongly reduce hallucinations but
> cannot *mathematically guarantee* that a language model never emits an
> unsupported sentence. Always treat citations as pointers to verify, not as
> proofs. See §16.

## 2. Prerequisites

- **Python 3.11+** (tested on 3.13)
- **Ollama** ≥ 0.4 — <https://ollama.com/download> (Windows/macOS/Linux)
- ~8–16 GB RAM for `qwen3:8b` (see §14)
- An SSD is recommended for faster indexing, not required



## 3. Install Ollama

1. Download from <https://ollama.com/download> and install.
2. Confirm it runs:
   ```powershell
   # Windows PowerShell
   ollama --version
   ollama serve   # only needed if the desktop app isn't running
   ```
   ```bash
   # macOS / Linux
   ollama --version
   ollama serve   # or: systemctl --user start ollama
   ```

## 4. Download the models (internet needed once)

```powershell
# Windows PowerShell / macOS / Linux — identical commands
ollama pull qwen3:8b           # ~5 GB download, answer generation
ollama pull embeddinggemma     # ~300 MB download, embeddings
```

After these two commands complete, the application itself never needs the
internet again (see §15).

If `qwen3:8b` is too heavy for your machine, pull a smaller model and point
the app at it (see §14).

## 5. Python virtual environment

```powershell
# Windows PowerShell
cd "d:\Projects\pdf project"
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

```bash
# macOS / Linux
cd "/path/to/pdf project"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 6. Install dependencies

```powershell
# Windows PowerShell (venv activated)
pip install -r requirements.txt
```

```bash
# macOS / Linux
pip install -r requirements.txt
```

## 7. Add PDFs

Two supported ways:

1. **Drop files into `pdfs/`** — any PDF placed there is picked up by
   "Index files from pdfs/ folder" in the UI or by `python ingest.py`.
2. **Upload through the UI** — the sidebar uploader saves files into `pdfs/`
   on this machine only (they are never sent anywhere).

Originals are opened read-only; the app never modifies your PDFs.

## 8. Run ingestion

**Option A — from the UI:** click *Index files from pdfs/ folder* (or upload
files and click *Index uploaded files*). The sidebar shows a per-document
report: pages, chunks, embedding status, and warnings (e.g. scanned PDFs).

**Option B — from the command line:**

```powershell
python ingest.py                 # incremental: skips unchanged files
python ingest.py --rebuild       # wipe the index and re-ingest everything
python ingest.py --dir D:\other\pdfs   # index a different folder
```

Incremental behavior:

- Each file's SHA-256 is stored; unchanged files are skipped instantly.
- A changed file is re-extracted and its old chunks/FTS/embeddings replaced
  atomically.
- A byte-identical copy under a different filename is recorded as
  `duplicate` and not indexed twice.
- If Ollama was down during indexing, text + keyword search still work;
  running the index command again repairs embeddings automatically.


## 9. Start the web interface

```powershell
# Windows PowerShell (venv activated)
streamlit run app.py
```

```bash
# macOS / Linux
streamlit run app.py
```

The browser opens `http://127.0.0.1:8501`. The server binds to **127.0.0.1
only** (configured in `.streamlit/config.toml`) and Streamlit telemetry is
disabled there as well.

## 10. Rebuild or update the index

- **Update one document:** expand it in the sidebar → *Reindex*.
- **Rebuild everything:** sidebar → *Danger zone* → check the confirmation →
  *Rebuild entire index*. Or `python ingest.py --rebuild`.
- **Model switch:** if you change `PDFQA_EMBEDDING_MODEL`, reindex (vectors
  from different models are never mixed; old-model vectors are ignored and
  re-embedded on reindex).

## 11. Remove a PDF

Sidebar → expand the document → *Remove*. This deletes the document row, all
its chunks, its embeddings, and its FTS entries in one transaction — no
orphans, no stale search hits (covered by tests).

## 12. Troubleshooting

| Symptom | Cause & fix |
|---|---|
| `Cannot reach Ollama at http://127.0.0.1:11434` | Ollama isn't running. Start the desktop app or `ollama serve`. |
| `Model 'qwen3:8b' is not available` | `ollama pull qwen3:8b` (same for `embeddinggemma`). |
| Answers always abstain | Thresholds too strict for your model. See §17 calibration; lower `PDFQA_MIN_COSINE` (e.g. 0.20) and retest. |
| Very slow answers | First question loads the model into RAM (cold start). Subsequent questions are faster. On CPU-only machines 8B models can take minutes per answer — consider a smaller model (§14). |
| `PDF is encrypted/password-protected` | Decrypt locally first: `qpdf --password='' --decrypt in.pdf out.pdf`, then re-add. |
| PDF indexed as `no_text` | It's a scan. Run local OCR (§13) and reindex the OCR'd copy. |
| `streamlit` command not found | Activate the virtual environment (§5). |
| PowerShell blocks `.venv\Scripts\Activate.ps1` | Run `Set-ExecutionPolicy -Scope Process RemoteSigned`, or call `.\.venv\Scripts\python.exe -m streamlit run app.py` directly. |
| Port 8501 already in use | `streamlit run app.py --server.port 8502`. |
| `Database is locked` | Another ingestion is running; wait for it (busy timeout is 30 s). |

## 13. Scanned PDFs & optional local OCR

Pages with images but no extractable text are flagged in the ingestion
report with the status `no_text` and an OCR hint. OCR is **optional** and
runs entirely offline via [Tesseract](https://github.com/tesseract-ocr/tesseract):

```powershell
# Windows (install Tesseract e.g. via winget, then:)
pip install ocrmypdf
ocrmypdf --skip-text input.pdf output.pdf
```

```bash
# macOS / Linux
brew install tesseract   # or: apt install tesseract-ocr
pip install ocrmypdf
ocrmypdf --skip-text input.pdf output.pdf
```

Place `output.pdf` in `pdfs/` and reindex. `ocrmypdf` is intentionally not in
`requirements.txt` — it is an optional extra needing a system binary.

## 14. Hardware & performance

Approximate, not guarantees — actual speed depends on CPU/GPU, RAM speed, and
PDF size:

| Component | `qwen3:8b` | Smaller alternative |
|---|---|---|
| RAM (model resident) | ~6–8 GB + overhead → **16 GB comfortable, 8 GB tight** | `qwen3:4b` (~4 GB), `llama3.2:3b` (~3 GB) |
| Disk | ~5 GB per 8B model | ~2–3 GB |
| Ingestion | dominated by embedding calls to local Ollama | same |
| Answer latency (CPU only) | 30 s – several min first question | a few seconds to ~1 min |

To switch models without editing code:

```powershell
# Windows PowerShell
$env:PDFQA_CHAT_MODEL="qwen3:4b"
ollama pull qwen3:4b
streamlit run app.py
```

```bash
# macOS / Linux
export PDFQA_CHAT_MODEL="qwen3:4b"
ollama pull qwen3:4b
streamlit run app.py
```

All other settings (chunk size, overlap, thresholds, timeouts, paths) are
overridable the same way — see `config.py` for the full list.


## 15. Privacy & offline operation

**What never leaves your machine:** PDFs, questions, answers, embeddings, the
SQLite database, chat history. The application code makes exactly one kind of
network connection — HTTP to `OLLAMA_HOST` (default
`http://127.0.0.1:11434`). There are no cloud APIs, no analytics, no
telemetry (also disabled in `.streamlit/config.toml`), no crash reporters,
and no API keys. The Streamlit server binds to loopback, not your LAN.

**What needs the internet once:** `pip install -r requirements.txt` and
`ollama pull` of the two models. After those succeed, you can disconnect and
verify offline operation yourself:

1. Turn off Wi-Fi / unplug Ethernet.
2. `streamlit run app.py` → index (already done) → ask questions.
3. Everything above must work; if any component attempted an external
   connection it would visibly fail.

This README deliberately does **not** claim "100% offline verified" — that
verification is yours to perform on your machine after setup, per the
distinction above.

## 16. Limitations of RAG & citation reliability

- **No mathematical hallucination guarantee.** The evidence gate, strict
  prompts, and citation validation make unsupported answers *unlikely*, but a
  language model can still occasionally produce a plausible-sounding claim
  that isn't in the excerpts. Treat citations as "check here", not proof.
- **Citations are validated, not generated** — the app drops any citation the
  model invents, but it cannot verify that a *retained* citation's sentence
  is a faithful paraphrase of the cited page.
- **Retrieval is the ceiling.** If the relevant passage isn't retrieved
  (threshold too high, OCR missing, table flattened awkwardly), the model
  never sees it. Inspect the *Evidence used* expander for every answer.
- **PDF text extraction quality matters.** Complex multi-column layouts,
  footnotes, and tables can extract imperfectly. Chunking is per-page, so a
  section split across a page boundary becomes two chunks with correct page
  numbers rather than one wrongly-attributed chunk.
- **BM25/cosine thresholds are model-specific.** Defaults
  (`MIN_COSINE_SIMILARITY=0.30`, `MIN_BM25_SCORE=6.0`) are starting points —
  see §17.

## 17. Calibrating the relevance threshold

Cosine distributions differ per embedding model, so calibrate instead of
trusting a universal number:

1. Index your corpus; ask 10–15 questions you *know* the corpus answers.
2. For each, read the "Retrieval quality: best cosine X, best bm25 Y" caption
   under the answer.
3. Note the **lowest** score among successful answers and the **highest**
   score among abstentions/wrong retrievals.
4. Set `PDFQA_MIN_COSINE` between those two values (bias slightly toward the
   abstention side — abstaining is safer than hallucinating). Do the same for
   `PDFQA_MIN_BM25` with the bm25 numbers.
5. Re-ask; verify good questions still answer and absent-info questions
   abstain.


## 18. Running the tests

Unit tests use temporary databases, a deterministic offline fake embedder,
and mocked Ollama responses — **no models, no network, no real LLM**:

```powershell
# Windows PowerShell
python -m pytest tests -q --ignore=tests/test_integration_ollama.py
```

```bash
# macOS / Linux
python -m pytest tests -q --ignore=tests/test_integration_ollama.py
```

Optional integration tests hit your local Ollama with the real models:

```powershell
# Windows PowerShell
$env:PDFQA_RUN_INTEGRATION="1"
python -m pytest tests/test_integration_ollama.py -v
```

```bash
# macOS / Linux
PDFQA_RUN_INTEGRATION=1 python -m pytest tests/test_integration_ollama.py -v
```

They are skipped automatically when Ollama or the models are unavailable.

**Test corpus** (`tests/conftest.py`): a six-document synthetic corpus
covering — a question explicitly answered by one PDF, a question needing two
PDFs, an absent-information question, paraphrased terminology, conflicting
contracts in two PDFs, and a scanned PDF with no extractable text. Expected
behavior for unsupported questions is **abstention**, never a
general-knowledge answer.

## 19. Configuration reference

All settings live in `config.py`; each has an environment-variable override:
chat model (`PDFQA_CHAT_MODEL`), embedding model (`PDFQA_EMBEDDING_MODEL`),
Ollama host (`OLLAMA_HOST`), chunk size (`PDFQA_CHUNK_SIZE`, default 1100
chars), overlap (`PDFQA_CHUNK_OVERLAP`, default 170), embedding batch size
(`PDFQA_EMBED_BATCH_SIZE`, 32), semantic/keyword candidate counts (30/30),
top-k (`PDFQA_TOP_K`, 6), RRF k (60), `PDFQA_MIN_COSINE` (0.30),
`PDFQA_MIN_BM25` (6.0), temperature (`PDFQA_TEMPERATURE`, 0.0), timeouts,
database path (`PDFQA_DB_PATH`), PDF dir (`PDFQA_PDF_DIR`).


## 20. Project layout

```
local-pdf-qa/
├── app.py                  # Streamlit UI (localhost only)
├── config.py               # all settings + env overrides
├── ingest.py               # CLI ingestion / rebuild
├── requirements.txt
├── .streamlit/config.toml  # bind 127.0.0.1, telemetry off
├── pdfs/                   # your PDFs live here (gitignored)
├── data/                   # knowledge.db lives here (gitignored)
├── src/
│   ├── ollama_client.py    # local Ollama HTTP client + friendly errors
│   ├── database.py         # SQLite schema, FTS5 sync triggers, CRUD
│   ├── pdf_processor.py    # PyMuPDF extraction, scanned/encrypted handling
│   ├── chunker.py          # sentence-aware overlapping chunks
│   ├── embeddings.py       # batched, validated, normalized vectors
│   ├── retriever.py        # hybrid cosine+BM25 with RRF fusion
│   ├── answer_generator.py # evidence gate, JSON validation, citations
│   └── ingestion.py        # incremental pipeline + reports
└── tests/
    ├── conftest.py         # FakeEmbedder + synthetic corpus fixtures
    ├── test_chunker.py
    ├── test_database.py
    ├── test_retriever.py
    ├── test_grounding.py
    ├── test_embeddings.py
    ├── test_ingestion.py
    └── test_integration_ollama.py   # optional, real models
```

### Design decisions (where the spec allowed choice)

- **Embeddings as float32 BLOBs** (not JSON): ~4× smaller and zero-copy into
  NumPy via `np.frombuffer`.
- **FTS5 as an external-content table** with sync triggers: text is stored
  once; deletes/updates can never leave stale search entries.
- **Per-page chunking**: guarantees exact page attribution for citations; a
  section crossing a page boundary becomes two chunks, both correctly
  numbered.
- **RRF over score blending**: BM25 and cosine aren't comparable; rank fusion
  sidesteps scale calibration entirely. Absolute thresholds then act as the
  per-model-calibrated evidence gate.
- **embeddinggemma task prefixes** (`task: search result\nquery: ` /
  `task: search result\ndocument: `) applied consistently to documents and
  queries — required for this model family's retrieval quality.
