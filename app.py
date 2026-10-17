"""Streamlit user interface for the local PDF question-answering app.

- Runs fully locally; the server binds to 127.0.0.1 (see .streamlit/config.toml).
- Uploaded documents and questions never leave this machine: the only network
  calls the backend makes are to the local Ollama server.
- Chat history is kept for display only; it is NEVER sent to the model as
  evidence (each answer uses freshly retrieved document excerpts).
"""
from __future__ import annotations

import time

import streamlit as st

import config
from src.answer_generator import Answer, AnswerGenerator
from src.database import Database
from src.embeddings import Embedder, EmbeddingError
from src.ingestion import IngestionService
from src.ollama_client import OllamaClient, OllamaError
from src.retriever import HybridRetriever

st.set_page_config(
    page_title=config.APP_TITLE, page_icon=config.APP_ICON, layout="wide"
)


# ----------------------------------------------------------------- helpers
@st.cache_resource(show_spinner=False)
def get_db() -> Database:
    db = Database(config.DB_PATH)
    db.initialize()
    return db


@st.cache_resource(show_spinner=False)
def get_embedder(model: str) -> Embedder:
    return Embedder(client=OllamaClient(), model=model)


def build_service(db: Database) -> IngestionService:
    return IngestionService(db, embedder=get_embedder(config.EMBEDDING_MODEL))


def build_generator(db: Database, top_k: int, min_cosine: float) -> AnswerGenerator:
    retriever = HybridRetriever(
        db,
        get_embedder(config.EMBEDDING_MODEL),
        top_k=top_k,
        min_cosine=min_cosine,
    )
    return AnswerGenerator(retriever, client=OllamaClient())


def ollama_status(client: OllamaClient) -> tuple[bool, bool, str]:
    """Return (chat_model_ok, embed_model_ok, status_text)."""
    try:
        reachable, message = client.ping()
    except OllamaError as exc:
        return False, False, str(exc)
    if not reachable:
        return False, False, message
    try:
        chat_ok = client.has_model(config.CHAT_MODEL)
        embed_ok = client.has_model(config.EMBEDDING_MODEL)
    except OllamaError as exc:
        return False, False, str(exc)
    return chat_ok, embed_ok, message


def render_answer(answer: Answer) -> None:
    """Render one assistant answer with citations and evidence."""
    if answer.insufficient_evidence:
        st.warning(config.ABSTENTION_MESSAGE + "\n\n"
                   "_No language-model call was made because no retrieved "
                   "excerpt cleared the relevance threshold"
                   + ("." if not answer.evidence else
                      " (see evidence below)."))
    else:
        st.markdown(answer.text)

    if answer.citations:
        chips = " · ".join(
            f"`{c.filename}` **p.{c.page}**" for c in answer.citations
        )
        st.markdown(f"**Sources:** {chips}")
    elif not answer.insufficient_evidence and answer.structured:
        st.caption("⚠ The model provided no citations for this answer.")

    for warning in answer.warnings:
        st.caption(f"⚠ {warning}")

    if answer.evidence:
        label = f"Evidence used ({len(answer.evidence)} excerpts)"
        with st.expander(label):
            for i, chunk in enumerate(answer.evidence, start=1):
                scores = []
                if chunk.semantic_score is not None:
                    scores.append(f"cosine {chunk.semantic_score:.2f}")
                if chunk.keyword_score is not None:
                    scores.append(f"bm25 {chunk.keyword_score:.2f}")
                score_text = f" — {', '.join(scores)}" if scores else ""
                st.markdown(
                    f"**[E{i}] {chunk.filename}, page {chunk.page_number}**"
                    f"{score_text}"
                )
                st.text(chunk.text[:2000])
                st.divider()

    if answer.retrieval is not None:
        r = answer.retrieval
        parts = []
        if r.best_semantic is not None:
            parts.append(f"best cosine {r.best_semantic:.2f}")
        if r.best_keyword is not None:
            parts.append(f"best bm25 {r.best_keyword:.2f}")
        if parts:
            st.caption(
                "Retrieval quality: " + ", ".join(parts)
                + f" (thresholds: cosine ≥ {config.MIN_COSINE_SIMILARITY:.2f}"
                + f" or bm25 ≥ {config.MIN_BM25_SCORE:.1f})"
            )



# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.title(f"{config.APP_ICON} {config.APP_TITLE}")
    st.caption("Fully local RAG: your PDFs, your Ollama, your SQLite.")

    client = OllamaClient()
    chat_ok, embed_ok, status_text = ollama_status(client)
    if chat_ok and embed_ok:
        st.success(status_text)
    else:
        st.error(status_text)
        st.code(
            f"ollama pull {config.CHAT_MODEL}\nollama pull {config.EMBEDDING_MODEL}",
            language="bash",
        )
    col1, col2 = st.columns(2)
    col1.markdown(f"**Chat model:** `{config.CHAT_MODEL}`"
                  + (" ✓" if chat_ok else " ✗"))
    col2.markdown(f"**Embedding model:** `{config.EMBEDDING_MODEL}`"
                  + (" ✓" if embed_ok else " ✗"))

    st.divider()
    st.subheader("Add PDFs")
    db = get_db()
    service = build_service(db)
    uploads = st.file_uploader(
        "Upload PDF files (stored only on this machine)",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if st.button("Index uploaded files", type="primary", disabled=not uploads):
        config.PDF_DIR.mkdir(parents=True, exist_ok=True)
        saved = []
        for upload in uploads:
            # Keep only the basename; never trust client-supplied paths.
            target = config.PDF_DIR / upload.name.split("/")[-1].split("\\")[-1]
            target.write_bytes(upload.getbuffer())
            saved.append(target)
        bar = st.progress(0.0, "Indexing uploads…")
        for i, target in enumerate(saved, start=1):
            entry = service.ingest_file(target)
            bar.progress(i / len(saved), f"{entry.filename}: {entry.status}")
            if entry.status in ("error", "encrypted", "no_text"):
                st.warning(f"**{entry.filename}** — {entry.status}: {entry.message}")
            elif entry.status == "duplicate":
                st.info(f"**{entry.filename}** — {entry.message}")
            else:
                st.success(
                    f"**{entry.filename}** — {entry.status} "
                    f"({entry.pages} pages, {entry.chunks} chunks"
                    + (", embedded" if entry.embedded else ", NOT embedded") + ")"
                )
        bar.empty()

    if st.button(f"Index files from {config.PDF_DIR.name}/ folder"):
        with st.spinner(f"Indexing {config.PDF_DIR.name}/ directory…"):
            report = service.ingest_directory()
        st.success(f"Indexed/updated {report.indexed} document(s).")
        for entry in report.entries:
            line = f"**{entry.filename}** — {entry.status}"
            if entry.chunks:
                line += f" ({entry.chunks} chunks)"
            st.caption(line)
            if entry.message and entry.status != "unchanged":
                st.caption(f"  ↳ {entry.message}")

    st.divider()
    st.subheader("Indexed documents")
    documents = db.list_documents()
    if not documents:
        st.caption("No documents indexed yet. Upload PDFs or add files to "
                   f"the {config.PDF_DIR.name}/ folder.")
    for doc in documents:
        emb = "✓" if doc["embedding_status"] == "ok" else "⚠"
        with st.expander(f"{emb} {doc['filename']} — {doc['page_count']}p, "
                         f"{doc['chunk_count']} chunks"):
            st.caption(f"Status: {doc['status']} · embeddings: "
                       f"{doc['embedding_status']} · added: {doc['ingested_at']}")
            if doc["message"]:
                st.caption(doc["message"])
            c1, c2 = st.columns(2)
            if c1.button("Reindex", key=f"reindex_{doc['id']}"):
                with st.spinner(f"Reindexing {doc['filename']}…"):
                    entry = service.ingest_file(
                        config.PDF_DIR / doc["filename"], force=True
                    )
                st.rerun()
            if c2.button("Remove", key=f"remove_{doc['id']}"):
                service.remove_document(doc["filename"])
                st.rerun()

    with st.popover("Danger zone"):
        st.caption(f"Rebuild deletes the whole index and re-ingests "
                   f"{config.PDF_DIR.name}/.")
        if st.checkbox("I understand"):
            if st.button("Rebuild entire index", type="primary"):
                with st.spinner("Rebuilding…"):
                    report = service.rebuild()
                st.success(f"Rebuilt: {report.indexed} document(s) indexed.")

    st.divider()
    st.subheader("Search settings")
    top_k = st.slider("Retrieved chunks (top-k)", 2, 15, config.DEFAULT_TOP_K)
    min_cosine = st.slider(
        "Cosine relevance threshold", 0.0, 0.9,
        float(config.MIN_COSINE_SIMILARITY), 0.05,
        help="Chunks below this cosine similarity (and below the BM25 floor) "
             "are treated as no evidence. Calibrate with known questions.",
    )
    doc_names = [d["filename"] for d in documents]
    selected = st.multiselect(
        "Limit search to documents (empty = all)",
        doc_names, default=[],
    )
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()



# ------------------------------------------------------------ chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.header("Ask your documents")
st.caption(
    "Answers use only excerpts retrieved from the indexed PDFs. "
    "Unsupported questions get an explicit abstention."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        elif message.get("error"):
            st.error(message["content"])
        else:
            render_answer(message["answer"])

question = st.chat_input("Ask a question about your PDFs…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    doc_ids = None
    if selected:
        name_to_id = {d["filename"]: d["id"] for d in documents}
        doc_ids = [name_to_id[n] for n in selected if n in name_to_id]

    generator = build_generator(db, top_k=top_k, min_cosine=min_cosine)
    started = time.perf_counter()
    try:
        with st.chat_message("assistant"):
            with st.status("Working…", expanded=False) as status:
                answer = generator.ask(
                    question,
                    doc_ids=doc_ids,
                    on_stage=lambda s: status.update(label=s),
                )
                elapsed = time.perf_counter() - started
                status.update(label=f"Done in {elapsed:.1f}s", state="complete")
            render_answer(answer)
            st.caption(
                f"Answered with `{config.CHAT_MODEL}` on this machine "
                f"in {elapsed:.1f}s."
            )
        st.session_state.messages.append(
            {"role": "assistant", "content": question, "answer": answer}
        )
    except (OllamaError, EmbeddingError) as exc:
        friendly = str(exc)
        with st.chat_message("assistant"):
            st.error(friendly)
        st.session_state.messages.append(
            {"role": "assistant", "content": friendly, "error": True}
        )
    except Exception as exc:  # never show a raw traceback to the user
        friendly = (
            f"Unexpected error while answering: "
            f"{exc.__class__.__name__}: {exc}"
        )
        with st.chat_message("assistant"):
            st.error(friendly + " — see the terminal running Streamlit for details.")
        st.session_state.messages.append(
            {"role": "assistant", "content": friendly, "error": True}
        )
