import importlib
import sys

modules = [
    "config",
    "src.ollama_client",
    "src.database",
    "src.pdf_processor",
    "src.chunker",
    "src.embeddings",
    "src.retriever",
    "src.answer_generator",
    "src.ingestion",
]
ok = True
for m in modules:
    try:
        importlib.import_module(m)
        print("OK ", m)
    except Exception as e:
        ok = False
        print("FAIL", m, "->", type(e).__name__, e)
sys.exit(0 if ok else 1)
