import py_compile
import sys

files = [
    "config.py",
    "src/__init__.py",
    "src/ollama_client.py",
    "src/database.py",
    "src/pdf_processor.py",
    "src/chunker.py",
    "src/embeddings.py",
    "src/retriever.py",
    "src/answer_generator.py",
    "src/ingestion.py",
]
ok = True
for f in files:
    try:
        py_compile.compile(f, doraise=True)
        print("OK ", f)
    except Exception as e:
        ok = False
        print("FAIL", f, e)
sys.exit(0 if ok else 1)
