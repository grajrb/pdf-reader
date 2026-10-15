import importlib.metadata as md
import sqlite3
import sys

print("python", sys.version)
for pkg in ["streamlit", "pymupdf", "numpy", "pytest", "requests", "httpx"]:
    try:
        print(pkg, md.version(pkg))
    except Exception:
        print(pkg, "NOT INSTALLED")

c = sqlite3.connect(":memory:")
try:
    c.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
    print("FTS5 OK")
except Exception as e:
    print("FTS5 FAIL:", e)
print("sqlite", sqlite3.sqlite_version)
