import ast
import sys
from pathlib import Path

ok = True
for f in Path("src").glob("*.py"):
    tree = ast.parse(f.read_text(encoding="utf-8"))
    names = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        ok = False
        print(f"DUPES in {f}: {dupes}")
    else:
        print(f"clean {f} ({len(names)} top-level defs)")

sys.exit(0 if ok else 1)
