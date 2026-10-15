import os
import subprocess
import sys

os.environ["PDFQA_RUN_INTEGRATION"] = "1"
result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/test_integration_ollama.py", "-v", "--no-header"],
    env=os.environ,
)
sys.exit(result.returncode)
