 import sys

sys.path.insert(0, ".")

from src.ollama_client import OllamaClient

client = OllamaClient()
print("warming qwen3:8b …")
print(client.chat("qwen3:8b", [{"role": "user", "content": "Say OK."}]))
