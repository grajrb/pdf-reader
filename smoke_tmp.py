"""Live smoke test: real embeddinggemma via local Ollama."""
import sys

sys.path.insert(0, ".")

import numpy as np

from src.embeddings import Embedder
from src.ollama_client import OllamaClient

client = OllamaClient()
print("models:", client.list_models())
e = Embedder(client)
e.ensure_available()
v = e.embed_documents(["hello world", "protein target grams"])
q = e.embed_query("hello world")
print("doc matrix shape:", v.shape, "| query shape:", q.shape)
print("doc norm:", float(np.linalg.norm(v[0])), "| query norm:", float(np.linalg.norm(q)))
print("self-cosine:", float(v[0] @ q))
