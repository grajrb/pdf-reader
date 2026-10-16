"""Minimal HTTP client for a *local* Ollama server.

This module only ever talks to the host configured in ``config.OLLAMA_HOST``
(default ``http://127.0.0.1:11434``). It performs no cloud calls, needs no
API keys, and raises friendly, actionable errors when Ollama or a model is
unavailable.
"""
from __future__ import annotations

import ipaddress
from typing import Any, Sequence
from urllib.parse import urlsplit

import requests

import config


class OllamaError(RuntimeError):
    """Base class for Ollama-related failures with a user-friendly message."""


class OllamaUnavailable(OllamaError):
    """Ollama server cannot be reached."""


class OllamaModelMissing(OllamaError):
    """The requested model is not installed in Ollama."""


class OllamaResponseError(OllamaError):
    """Ollama responded with an error or an unexpected payload."""


class OllamaConfigurationError(OllamaError):
    """Ollama is configured with a non-local or malformed server URL."""


class OllamaClient:
    """Small synchronous client for Ollama's /api endpoints."""

    def __init__(
        self,
        host: str | None = None,
        connect_timeout: float | None = None,
    ) -> None:
        self.host = (host or config.OLLAMA_HOST).rstrip("/")
        self.connect_timeout = (
            connect_timeout
            if connect_timeout is not None
            else config.OLLAMA_CONNECT_TIMEOUT
        )

    # ------------------------------------------------------------- helpers
    def _url(self, path: str) -> str:
        try:
            parsed = urlsplit(self.host)
            hostname = parsed.hostname
            is_loopback_ip = (
                ipaddress.ip_address(hostname).is_loopback
                if hostname is not None
                else False
            )
        except ValueError:
            raise OllamaConfigurationError(
                "Ollama must use a valid localhost URL; remote Ollama hosts "
                "are disabled to keep all document and question traffic on "
                "this machine."
            ) from None

        is_localhost = (
            hostname is not None
            and hostname.rstrip(".").lower() == "localhost"
        )
        if (
            parsed.scheme not in ("http", "https")
            or not (is_localhost or is_loopback_ip)
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
        ):
            raise OllamaConfigurationError(
                "Ollama must use a localhost URL (127.0.0.1, ::1, or "
                "localhost). Remote Ollama hosts are disabled to keep all "
                "document and question traffic on this machine."
            )
        return f"{self.host}{path}"

    @staticmethod
    def _error_payload(response: requests.Response) -> str:
        try:
            data = response.json()
            if isinstance(data, dict) and data.get("error"):
                return str(data["error"])
        except ValueError:
            pass
        return (response.text or "").strip()[:300]

    @staticmethod
    def _looks_like_missing_model(message: str) -> bool:
        lowered = message.lower()
        return any(
            needle in lowered
            for needle in ("not found", "not installed", "no such model")
        )

    def _post(
        self,
        path: str,
        payload: dict[str, Any],
        timeout: float,
        model: str,
    ) -> dict[str, Any]:
        try:
            response = requests.post(
                self._url(path), json=payload, timeout=timeout
            )
        except requests.Timeout as exc:
            raise OllamaError(
                f"Ollama at {self.host} timed out after {timeout:.0f}s "
                f"while processing '{model}'. The model may still be loading; "
                "try again in a moment or increase the timeout in config.py."
            ) from exc
        except requests.RequestException as exc:
            raise OllamaUnavailable(
                f"Cannot reach Ollama at {self.host} ({exc.__class__.__name__}). "
                "Make sure the Ollama desktop app or `ollama serve` is running."
            ) from exc

        if response.status_code >= 400:
            message = self._error_payload(response)
            if (
                response.status_code == 404
                or self._looks_like_missing_model(message)
            ):
                raise OllamaModelMissing(
                    f"Model '{model}' is not available in Ollama. "
                    f"Download it with:  ollama pull {model}"
                )
            raise OllamaResponseError(
                f"Ollama returned HTTP {response.status_code} for '{model}': "
                f"{message or 'no details'}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaResponseError(
                f"Ollama returned a non-JSON response for '{model}'."
            ) from exc
        if not isinstance(data, dict):
            raise OllamaResponseError(
                f"Ollama returned an unexpected payload for '{model}'."
            )
        return data

    # -------------------------------------------------------------- public
    def ping(self) -> tuple[bool, str]:
        """Return (reachable, human-readable status message)."""
        try:
            models = self.list_models()
        except OllamaError as exc:
            return False, str(exc)
        return True, f"Connected to Ollama at {self.host} ({len(models)} models)"

    def list_models(self) -> list[str]:
        """List installed model names, e.g. ['qwen3:8b', 'embeddinggemma:latest']."""
        try:
            response = requests.get(
                self._url("/api/tags"), timeout=self.connect_timeout
            )
        except requests.Timeout as exc:
            raise OllamaUnavailable(
                f"Ollama at {self.host} did not respond within "
                f"{self.connect_timeout:.0f}s."
            ) from exc
        except requests.RequestException as exc:
            raise OllamaUnavailable(
                f"Cannot reach Ollama at {self.host} ({exc.__class__.__name__}). "
                "Make sure the Ollama desktop app or `ollama serve` is running."
            ) from exc
        if response.status_code >= 400:
            raise OllamaResponseError(
                f"Ollama /api/tags returned HTTP {response.status_code}."
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaResponseError("Ollama /api/tags returned non-JSON.") from exc
        return [
            str(m.get("name", ""))
            for m in data.get("models", [])
            if isinstance(m, dict)
        ]

    def has_model(self, name: str) -> bool:
        """Check whether a model (with or without tag) is installed."""
        installed = self.list_models()
        wanted = name if ":" in name else f"{name}:latest"
        for model in installed:
            if model == wanted or model == name:
                return True
            # bare name matches any tag, e.g. 'qwen3' matches 'qwen3:8b'
            if ":" not in name and model.split(":")[0] == name:
                return True
        return False

    def embed(
        self,
        model: str,
        inputs: Sequence[str],
        timeout: float | None = None,
    ) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per input."""
        data = self._post(
            "/api/embed",
            {"model": model, "input": list(inputs)},
            timeout if timeout is not None else config.EMBEDDING_TIMEOUT,
            model,
        )
        vectors = data.get("embeddings")
        if vectors is None and "embedding" in data:  # single-input fallback
            vectors = [data["embedding"]]
        if not isinstance(vectors, list) or len(vectors) != len(inputs):
            count = len(vectors) if isinstance(vectors, list) else 0
            raise OllamaResponseError(
                f"Ollama returned {count} embeddings for {len(inputs)} "
                f"inputs using '{model}'."
            )
        return vectors

    def chat(
        self,
        model: str,
        messages: Sequence[dict[str, str]],
        temperature: float | None = None,
        timeout: float | None = None,
    ) -> str:
        """Run one non-streaming chat completion; returns the assistant text."""
        data = self._post(
            "/api/chat",
            {
                "model": model,
                "messages": list(messages),
                "stream": False,
                "options": {
                    "temperature": (
                        config.GENERATION_TEMPERATURE
                        if temperature is None
                        else temperature
                    )
                },
            },
            timeout if timeout is not None else config.CHAT_TIMEOUT,
            model,
        )
        message = data.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise OllamaResponseError(
                f"Ollama returned an empty assistant message for '{model}'."
            )
        return content
