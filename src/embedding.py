"""Ollama over HTTP. Stdlib only, so the dependency list stays small."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from .config import (
    CHAT_MODEL,
    KEEP_ALIVE,
    EMBED_MODEL,
    NUM_CTX,
    OLLAMA_URL,
    REQUEST_TIMEOUT,
    TEMPERATURE,
)


class OllamaError(RuntimeError):
    pass


class Ollama:
    def __init__(self, url: str = OLLAMA_URL, embed_model: str = EMBED_MODEL, chat_model: str = CHAT_MODEL):
        self.url = url.rstrip("/")
        self.embed_model = embed_model
        self.chat_model = chat_model

    # ------------------------------------------------------------ transport
    def _post(self, path: str, payload: dict, timeout: int = REQUEST_TIMEOUT) -> dict:
        body = json.dumps(payload).encode()
        request = urllib.request.Request(
            f"{self.url}{path}", data=body, headers={"Content-Type": "application/json"}
        )
        last: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    return json.loads(response.read().decode())
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last = exc
                time.sleep(1.5 * (attempt + 1))
        raise OllamaError(f"POST {path} failed after 3 attempts: {last}")

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(f"{self.url}{path}", timeout=30) as response:
            return json.loads(response.read().decode())

    # ------------------------------------------------------------ operations
    def embed(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            data = self._post(
                "/api/embed",
                {"model": self.embed_model, "input": batch, "keep_alive": KEEP_ALIVE},
            )
            embeddings = data.get("embeddings")
            if not embeddings or len(embeddings) != len(batch):
                raise OllamaError(f"bad embedding response for batch at {start}")
            vectors.extend(embeddings)
        return vectors

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    # nomic-embed-text is trained with task prefixes. Without them the model
    # mixes document and query semantics in one space.
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed([f"search_document: {text}" for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.embed_one(f"search_query: {text}")

    def chat(self, prompt: str, system: str | None = None, temperature: float = TEMPERATURE) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.chat_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "seed": 42, "num_ctx": NUM_CTX, "num_predict": 160},
        }
        payload["keep_alive"] = KEEP_ALIVE
        data = self._post("/api/chat", payload)
        message = data.get("message") or {}
        return (message.get("content") or "").strip()

    def health(self) -> dict:
        try:
            tags = self._get("/api/tags")
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            return {"reachable": False, "error": str(exc)}
        names = {model["name"] for model in tags.get("models", [])}
        bases = {name.split(":")[0] for name in names}

        def present(model: str) -> bool:
            return model in names or model.split(":")[0] in bases

        return {
            "reachable": True,
            "embed_model": self.embed_model,
            "chat_model": self.chat_model,
            "embed_model_present": present(self.embed_model),
            "chat_model_present": present(self.chat_model),
        }
