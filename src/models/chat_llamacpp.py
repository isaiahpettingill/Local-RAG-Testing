from __future__ import annotations

from typing import Any, cast

from llama_cpp import Llama, ChatCompletionRequestMessage

from src.models.config import MODELS


class ChatLlamaCpp:
    _instances: dict[str, Llama] = {}

    def __init__(self, role: str) -> None:
        self.role = role
        self.config = MODELS[role]

    def _get_instance(self) -> Llama:
        if self.role not in self._instances:
            kwargs: dict[str, Any] = {
                "model_path": self.config["path"],
                "n_ctx": self.config["n_ctx"],
                "n_gpu_layers": self.config["n_gpu_layers"],
                "n_batch": self.config["n_batch"],
            }
            if self.config["n_threads"] is not None:
                kwargs["n_threads"] = self.config["n_threads"]
            self._instances[self.role] = Llama(**kwargs)
        return self._instances[self.role]

    def chat(self, messages: list[dict[str, str]]) -> str:
        llm = self._get_instance()
        formatted: list[ChatCompletionRequestMessage] = cast(
            list[ChatCompletionRequestMessage],
            [{"role": "user", "content": m["content"]} for m in messages],
        )
        # stream=False returns the non-streaming response type
        response = llm.create_chat_completion(messages=formatted, stream=False)
        result = response["choices"][0]["message"]["content"]  # type: ignore
        return result if result else ""

    def completion(self, prompt: str) -> str:
        llm = self._get_instance()
        response = llm(prompt, stream=False)
        result = response["choices"][0]["text"]  # type: ignore
        return result if result else ""


_embedding_model: ChatLlamaCpp | None = None


def get_embedding_model() -> ChatLlamaCpp:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = ChatLlamaCpp("embedding")
    return _embedding_model


def embed_query(text: str) -> list[float]:
    model = get_embedding_model()
    result = model.completion(f"Embed: {text}")
    import json

    try:
        return json.loads(result)
    except json.JSONDecodeError:
        return [0.0] * 384


def embed_documents(texts: list[str]) -> list[list[float]]:
    return [embed_query(t) for t in texts]
