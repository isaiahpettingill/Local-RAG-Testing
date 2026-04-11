from __future__ import annotations

import os
from typing import Any, cast

from llama_cpp import Llama, ChatCompletionRequestMessage
from openai import OpenAI

from src.models.config import MODELS


class ChatLlamaCpp:
    _instances: dict[str, Llama] = {}
    _openai_clients: dict[str, OpenAI] = {}

    def __init__(self, role: str) -> None:
        self.role = role
        self.config = MODELS[role]

    def _provider(self) -> str:
        return str(self.config.get("provider", "local"))

    def _get_instance(self) -> Llama:
        if self._provider() == "openai":
            raise RuntimeError("Remote providers do not use local llama instances")
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

    def _get_openai_client(self) -> OpenAI:
        if self.role not in self._openai_clients:
            base_url = self.config.get("base_url") or os.environ.get("OPENAI_BASE_URL")
            api_key = self.config.get("api_key") or os.environ.get("OPENAI_API_KEY")
            if not base_url or not api_key:
                raise RuntimeError("OpenAI provider requires OPENAI_BASE_URL and OPENAI_API_KEY")
            self._openai_clients[self.role] = OpenAI(base_url=base_url, api_key=api_key)
        return self._openai_clients[self.role]

    def _model_name(self) -> str:
        return str(self.config.get("model_id") or self.config["name"])

    def chat(self, messages: list[dict[str, str]]) -> str:
        if self._provider() == "openai":
            client = self._get_openai_client()
            response = client.chat.completions.create(
                model=self._model_name(),
                messages=cast(list[dict[str, str]], messages),
                temperature=0,
            )
            result = response.choices[0].message.content
            return result if result else ""

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
        if self._provider() == "openai":
            client = self._get_openai_client()
            response = client.chat.completions.create(
                model=self._model_name(),
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            result = response.choices[0].message.content
            return result if result else ""

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
    if model._provider() == "openai":
        client = model._get_openai_client()
        response = client.embeddings.create(model=model._model_name(), input=text)
        return list(response.data[0].embedding)

    result = model.completion(f"Embed: {text}")
    import json

    try:
        return json.loads(result)
    except json.JSONDecodeError:
        return [0.0] * 384


def embed_documents(texts: list[str]) -> list[list[float]]:
    return [embed_query(t) for t in texts]
