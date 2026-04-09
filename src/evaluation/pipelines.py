from __future__ import annotations

from src.models.chat_llamacpp import ChatLlamaCpp


def run_pipeline_a(prompt: str) -> str:
    model = ChatLlamaCpp("agent")
    return model.completion(prompt)


def run_pipeline_b(prompt: str) -> str:
    model = ChatLlamaCpp("agent")
    return model.completion(prompt)


def run_pipeline_c(prompt: str) -> str:
    model = ChatLlamaCpp("agent")
    return model.completion(prompt)
