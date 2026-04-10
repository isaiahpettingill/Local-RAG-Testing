from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def clear_chat_llama_instances():
    from src.models.chat_llamacpp import ChatLlamaCpp

    ChatLlamaCpp._instances.clear()
    yield
    ChatLlamaCpp._instances.clear()
