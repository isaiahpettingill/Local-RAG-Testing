from __future__ import annotations

from src.models.chat_llamacpp import ChatLlamaCpp
from src.evaluation.tools import search_knowledge, get_entity_relationships


def run_pipeline_a(prompt: str) -> str:
    model = ChatLlamaCpp("agent")
    return model.completion(prompt)


def run_pipeline_b(prompt: str) -> str:
    from langchain_core.messages import HumanMessage
    from langchain.agents import AgentExecutor, create_react_agent
    from src.models.config import MODELS

    model = ChatLlamaCpp("agent")
    tools = [search_knowledge]
    agent = create_react_agent(model, tools)
    executor = AgentExecutor(agent=agent, tools=tools)
    result = executor.invoke({"input": prompt})
    return result["output"]


def run_pipeline_c(prompt: str) -> str:
    from langchain_core.messages import HumanMessage
    from langchain.agents import AgentExecutor, create_react_agent

    model = ChatLlamaCpp("agent")
    tools = [get_entity_relationships]
    agent = create_react_agent(model, tools)
    executor = AgentExecutor(agent=agent, tools=tools)
    result = executor.invoke({"input": prompt})
    return result["output"]
