"""Scripted chat model so the full multi-agent graph runs in CI without API keys."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

_ids = itertools.count()


def call(name: str, args: dict | None = None) -> dict:
    return {"name": name, "args": args or {}, "id": f"call_{next(_ids)}", "type": "tool_call"}


Script = Callable[[int, list[BaseMessage]], AIMessage]


class ScriptedChatModel(BaseChatModel):
    """Picks a script from a keyword in the system prompt; the turn is the number of
    tool results already in the conversation."""

    scripts: dict[str, Script]

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
        return self

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        system = next((m.content for m in messages if isinstance(m, SystemMessage)), "")
        key = next(k for k in self.scripts if k in system)
        turn = sum(isinstance(m, ToolMessage) for m in messages)
        msg = self.scripts[key](turn, messages)
        msg.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        msg.response_metadata = {"model_name": f"scripted-{key}"}
        return ChatResult(generations=[ChatGeneration(message=msg)])
