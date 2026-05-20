"""A deterministic chat model double used by the E2E dialog tests.

The real ``Agent`` binds a chat model to its tool list and drives a
langgraph state machine. Tests must avoid hitting OpenAI, so we inject
this fake which plays back a pre-programmed script of ``AIMessage``
objects (one per ``invoke`` call).
"""

from __future__ import annotations

from typing import Any, Iterable

from langchain_core.messages import AIMessage


class FakeChatModel:
    """Minimal stand-in for ``ChatOpenAI(...).bind_tools(TOOLS)``.

    Each call to :meth:`invoke` returns the next scripted ``AIMessage``.
    ``bind_tools`` is a no-op that returns ``self`` so the Agent's
    conditional handling in ``__init__`` works unchanged.
    """

    def __init__(self, scripted_responses: Iterable[AIMessage]):
        self._responses = list(scripted_responses)
        self._call_log: list[Any] = []

    def bind_tools(self, _tools):
        return self

    @property
    def call_log(self) -> list[Any]:
        return self._call_log

    def invoke(self, messages, *args: Any, **kwargs: Any) -> AIMessage:
        self._call_log.append(messages)
        if not self._responses:
            raise AssertionError(
                'FakeChatModel ran out of scripted responses — the real '
                'agent made more LLM calls than expected.'
            )
        return self._responses.pop(0)


def ai_tool_call(tool_name: str, args: dict, *, tool_call_id: str = 'call-1') -> AIMessage:
    """Build an AIMessage carrying a single tool_call, with no final text."""
    return AIMessage(
        content='',
        tool_calls=[{
            'id': tool_call_id,
            'name': tool_name,
            'args': args,
            'type': 'tool_call',
        }],
    )


def ai_final(text: str) -> AIMessage:
    """Build a terminal AIMessage (no tool_calls) that ends the loop."""
    return AIMessage(content=text)
