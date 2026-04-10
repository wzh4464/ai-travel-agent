"""End-to-end dialog tests driving the real LangGraph state machine.

These tests assemble an ``Agent`` with a deterministic :class:`FakeChatModel`,
patch the aggregator so no network traffic leaks, and assert that the
graph runs the expected node sequence and produces the expected final
state. They are gated on langchain_core + langgraph being installed.
"""

from __future__ import annotations

import uuid

import pytest

pytest.importorskip('langchain_core')
pytest.importorskip('langgraph')

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from agents.agent import Agent  # noqa: E402
from tests.e2e.fake_llm import FakeChatModel, ai_final, ai_tool_call  # noqa: E402


@pytest.fixture
def canned_flights():
    """A tiny canonical Flight list the fake flights_finder returns."""
    return [{
        'flight_id': 'f1',
        'price': 420.5,
        'currency': 'USD',
        'total_duration_minutes': 690,
        'stops': 0,
        'legs': [{
            'airline': 'American Airlines',
            'flight_number': 'AA100',
            'departure_airport': 'JFK',
            'departure_time': '2026-05-01T08:00:00',
            'arrival_airport': 'LHR',
            'arrival_time': '2026-05-01T19:30:00',
            'duration_minutes': 690,
            'aircraft': '777',
            'cabin_class': 'economy',
        }],
        'airline_logo': '',
        'booking_url': '',
        'provider': 'amadeus',
        'raw': {},
    }]


class _FakeTool:
    """Stand-in for a LangChain tool. Only needs ``name`` and ``invoke``."""

    def __init__(self, name: str, handler):
        self.name = name
        self._handler = handler

    def invoke(self, args):
        return self._handler(args)


def _replace_tool(agent, name: str, handler):
    """Swap the real tool out of ``agent._tools`` with a deterministic stub.

    ``agent._tools`` is just a name→tool dict built in ``Agent.__init__``.
    The dispatch in ``invoke_tools`` only reads ``.invoke``, so any object
    with that method works.
    """
    agent._tools[name] = _FakeTool(name, handler)


def _invoke(agent, message: str, thread_id: str | None = None):
    thread_id = thread_id or str(uuid.uuid4())
    config = {'configurable': {'thread_id': thread_id}}
    return agent.graph.invoke({'messages': [HumanMessage(content=message)]}, config=config), thread_id


class TestHappyPathSearch:
    def test_search_to_final_answer(self, canned_flights):
        """Golden path: user asks, LLM calls flights_finder, LLM summarises."""
        llm = FakeChatModel([
            ai_tool_call(
                'flights_finder',
                {'params': {
                    'departure_airport': 'JFK',
                    'arrival_airport': 'LHR',
                    'outbound_date': '2026-05-01',
                    'adults': 1,
                    'cabin_class': 'economy',
                }},
            ),
            ai_final('American Airlines AA100 at USD 420.50, non-stop from JFK to LHR.'),
        ])
        agent = Agent(llm=llm)
        _replace_tool(agent, 'flights_finder', lambda args: canned_flights)

        result, _ = _invoke(agent, 'from New York to London on 2026-05-01')

        final = result['messages'][-1]
        assert isinstance(final, AIMessage)
        assert 'American Airlines' in final.content
        assert '420' in final.content
        # LLM should have been called twice: once to produce the tool call,
        # once to summarise the tool result.
        assert len(llm.call_log) == 2


class TestClarification:
    def test_missing_destination_triggers_clarifier(self):
        """parse_intent should short-circuit the graph when a slot is missing."""
        llm = FakeChatModel([])  # no LLM calls expected
        agent = Agent(llm=llm)

        result, _ = _invoke(agent, 'I want to fly somewhere cheap')

        final = result['messages'][-1]
        assert isinstance(final, AIMessage)
        # The clarification message comes from parse_intent, not the LLM.
        assert '?' in final.content
        assert llm.call_log == []  # LLM never invoked


class TestToolFailureDegradesGracefully:
    def test_tool_exception_is_scrubbed_and_fed_back(self):
        """A tool that raises should surface as a structured error to the LLM,
        with any PII in the exception message stripped."""

        def _boom(args):
            raise RuntimeError('upstream ops@example.com returned 503')

        llm = FakeChatModel([
            ai_tool_call(
                'flights_finder',
                {'params': {
                    'departure_airport': 'JFK',
                    'arrival_airport': 'LHR',
                    'outbound_date': '2026-05-01',
                }},
            ),
            ai_final('Sorry, the upstream search is unavailable right now.'),
        ])
        agent = Agent(llm=llm)
        _replace_tool(agent, 'flights_finder', _boom)

        result, _ = _invoke(agent, 'from New York to London on 2026-05-01')

        # Inspect the ToolMessage that was fed back to the LLM (second call).
        tool_messages = [m for m in result['messages'] if m.type == 'tool']
        assert tool_messages, 'expected a ToolMessage after the failing tool call'
        payload = tool_messages[0].content
        assert 'ops@example.com' not in payload
        assert '[REDACTED]' in payload
        assert 'UnknownError' in payload or 'error' in payload
