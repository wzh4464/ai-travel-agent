"""Regression tests for PII leaks via logs, degrade(), and tool results.

These tests pin behaviours we explicitly fixed in Package C:

1. ``errors.degrade()`` must scrub PII from any exception it wraps.
2. ``Agent.invoke_tools`` must log *scrubbed* tool-call arguments, never
   the raw dict (which could contain passenger email, passport, etc.).
3. A tool's return payload that is forwarded to the LLM must not contain
   PII-bearing upstream error text.
"""

from __future__ import annotations

import logging
import uuid

import pytest

pytest.importorskip('langchain_core')
pytest.importorskip('langgraph')

from langchain_core.messages import HumanMessage  # noqa: E402

from agents.agent import Agent  # noqa: E402
from agents.errors import UpstreamAPIError, degrade  # noqa: E402
from agents.privacy import REDACTED, contains_pii  # noqa: E402
from tests.e2e.fake_llm import FakeChatModel, ai_final, ai_tool_call  # noqa: E402


_PII = {
    'email': 'alice@example.com',
    'card': '4111 1111 1111 1111',
    'passport': 'A12345678',
}


class TestDegradeScrubs:
    def test_upstream_error_detail_scrubbed(self):
        payload = degrade(UpstreamAPIError('amadeus', status=500, detail=f'contact {_PII["email"]}'))
        assert _PII['email'] not in payload['details']
        assert not contains_pii(payload['details'])

    def test_unknown_exception_scrubbed(self):
        payload = degrade(RuntimeError(f'booking failed for card {_PII["card"]}'))
        assert _PII['card'] not in payload['details']
        assert REDACTED in payload['details']


class TestAgentLogRedaction:
    def test_invoke_tools_does_not_log_raw_args(self, caplog):
        """The ``invoke_tools`` node logs the tool call for observability;
        the message must use scrubbed args, never the raw dict."""
        llm = FakeChatModel([
            ai_tool_call('flights_finder', {
                'params': {
                    'departure_airport': 'JFK',
                    'arrival_airport': 'LHR',
                    'outbound_date': '2026-05-01',
                    # Extraneous PII as if the LLM over-shared.
                    'passenger_email': _PII['email'],
                    'passport_number': _PII['passport'],
                },
            }),
            ai_final('Done.'),
        ])
        agent = Agent(llm=llm)

        # Stub the tool so it returns something cheap.
        class _Stub:
            name = 'flights_finder'
            def invoke(self, args):
                return []
        agent._tools['flights_finder'] = _Stub()

        with caplog.at_level(logging.DEBUG, logger='agents.agent'):
            config = {'configurable': {'thread_id': str(uuid.uuid4())}}
            agent.graph.invoke({'messages': [HumanMessage(content='from New York to London on 2026-05-01')]}, config=config)

        log_blob = '\n'.join(rec.getMessage() for rec in caplog.records)
        # Raw PII must not appear anywhere in the log stream.
        assert _PII['email'] not in log_blob
        assert _PII['passport'] not in log_blob
        # But the tool name should still be there for debugging.
        assert 'flights_finder' in log_blob


class TestToolMessageScrubbing:
    def test_tool_error_payload_reaching_llm_is_scrubbed(self):
        """When a tool raises with PII in the message, the ToolMessage that
        goes back into the LLM context must be scrubbed by degrade()."""
        def _boom(_args):
            raise RuntimeError(f'upstream failure: {_PII["email"]} / card {_PII["card"]}')

        llm = FakeChatModel([
            ai_tool_call('flights_finder', {'params': {
                'departure_airport': 'JFK',
                'arrival_airport': 'LHR',
                'outbound_date': '2026-05-01',
            }}),
            ai_final('Sorry, upstream is down.'),
        ])
        agent = Agent(llm=llm)

        class _Stub:
            name = 'flights_finder'
            def invoke(self, args):
                return _boom(args)
        agent._tools['flights_finder'] = _Stub()

        config = {'configurable': {'thread_id': str(uuid.uuid4())}}
        result = agent.graph.invoke(
            {'messages': [HumanMessage(content='from New York to London on 2026-05-01')]},
            config=config,
        )

        tool_messages = [m for m in result['messages'] if getattr(m, 'type', None) == 'tool']
        assert tool_messages
        blob = '\n'.join(m.content for m in tool_messages)
        assert _PII['email'] not in blob
        assert _PII['card'] not in blob
        assert REDACTED in blob
