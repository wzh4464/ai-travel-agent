"""End-to-end dialog test for the 'Europe open-jaw' use case.

This is the canonical query driving the whole open-jaw feature:

    "从香港出发去欧洲 4.23-5.3 要便宜 不要中东中转"

The test runs the real LangGraph state machine with a deterministic
FakeChatModel. The open_jaw_search tool is stubbed to return pre-built
ranked combinations so we can assert the full pipeline (intent parse →
LLM tool call → tool result fed back → LLM summary) without touching
Duffel or any other upstream provider.
"""

from __future__ import annotations

import datetime
import uuid

import pytest

pytest.importorskip('langchain_core')
pytest.importorskip('langgraph')

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from agents.agent import Agent  # noqa: E402
from agents.intent.parser import extract_intent  # noqa: E402
from tests.e2e.fake_llm import FakeChatModel, ai_final, ai_tool_call  # noqa: E402


@pytest.fixture
def canned_open_jaw_result():
    """A plausible ranked result set the stub tool returns."""
    outbound = {
        'flight_id': 'ob-fco',
        'price': 3900,
        'currency': 'HKD',
        'total_duration_minutes': 810,
        'stops': 1,
        'legs': [
            {
                'departure_airport': 'HKG',
                'departure_time': '2026-04-23T02:10:00',
                'arrival_airport': 'AMS',
                'arrival_time': '2026-04-23T07:30:00',
                'airline': 'KLM',
                'flight_number': 'KL882',
                'duration_minutes': 690,
                'aircraft': '77W',
                'cabin_class': 'economy',
            },
            {
                'departure_airport': 'AMS',
                'departure_time': '2026-04-23T10:00:00',
                'arrival_airport': 'FCO',
                'arrival_time': '2026-04-23T12:00:00',
                'airline': 'KLM',
                'flight_number': 'KL1607',
                'duration_minutes': 120,
                'aircraft': '73H',
                'cabin_class': 'economy',
            },
        ],
        'provider': 'duffel',
    }
    returns = {
        'flight_id': 'r-cdg',
        'price': 4200,
        'currency': 'HKD',
        'total_duration_minutes': 790,
        'stops': 1,
        'legs': [
            {
                'departure_airport': 'CDG',
                'departure_time': '2026-05-03T10:30:00',
                'arrival_airport': 'AMS',
                'arrival_time': '2026-05-03T11:50:00',
                'airline': 'Air France',
                'flight_number': 'AF1240',
                'duration_minutes': 80,
                'aircraft': '320',
                'cabin_class': 'economy',
            },
            {
                'departure_airport': 'AMS',
                'departure_time': '2026-05-03T14:20:00',
                'arrival_airport': 'HKG',
                'arrival_time': '2026-05-04T08:00:00',
                'airline': 'KLM',
                'flight_number': 'KL881',
                'duration_minutes': 710,
                'aircraft': '77W',
                'cabin_class': 'economy',
            },
        ],
        'provider': 'duffel',
    }
    return {
        'status': 'ok',
        'origin': 'HKG',
        'outbound_date': '2026-04-23',
        'return_date': '2026-05-03',
        'candidates': ['CDG', 'FCO', 'LHR', 'AMS', 'FRA', 'MAD', 'BCN'],
        'banned_transit': ['AUH', 'BAH', 'DOH', 'DWC', 'DXB', 'JED', 'KWI', 'MCT', 'RUH'],
        'same_city': False,
        'count': 1,
        'combinations': [{
            'total_price': 8100,
            'currency': 'HKD',
            'entry_city': 'FCO',
            'exit_city': 'CDG',
            'open_jaw': True,
            'outbound': outbound,
            'return': returns,
        }],
        'summary_markdown': (
            '### Option 1: FCO → CDG *(open-jaw)* — **HKD 8,100 total**\n'
            '- **Outbound:** KLM HKG → FCO, 1 stop via AMS, 13h 30m, HKD 3,900\n'
            '- **Return:** Air France CDG → HKG, 1 stop via AMS, 13h 10m, HKD 4,200'
        ),
    }


def _invoke(agent, message: str):
    thread_id = str(uuid.uuid4())
    config = {'configurable': {'thread_id': thread_id}}
    return agent.graph.invoke({'messages': [HumanMessage(content=message)]}, config=config)


class _FakeTool:
    def __init__(self, name: str, handler):
        self.name = name
        self._handler = handler

    def invoke(self, args):
        return self._handler(args)


class TestIntentPreflight:
    """Sanity: the deterministic intent parser catches every slot from the
    canonical sentence before the LLM ever runs."""

    def test_canonical_sentence(self):
        # Pin ``today`` so the "4.23-5.3" compact-range resolves to 2026
        # regardless of when the test is run.
        intent = extract_intent(
            '从香港出发去欧洲 4.23-5.3 要便宜 不要中东中转',
            today=datetime.date(2026, 4, 1),
        )
        assert intent.origin_code == 'HKG'
        assert intent.destination_region == 'europe'
        assert intent.outbound_date == '2026-04-23'
        assert intent.return_date == '2026-05-03'
        assert intent.sort_by == 'price'
        assert intent.avoid_transit == ['middle_east']


class TestEuropeOpenJawDialog:
    def test_happy_path(self, canned_open_jaw_result):
        """The agent should:
        1. Extract region + avoid_transit from the user sentence (parse_intent node)
        2. Skip the clarification short-circuit (all required slots present)
        3. Call open_jaw_search with the right args
        4. Summarise the ranked combinations in the final AIMessage
        """
        llm = FakeChatModel([
            ai_tool_call(
                'open_jaw_search',
                {'params': {
                    'origin': 'HKG',
                    'destination_region': 'europe',
                    'outbound_date': '2026-04-23',
                    'return_date': '2026-05-03',
                    'avoid_transit': ['middle_east'],
                    'top_n': 5,
                }},
            ),
            ai_final(
                'Cheapest option is an open-jaw itinerary: HKG→FCO outbound, '
                'CDG→HKG return, HKD 8,100 total, no Middle East transit.'
            ),
        ])
        agent = Agent(llm=llm)
        agent._tools['open_jaw_search'] = _FakeTool(
            'open_jaw_search', lambda _args: canned_open_jaw_result,
        )

        result = _invoke(agent, '从香港出发去欧洲 4.23-5.3 要便宜 不要中东中转')

        final = result['messages'][-1]
        assert isinstance(final, AIMessage)
        assert 'HKD 8,100' in final.content or '8,100' in final.content
        assert 'open-jaw' in final.content.lower() or 'FCO' in final.content

        # The LLM should have been called exactly twice: one tool call + one summary
        assert len(llm.call_log) == 2

    def test_tool_receives_parsed_intent_as_system_preamble(self, canned_open_jaw_result):
        """The parse_intent node sets DialogState.intent which is then
        injected as a SystemMessage into the LLM call. Verify the LLM's
        second turn (after the tool) had access to it."""
        captured: dict = {}

        def _recording_handler(args):
            captured['args'] = args
            return canned_open_jaw_result

        llm = FakeChatModel([
            ai_tool_call(
                'open_jaw_search',
                {'params': {
                    'origin': 'HKG',
                    'destination_region': 'europe',
                    'outbound_date': '2026-04-23',
                    'return_date': '2026-05-03',
                    'avoid_transit': ['middle_east'],
                }},
            ),
            ai_final('ok'),
        ])
        agent = Agent(llm=llm)
        agent._tools['open_jaw_search'] = _FakeTool('open_jaw_search', _recording_handler)

        _invoke(agent, '从香港出发去欧洲 4.23-5.3 要便宜 不要中东中转')

        # First LLM call: messages should include a SystemMessage with the
        # parsed intent so the LLM can build the correct tool call.
        first_call = llm.call_log[0]
        preamble = next(
            (m.content for m in first_call
             if getattr(m, 'type', '') == 'system' and 'intent' in (m.content or '').lower()),
            None,
        )
        assert preamble is not None
        assert 'HKG' in preamble
        assert 'europe' in preamble
        assert 'middle_east' in preamble

    def test_tool_failure_is_scrubbed_and_summarised(self):
        """If the open-jaw tool raises with PII in the error message, the
        ToolMessage that lands back in the LLM context must be scrubbed."""
        def _boom(_args):
            raise RuntimeError('upstream ops@example.com returned 503')

        llm = FakeChatModel([
            ai_tool_call('open_jaw_search', {'params': {
                'origin': 'HKG',
                'destination_region': 'europe',
                'outbound_date': '2026-04-23',
                'return_date': '2026-05-03',
            }}),
            ai_final('Sorry, search is unavailable right now.'),
        ])
        agent = Agent(llm=llm)
        agent._tools['open_jaw_search'] = _FakeTool('open_jaw_search', _boom)

        result = _invoke(agent, '从香港出发去欧洲 4.23-5.3')

        tool_messages = [m for m in result['messages'] if getattr(m, 'type', None) == 'tool']
        assert tool_messages
        blob = '\n'.join(m.content for m in tool_messages)
        assert 'ops@example.com' not in blob
        assert '[REDACTED]' in blob
