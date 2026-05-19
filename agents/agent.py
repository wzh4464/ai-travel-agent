# pylint: disable = http-used,print-used,no-self-use

import datetime
import json
import logging
import operator
import os
import re
from typing import Annotated, Any, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

from agents.errors import degrade
from agents.intent import (
    DialogState,
    clarification_question,
    extract_intent,
    missing_slots,
)
from agents.privacy import scrub, scrub_mapping
from agents.tools import ALL_TOOLS

logger = logging.getLogger(__name__)

_ = load_dotenv()

CURRENT_YEAR = datetime.datetime.now().year


_FLIGHT_INTENT_PATTERN = re.compile(
    r'(\bflight\w*|\bfly(ing)?\b|\bairline\w*|\bairfare\b|\bairport\b|'
    r'\bnonstop\b|\bnon-stop\b|\bone[- ]way\b|\bround[- ]?trip\b|'
    r'\bdeparture\b|\barrival\b|\blayover\b|\blanding\b|\btake[- ]?off\b|'
    r'机票|航班|航空|飞|班机|往返|单程|起飞|降落|登机|转机)',
    re.I,
)


def _looks_like_flight_request(text: str, intent) -> bool:
    """Heuristic gate: only force a flight clarifier when this *seems* like a flight request.

    The deterministic parser fills slots like origin/destination/dates only for
    flight-shaped phrases. If none of those parsed AND the raw text has no
    flight keywords, this is most likely a hotel / general / multi-tool request
    that should pass straight to the LLM.
    """
    if any((
        intent.origin_code, intent.origin_city,
        intent.destination_code, intent.destination_city, intent.destination_region,
        intent.outbound_date, intent.return_date,
    )):
        return True
    return bool(_FLIGHT_INTENT_PATTERN.search(text or ''))


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    dialog: DialogState


TOOLS_SYSTEM_PROMPT = f"""You are a smart travel agency with a layered flight-search toolchain.

Tool usage guidance:
  * Always resolve city names with `get_airport_code` before calling `flights_finder`.
  * Call `flights_finder` with structured, canonical parameters when the user
    has a specific origin and destination city pair. The result is a list of
    normalised Flight dicts with a `flight_id` for each item.
  * Call `open_jaw_search` instead of `flights_finder` when the user gives a
    flexible/region-level destination such as "somewhere in Europe", "欧洲",
    "northern europe", or when they say entry and exit cities don't have to
    match ("进出不同城市都可以", "open-jaw", "any European city"). Pass
    `avoid_transit=["middle_east"]` when the user says things like "不要中东中转"
    or "no Dubai/Doha connection".
  * To sort, filter, or compare existing results, call `compare_prices`—do not
    re-query `flights_finder` for trivial re-ranking.
  * For baggage / layover / booking URLs on a specific option, call
    `get_flight_details` with the original flights list and a `flight_id`.
  * For seat availability, call `check_seat_availability`.
  * If a tool returns a payload where ``status == 'error'``, read
    ``user_message`` and either retry with better arguments or ask the user
    a clarifying question—do not invent data.

Dialog rules:
  * You may ask multi-step clarifying questions when required slots
    (origin, destination, outbound_date) are missing or ambiguous.
  * The current year is {CURRENT_YEAR}. Never invent past dates.

Output rules:
  * Always include the price and currency for every flight.
  * Include the airline logo and booking URL when available.
  * Prefer tables/cards when presenting 3 or more flight options.
"""

TOOLS = ALL_TOOLS

EMAILS_SYSTEM_PROMPT = """Your task is to convert structured markdown-like text into a valid HTML email body.

- Do not include a ```html preamble in your response.
- The output should be in proper HTML format, ready to be used as the body of an email.
Here is an example:
<example>
Input:

I want to travel to New York from Madrid from October 1-7. Find me flights and 4-star hotels.

Expected Output:

<!DOCTYPE html>
<html>
<head>
    <title>Flight and Hotel Options</title>
</head>
<body>
    <h2>Flights from Madrid to New York</h2>
    <ol>
        <li>
            <strong>American Airlines</strong><br>
            <strong>Departure:</strong> Adolfo Suárez Madrid–Barajas Airport (MAD) at 10:25 AM<br>
            <strong>Arrival:</strong> John F. Kennedy International Airport (JFK) at 12:25 PM<br>
            <strong>Duration:</strong> 8 hours<br>
            <strong>Aircraft:</strong> Boeing 777<br>
            <strong>Class:</strong> Economy<br>
            <strong>Price:</strong> $702<br>
            <img src="https://www.gstatic.com/flights/airline_logos/70px/AA.png" alt="American Airlines"><br>
            <a href="https://www.google.com/flights">Book on Google Flights</a>
        </li>
    </ol>
</body>
</html>
</example>
"""


class Agent:

    def __init__(self, llm: Any = None, email_llm: Any = None):
        """Construct the agent graph.

        Parameters
        ----------
        llm:
            Optional chat model (already bound to ``TOOLS`` or not). When
            ``None`` a default ``ChatOpenAI('gpt-4o')`` is created. Tests
            inject a deterministic fake here to avoid real LLM calls.
        email_llm:
            Optional chat model used by the email-sender node.
        """
        self._tools = {t.name: t for t in TOOLS}
        if llm is None:
            self._tools_llm = ChatOpenAI(model='gpt-4o').bind_tools(TOOLS)
        elif hasattr(llm, 'bind_tools'):
            # Real chat models still need their tools bound, even if they
            # also expose .invoke (which every LangChain runnable does).
            self._tools_llm = llm.bind_tools(TOOLS)
        else:
            # Pre-bound runnable / test fake — use as-is.
            self._tools_llm = llm
        self._email_llm = email_llm

        builder = StateGraph(AgentState)
        builder.add_node('parse_intent', self.parse_intent)
        builder.add_node('call_tools_llm', self.call_tools_llm)
        builder.add_node('invoke_tools', self.invoke_tools)
        builder.add_node('email_sender', self.email_sender)
        builder.set_entry_point('parse_intent')

        builder.add_conditional_edges(
            'parse_intent',
            Agent.needs_clarification,
            {'clarify': END, 'ready': 'call_tools_llm'},
        )
        builder.add_conditional_edges(
            'call_tools_llm',
            Agent.exists_action,
            {'more_tools': 'invoke_tools', 'email_sender': 'email_sender'},
        )
        builder.add_edge('invoke_tools', 'call_tools_llm')
        builder.add_edge('email_sender', END)
        memory = MemorySaver()
        self.graph = builder.compile(checkpointer=memory, interrupt_before=['email_sender'])

        print(self.graph.get_graph().draw_mermaid())

    # ------------------------------------------------------------------
    # graph nodes
    # ------------------------------------------------------------------

    def parse_intent(self, state: AgentState):
        """Fast, deterministic intent extraction that runs before the LLM."""
        dialog = state.get('dialog') or DialogState()
        last_user = next(
            (m for m in reversed(state['messages']) if isinstance(m, HumanMessage)),
            None,
        )
        if last_user is not None:
            new_intent = extract_intent(last_user.content or '')
            dialog.merge(new_intent)

        updates: dict = {'dialog': dialog}

        missing = missing_slots(dialog.intent)
        # Only ask a clarification the first time a slot is missing, so the
        # LLM still has a chance to recover in later turns.
        pending = [m for m in missing if m not in dialog.clarifications_asked]
        message_text = last_user.content if last_user is not None else ''
        if pending and _looks_like_flight_request(message_text, dialog.intent):
            slot = pending[0]
            dialog.clarifications_asked.append(slot)
            updates['messages'] = [AIMessage(content=clarification_question(slot))]
        return updates

    @staticmethod
    def needs_clarification(state: AgentState) -> str:
        dialog = state.get('dialog')
        if not dialog:
            return 'ready'
        missing = missing_slots(dialog.intent)
        if missing and missing[0] in dialog.clarifications_asked:
            last = state['messages'][-1] if state['messages'] else None
            if isinstance(last, AIMessage) and last.content and getattr(last, 'tool_calls', None) in (None, []):
                return 'clarify'
        return 'ready'

    @staticmethod
    def exists_action(state: AgentState):
        result = state['messages'][-1]
        if len(getattr(result, 'tool_calls', []) or []) == 0:
            return 'email_sender'
        return 'more_tools'

    def email_sender(self, state: AgentState):
        logger.info('email_sender: preparing HTML email')
        email_llm = self._email_llm or ChatOpenAI(model='gpt-4o', temperature=0.1)
        email_message = [
            SystemMessage(content=EMAILS_SYSTEM_PROMPT),
            HumanMessage(content=state['messages'][-1].content),
        ]
        email_response = email_llm.invoke(email_message)
        logger.debug('email_sender: rendered %d chars of HTML', len(email_response.content or ''))

        message = Mail(
            from_email=os.environ['FROM_EMAIL'],
            to_emails=os.environ['TO_EMAIL'],
            subject=os.environ['EMAIL_SUBJECT'],
            html_content=email_response.content,
        )
        try:
            sg = SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))
            response = sg.send(message)
            logger.info('email_sender: SendGrid responded status=%s', response.status_code)
        except Exception as e:  # pylint: disable=broad-except
            logger.error('email_sender: SendGrid call failed: %s', scrub(str(e)))

    def call_tools_llm(self, state: AgentState):
        dialog = state.get('dialog') or DialogState()
        preamble = (
            'Known structured intent so far (merge with the latest user message): '
            f'{dialog.intent.as_dict()}'
        )
        messages = [
            SystemMessage(content=TOOLS_SYSTEM_PROMPT),
            SystemMessage(content=preamble),
        ] + state['messages']
        message = self._tools_llm.invoke(messages)
        return {'messages': [message]}

    def invoke_tools(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            # Log only the tool name and scrubbed argument keys — never the
            # raw values, which may contain passenger emails or passports.
            scrubbed_args = scrub_mapping(t.get('args') or {})
            logger.info('invoke_tools: calling %s with %s', t['name'], scrubbed_args)
            if t['name'] not in self._tools:
                logger.warning('invoke_tools: unknown tool %s', t['name'])
                result = {'status': 'error', 'error_type': 'UnknownTool',
                          'user_message': 'bad tool name, retry'}
            else:
                try:
                    result = self._tools[t['name']].invoke(t['args'])
                except Exception as exc:  # pylint: disable=broad-except
                    result = degrade(exc)
            # JSON-encode so the LLM can inspect status/error_type/user_message
            # rather than parsing a flattened Python repr.
            if isinstance(result, (dict, list)):
                content = json.dumps(result, default=str, ensure_ascii=False)
            else:
                content = str(result)
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=content))
        logger.debug('invoke_tools: handled %d tool call(s)', len(tool_calls))
        return {'messages': results}
