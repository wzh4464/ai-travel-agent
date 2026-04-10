# pylint: disable = http-used,print-used,no-self-use

import datetime
import operator
import os
from typing import Annotated, TypedDict

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
from agents.tools import ALL_TOOLS

_ = load_dotenv()

CURRENT_YEAR = datetime.datetime.now().year


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    dialog: DialogState


TOOLS_SYSTEM_PROMPT = f"""You are a smart travel agency with a layered flight-search toolchain.

Tool usage guidance:
  * Always resolve city names with `get_airport_code` before calling `flights_finder`.
  * Call `flights_finder` with structured, canonical parameters. The result is
    already a list of normalised Flight dicts with a `flight_id` for each item.
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

    def __init__(self):
        self._tools = {t.name: t for t in TOOLS}
        self._tools_llm = ChatOpenAI(model='gpt-4o').bind_tools(TOOLS)

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
        if pending:
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
        print('Sending email')
        email_llm = ChatOpenAI(model='gpt-4o', temperature=0.1)
        email_message = [
            SystemMessage(content=EMAILS_SYSTEM_PROMPT),
            HumanMessage(content=state['messages'][-1].content),
        ]
        email_response = email_llm.invoke(email_message)
        print('Email content:', email_response.content)

        message = Mail(
            from_email=os.environ['FROM_EMAIL'],
            to_emails=os.environ['TO_EMAIL'],
            subject=os.environ['EMAIL_SUBJECT'],
            html_content=email_response.content,
        )
        try:
            sg = SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))
            response = sg.send(message)
            print(response.status_code)
            print(response.body)
            print(response.headers)
        except Exception as e:
            print(str(e))

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
            print(f'Calling: {t}')
            if t['name'] not in self._tools:
                print('\n ....bad tool name....')
                result = 'bad tool name, retry'
            else:
                try:
                    result = self._tools[t['name']].invoke(t['args'])
                except Exception as exc:  # pylint: disable=broad-except
                    result = degrade(exc)
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print('Back to the model!')
        return {'messages': results}
