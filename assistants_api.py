"""
DEMO: testing Assistants API
"""
import time
from typing import Final, Any, Callable
from dataclasses import dataclass, field
import json
import argparse
import os
import pprint

from dotenv import load_dotenv  # pip install python-dotenv
from openai import OpenAI  # pip install openai
from openai.types import FileObject
from openai.types.beta import Assistant, Thread
from openai.types.beta.threads import ThreadMessage, Run, MessageContentText, RequiredActionFunctionToolCall
from openai.types.beta.threads.run import RequiredAction
from openai.pagination import SyncCursorPage
# from pydantic import BaseModel  # pip install pydantic

from prompts_creator import PERSONA_DEF
# noinspection PyUnresolvedReferences
from agent_tools import query_tank_specs_tool, query_tank_specs  # should import for globals()!


load_dotenv()
pp = pprint.PrettyPrinter(indent=2)

# creating Open AI API Client instance
OPENAI_API_KEY: Final[str] = os.environ.get('OPENAI_API_KEY')
CLIENT: Final[OpenAI] = OpenAI(api_key=OPENAI_API_KEY)


def pretty_print_obj_json(obj: Any) -> None:
    """ assuming obj: Pydantic BaseModel
     | not typing since not in requirements.txt """
    try:
        pp.pprint(json.loads(obj.model_dump_json()))
    except:
        pp.pprint(obj)


def create_file_ids() -> list[str]:
    """ Creates a list of file ids """
    # 📝 in production/dev we would do this once, storing IDs | here small hack
    saved_file_id = os.environ.get('SAVED_FILE_ID')
    if saved_file_id:
        print(f'\ncreate_file_ids: loaded saved file id: {saved_file_id}')
        return [saved_file_id]
    # reading the document from a file
    file_path = 'docs/listing_of_standards_and_practices.pdf'
    with open(file_path, "rb") as _file:
        file_content = _file.read()
    try:
        file: FileObject = CLIENT.files.create(
            file=file_content,
            purpose="assistants",
        )
    except Exception as e:
        # openai.InternalServerError: Error code: 500 seen randomly @ high load
        print(repr(e), e)
        return []
    pp.pprint(file)
    print(f'\nfile id: {file.id} created')
    return [file.id]


def create_assistant() -> Assistant:
    """ Creates an assistant instance """
    # 📝 in production/dev we would do this once, storing IDs
    tools = [
        # {"type": "code_interpreter"},
        {"type": "retrieval"},  # built-in Retriever
        query_tank_specs_tool()  # our custom function mock
    ]
    # it seems Vision API for image analysis should be implemented separately if required
    # although "user messages with images coming soon" as per OpenAI API docs
    file_ids = create_file_ids()
    assistant = CLIENT.beta.assistants.create(
        name='TankOps / MarineOps Tutor',
        instructions=PERSONA_DEF,
        model='gpt-4-1106-preview',  # typically from config
        tools=tools,
        file_ids=file_ids
    )
    pretty_print_obj_json(assistant)
    print(f'\nAssistant id: {assistant.id} created')
    return assistant


def retrieve_messages_content(messages: SyncCursorPage[ThreadMessage]) -> str:
    """ Retrieves messages content """
    def _get_content(msg: ThreadMessage) -> str:
        # data => content[0].text => value
        _content: MessageContentText = msg.content[0] if msg.content and len(msg.content) else None
        return _content.text.value if _content and _content.text else ''

    # 📝 here we could format the references nicely:
    # https://platform.openai.com/docs/assistants/how-it-works/managing-threads-and-messages
    return '\n\n'.join([_get_content(msg) for msg in messages])


@dataclass
class AssistantController:
    client: OpenAI = field(init=False, default=CLIENT)
    # assistant: Assistant = field(init=False, default=None)
    assistant_id: str = field(default=None)
    thread_id: str = field(default=None)

    def __post_init__(self):
        if not self.assistant_id:
            assistant = create_assistant()
            self.assistant_id = assistant.id

    # creating a thread for the assistant
    def create_thread(self) -> str:
        """ Creates a Thread for the new Conversation """
        thread: Thread = CLIENT.beta.threads.create()
        pretty_print_obj_json(thread)
        self.thread_id = thread.id
        print(f'\nstarted a new Thread | id: {thread.id}')
        return thread.id

    def call_tool_and_re_run(self, run: Run, func_call: RequiredActionFunctionToolCall) -> Run:
        call_id = func_call.id
        func_name = func_call.function.name
        func_args = json.loads(func_call.function.arguments) if func_call.function.arguments else None
        if not func_name:
            print(f'\n===> ❌ no function name found in call: {call_id}')
            return run
        print(f'\n===> calling: {func_name}({func_args})')
        # assert func_name == 'query_tank_specs'
        func: Callable = globals()[func_name]
        func_result = func(**func_args)
        # now we need to re-run given the func output
        run = self.client.beta.threads.runs.submit_tool_outputs(
            thread_id=self.thread_id,
            run_id=run.id,
            tool_outputs=[
                {
                    "tool_call_id": call_id,
                    "output": json.dumps(func_result),
                }
            ],
        )
        pretty_print_obj_json(run)
        return self.wait_on_run(run)

    def process_run_result(self, run: Run) -> Run:
        """ taking necessary actions depending on the run status """
        def extract_func_call(act: RequiredAction) -> RequiredActionFunctionToolCall:
            """  extracts the function call | DEMO: assuming just one at a time! """
            # required_action.submit_tool_outputs.tool_calls[0]
            _call = act.submit_tool_outputs.tool_calls[0] if (
                    act.submit_tool_outputs and act.submit_tool_outputs.tool_calls) else None
            return _call

        # https://platform.openai.com/docs/assistants/how-it-works/runs-and-run-steps
        match run.status:  # all statuses ↗
            case 'requires_action':
                print(f'\nRun requires action:')
                required_action: RequiredAction = run.required_action
                pretty_print_obj_json(required_action)
                # for sake of demo we assume there's only one call | in prod use ↘
                # https://platform.openai.com/docs/guides/function-calling/parallel-function-calling
                func_call: RequiredActionFunctionToolCall = extract_func_call(required_action)
                if func_call and func_call.function:
                    return self.call_tool_and_re_run(run, func_call)
        return run

    def wait_on_run(self, run: Run) -> Run:
        """ Waits for the Run to complete """
        run_started = time.perf_counter()
        print(f'\nwait_on_run | status: {run.status}')
        intermediary_statuses = ['queued', 'in_progress']
        while run.status in intermediary_statuses:
            run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread_id,
                run_id=run.id,
            )
            time.sleep(0.5)
        run_time = time.perf_counter() - run_started
        pretty_print_obj_json(run)
        print(f'\nRun complete with status: {run.status} | took {run_time:.2f} seconds')
        self.process_run_result(run)
        return run

    def chat(self, message: str) -> tuple[str, str, str]:
        """ Sends a User message to the Assistant returning:
            -> Assistant ID
            -> Thread ID
            -> and the text of answer(s) """
        # creating a new thread if none provided
        if not self.thread_id:
            self.create_thread()
        # now we need to create a Message & add to the thread
        message: ThreadMessage = self.client.beta.threads.messages.create(
            thread_id=self.thread_id,
            role='user',
            content=message,
        )
        pretty_print_obj_json(message)
        # starting a Run | should supply Assistant ID since Thread is not tied to an Assistant
        run: Run = self.client.beta.threads.runs.create(
            assistant_id=self.assistant_id,
            thread_id=self.thread_id,
        )
        pretty_print_obj_json(run)
        # creating a Run is an asynchronous operation while it's not Awaitable
        # hence a bit clunky approach to awaiting the result
        self.wait_on_run(run)
        # retrieve all the messages added after our last user message
        new_messages: SyncCursorPage[ThreadMessage] = self.client.beta.threads.messages.list(
            thread_id=self.thread_id, order='asc', after=message.id
        )
        pretty_print_obj_json(new_messages)
        # and finally the answer(s) text:
        answer = retrieve_messages_content(new_messages)
        pp.pprint(answer)
        return self.assistant_id, self.thread_id, answer


# CLI arguments
PROJ_DESC: Final[str] = 'Testing OpenAI Assistants API'
QUESTION_HELP_TXT: Final[str] = 'a question to ask the Tutor'
ASSISTANT_ID_HELP_TXT: Final[str] = 'the assistant ID to use | optional'
THREAD_ID_HELP_TXT: Final[str] = 'the thread ID to use | optional'
# and the parser for these:
parser = argparse.ArgumentParser(description=PROJ_DESC)
parser.add_argument('--question', '-q',
                    help=QUESTION_HELP_TXT,  # optional
                    action='store')
parser.add_argument('--assistant-id', '-a',
                    help=ASSISTANT_ID_HELP_TXT,
                    action='store')
parser.add_argument('--thread-id', '-t',
                    help=THREAD_ID_HELP_TXT,
                    action='store')

if __name__ == '__main__':
    # input params: question, assistant_id, thread_id
    args = parser.parse_args()
    question = args.question
    if not question:
        parser.print_help()
        exit(1)
    assistant_id = args.assistant_id
    thread_id = args.thread_id
    controller = AssistantController(assistant_id=assistant_id, thread_id=thread_id)
    results = controller.chat(question)
    print(f'\n\n\nAssistant ID: {results[0]}\nThread ID: {results[1]}\nAnswer(s):\n\n{results[2]}')
