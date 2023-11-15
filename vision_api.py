"""
testing Vision API
"""
from typing import Final
from dataclasses import dataclass, field
import os
import argparse
import base64
import pprint

from dotenv import load_dotenv  # pip install python-dotenv
from openai import OpenAI, ChatCompletion  # pip install openai
from openai.types.chat.chat_completion import Choice

from prompts_creator import PromptCreator


load_dotenv()
pp = pprint.PrettyPrinter(indent=2)

OPENAI_API_KEY: Final[str] = os.environ.get('OPENAI_API_KEY')


# Function to encode the image
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# Returns the answer from OpenAI response
def get_answer(response: ChatCompletion) -> str:
    choice: Choice = response.choices[0]
    # Choice & ChatCompletionMessage are Pydantic Base Models
    msg_content = choice.message.content
    return msg_content


@dataclass
class VisionController:
    _api_key: str = field(default=OPENAI_API_KEY, repr=False)
    llm: str = field(default='gpt-4-vision-preview')
    use_generic_prompt: bool = field(default=False)

    def __post_init__(self):
        self.client = OpenAI(api_key=self._api_key)

    def ask_question(self, question: str, image_path: str) -> str:
        """
        :param question: The question to ask the model
        :param image_path: The path to the image file
        :return: The chat completion answer
        """
        img_encoded = encode_image(image_path)
        file_url = f'data:image/png;base64,{img_encoded}'
        prompt_override = 'Helpful AI Assistant' if self.use_generic_prompt else None
        params = PromptCreator(model=self.llm, user_query=question,
                               file_url=file_url,
                               sys_prompt_override=prompt_override).chat_completion_params()
        response: ChatCompletion = self.client.chat.completions.create(**params)
        pp.pprint(response)
        answer = get_answer(response)
        return answer


# CLI arguments
PROJ_DESC: Final[str] = 'Testing OpenAI Vision API'
QUESTION_HELP_TXT: Final[str] = 'a question to ask about the image'
IMAGE_PATH_HELP_TXT: Final[str] = 'full path to the image'
USE_GENERIC_PROMPT: Final[str] = 'use generic prompt "Helpful AI Assistant"'
# and the parser for these:
parser = argparse.ArgumentParser(description=PROJ_DESC)
parser.add_argument('--question', '-q',
                    help=QUESTION_HELP_TXT,  # optional
                    action='store')
parser.add_argument('--image-path', '-i',
                    help=IMAGE_PATH_HELP_TXT,
                    action='store')
parser.add_argument('--use-generic-prompt', '-g',
                    help=USE_GENERIC_PROMPT,
                    action=argparse.BooleanOptionalAction)


if __name__ == '__main__':
    # input params: question, image_path
    args = parser.parse_args()
    img_question = args.question
    img_path = args.image_path
    use_generic_prompt = args.use_generic_prompt
    if not (img_question and img_path):
        parser.print_help()
        exit(1)
    controller = VisionController(use_generic_prompt=use_generic_prompt)
    result = controller.ask_question(img_question, img_path)
    print(f'\n\nanswer: {result}')
