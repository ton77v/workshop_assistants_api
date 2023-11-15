"""
A CLASS 2 GENERATE PROMPTS; SET UP PERSONAS ETC <FOR TESTING VISION API>
based on 1st try out of OpenAI API
"""
from typing import Final
from dataclasses import dataclass, field, InitVar, asdict

from unittest import TestCase


# simple example of system prompt; normally we use much more detailed version with misc "tricks"
PERSONA_DEF: Final[str] = (
    'You are the world-class tutor for the students learning such matters as' + ': ' +
    'tank storage facility management and operations' + ', ' +
    'port marine terminals operations for oil and general liquid bulk operations' + ', etc. ' +
    'Your task is to help students and operators better understand their operations' + '.'
)


@dataclass
class PromptCreator:
    """ creates the params for use with OpenAI Client """
    user_query: InitVar[str]
    file_url: InitVar[str]
    model: str = field(init=True, default='gpt-4-vision-preview')
    sys_prompt_override: str = field(init=True, default=None)
    messages: list[dict[str, str]] = field(init=False, default=None)
    max_tokens: int = field(init=False, default=1024)  # 512 wasn't enough for Tank Specs!
    temperature: float = field(init=False, default=0)  # 1 by def | up to 2 now

    def prepare_messages(self, user_query: str, file_url: str) -> None:
        """ prepares the messages with Persona setting & passes text to the prompt """
        system_def = {
            'role': 'system',
            'content': self.sys_prompt_override or PERSONA_DEF
        }
        user_prompt = {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': user_query},
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': file_url,
                    },
                },
            ]
        }
        self.messages = [system_def, user_prompt]

    def __post_init__(self, user_query: str, file_url: str) -> None:
        self.prepare_messages(user_query, file_url)

    def chat_completion_params(self) -> dict[str, str | list[dict[str, str]] | int]:
        """
        Returns the chat completion parameters as a dictionary.

        :return: The chat completion parameters as a dictionary.
        :rtype: dict[str, str | list[dict[str, str]] | int]
        """
        res_dict = asdict(self)
        # will need to get rid of the prompt_override
        res_dict.pop('sys_prompt_override', None)
        return res_dict


class PromptCreatorTests(TestCase):

    def test_init(self):
        print(f'--test_init-- @ {self.__class__.__name__}')
        p = PromptCreator(user_query='TEST', file_url='https://example.com')
        print(p)
        self.assertIsInstance(p, PromptCreator)
        print(p.model)
        self.assertEqual(p.model, 'gpt-4-vision-preview')
        print(p.sys_prompt_override)
        self.assertIsNone(p.sys_prompt_override)
        print(p.messages)
        self.assertIsInstance(p.messages, list)
        self.assertEqual(len(p.messages), 2)
        print(p.messages[0])
        self.assertIsInstance(p.messages[0], dict)
        model_override = 'gpt-4-1106-preview'
        sys_prompt_override = 'You are Helpful AI Assistant'
        p2 = PromptCreator(user_query='TEST', file_url='https://example.com',
                           model=model_override, sys_prompt_override=sys_prompt_override)
        print(p2)
        self.assertIsInstance(p2, PromptCreator)
        print(p2.model)
        self.assertEqual(p2.model, model_override)
        print(p.sys_prompt_override)
        self.assertEqual(p2.sys_prompt_override, sys_prompt_override)

    def test_chat_completion_params(self):
        print('--test_chat_completion_params--')
        model = 'gpt-4-vision-preview'
        txt = 'TEST'
        url = 'https://example.com'
        p = PromptCreator(model=model, user_query=txt, file_url=url)
        print(p)
        params = p.chat_completion_params()
        print(params)
        self.assertIsInstance(params, dict)
        self.assertEqual(len(params.items()), 4)
        for k, v in params.items():
            print(k, v)
            self.assertIsInstance(k, str)
            match k:
                case 'model':
                    self.assertEqual(v, model)
                case 'messages':
                    self.assertIsInstance(v, list)
                case _:
                    self.assertIsInstance(v, int)
        user_prompt = p.messages
        print(user_prompt)
        self.assertIsInstance(user_prompt, list)
        self.assertEqual(len(user_prompt), 2)
        self.assertIsInstance(user_prompt[0], dict)
        self.assertIsInstance(user_prompt[1], dict)
        user_msgs = user_prompt[1].get('content', [])
        self.assertIsInstance(user_msgs, list)
        self.assertEqual(len(user_msgs), 2)
        self.assertEqual(user_msgs[0].get('text'), txt)
        self.assertEqual(user_msgs[1].get('type'), 'image_url')
        self.assertEqual(user_msgs[1].get('image_url', {}).get('url'), url)
