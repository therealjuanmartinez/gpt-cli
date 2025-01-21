from typing import Iterator, List, cast, TypedDict, Optional, Dict
from groq import Groq
#from groqapi.types.chat import ChatCompletionMessageParam
import os


import tiktoken

from gptcli.completion import CompletionProvider, Message


class GroqModelConfig(TypedDict):
    #path: str
    human_prompt: str
    assistant_prompt: str

#model_config = DOLPHIN_MODELS[args["model"]]
model_config: Optional[Dict[str, GroqModelConfig]] = {
        "human_prompt": "Human",
        "assistant_prompt": "Assistant",
}


class GroqCompletionProvider(CompletionProvider):
    def __init__(self):
        self.client = Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
        )

    def complete(
        self, messages: List[Message], args: dict, stream: bool = False
    ) -> Iterator[str]:
        kwargs = {}
        if "temperature" in args:
            kwargs["temperature"] = args["temperature"]
        if "top_p" in args:
            kwargs["top_p"] = args["top_p"]

        if not stream:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                #model="llama3-8b-8192",
                model="llama3-8b-8192",
            )
            if chat_completion.choices[0].message.content:
                yield chat_completion.choices[0].message.content
        else:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                #model="llama3-8b-8192",
                model="llama3-70b-8192",
                stream=True
            )

            for chunk in chat_completion:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

def make_prompt(messages: List[Message], model_config: GroqModelConfig) -> str:
    prompt = "\n".join(
        [
            f"{role_to_name(message['role'], model_config)} {message['content']}"
            for message in messages
        ]
    )
    prompt += f"\n{model_config['assistant_prompt']}"
    return prompt

def role_to_name(role: str, model_config: GroqModelConfig) -> str:
    if role == "system" or role == "user":
        return model_config["human_prompt"]
    elif role == "assistant":
        return model_config["assistant_prompt"]
    else:
        raise ValueError(f"Unknown role: {role}")


def num_tokens_from_messages_groqapi(messages: List[Message], model: str) -> int:
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        # every message follows <im_start>{role/name}\n{content}<im_end>\n
        num_tokens += 4
        for key, value in message.items():
            assert isinstance(value, str)
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens


def num_tokens_from_completion_groqapi(completion: Message, model: str) -> int:
    return num_tokens_from_messages_groqapi([completion], model)
