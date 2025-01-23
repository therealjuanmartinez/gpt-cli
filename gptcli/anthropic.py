import os
from typing import Iterator, List, cast
import anthropic

from gptcli.completion import CompletionProvider, Message

api_key = os.environ.get("ANTHROPIC_API_KEY")


def get_client():
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    return anthropic.Anthropic(api_key=api_key)


def role_to_name(role: str) -> str:
    if role == "system" or role == "user":
        return anthropic.HUMAN_PROMPT
    elif role == "assistant":
        return anthropic.AI_PROMPT
    else:
        raise ValueError(f"Unknown role: {role}")


def make_prompt(messages: List[Message]) -> str:
    prompt = "\n".join(
        [f"{role_to_name(message['role'])}{message['content']}" for message in messages]
    )
    prompt += f"{role_to_name('assistant')}"
    return prompt


#class OLDAnthropicCompletionProvider(CompletionProvider):
#    def complete(
#        self, messages: List[Message], args: dict, stream: bool = False
#    ) -> Iterator[str]:
#        kwargs = {
#            "stop_sequences": [anthropic.HUMAN_PROMPT],
#            "max_tokens": 4096,
#            "model": args["model"],
#        }
#
#        if "temperature" in args:
#            kwargs["temperature"] = args["temperature"]
#        if "top_p" in args:
#            kwargs["top_p"] = args["top_p"]
#
#        if len(messages) > 0 and messages[0]["role"] == "system":
#            kwargs["system"] = messages[0]["content"]
#            messages = messages[1:]
#
#        kwargs["messages"] = messages
#
#        client = get_client()
#        if stream:
#            with client.messages.stream(**kwargs) as stream:
#                for text in stream.text_stream:
#                    yield text
#        else:
#            response = client.messages.create(**kwargs, stream=False)
#            yield "".join(c.text for c in response.content)


class AnthropicCompletionProvider(CompletionProvider):
    def complete(
        self, messages: List[Message], args: dict, stream: bool = False, tools = []
    ) -> Iterator[str]:
        kwargs = {
            "stop_sequences": [anthropic.HUMAN_PROMPT],
            "max_tokens": 4096,
            "model": args["model"],
        }

        if "temperature" in args:
            kwargs["temperature"] = args["temperature"]
        if "top_p" in args:
            kwargs["top_p"] = args["top_p"]

        if len(messages) > 0 and messages[0]["role"] == "system":
            kwargs["system"] = messages[0]["content"]
            messages = messages[1:]

        kwargs["messages"] = messages

        # Parse tools if they're provided as a string
        while isinstance(tools, str):
            try:
                tools = json.loads(tools)
            except:
                break

        delme_tools = [{
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    # ... other properties
                },
                "required": ["location"]
            }
        }]


        # Add tools to the request if they're provided
        if len(tools) > 0:
            temptools = []

            for tool in tools:
                #tool->function->parameters <- this key needs to be renamed to input_schema
                temptool = tool.copy()
                try:
                    temptool['function']['input_schema'] = temptool['function']['parameters']
                    #now remove parameters
                    del temptool['function']['parameters']
                except:
                    pass
                temptools.append(temptool['function'])

            kwargs["tools"] = temptools

        client = get_client()

        if stream and len(tools) == 0:
            with client.messages.stream(**kwargs) as stream:
                tool = False
                first = True
                for content in stream.text_stream:
                    # Check if the content contains a tool call
                    if content.startswith("Function call:"):
                        tool = True
                        function_call = content.split("Function call: ", 1)[1]
                        function_name, function_arguments = function_call.split("(", 1)
                        function_arguments = function_arguments.rstrip(")")

                        if first:
                            yield("[", True)
                            first = False
                        else:
                            yield("}, ", True)
                        yield (f"{{ \"tool_call\" : \"{function_name}\", \"arguments\" : {function_arguments}", True)
                    else:
                        yield content

                if tool:
                    yield (" }]", True)

        else:
            response = client.messages.create(**kwargs, stream=False)
            content = "".join(c.text for c in response.content)

            # Check if the content contains a tool call
            if "Function call:" in content:
                tool_calls = []
                for line in content.split("\n"):
                    if line.startswith("Function call:"):
                        function_call = line.split("Function call: ", 1)[1]
                        function_name, function_arguments = function_call.split("(", 1)
                        function_arguments = function_arguments.rstrip(")")
                        tool_calls.append({
                            "tool_call": function_name,
                            "arguments": json.loads(function_arguments)
                        })
                yield json.dumps(tool_calls)
            else:
                yield content


def num_tokens_from_messages_anthropic(messages: List[Message], model: str) -> int:
    prompt = make_prompt(messages)
    client = get_client()
    return client.count_tokens(prompt)


def num_tokens_from_completion_anthropic(message: Message, model: str) -> int:
    client = get_client()
    return client.count_tokens(message["content"])
