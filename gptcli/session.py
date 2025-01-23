from abc import abstractmethod
from typing_extensions import TypeGuard
from gptcli.assistant import Assistant
from gptcli.completion import (
    Message,
    ModelOverrides,
    CompletionError,
    BadRequestError,
    UsageEvent,
)
from typing import Any, Dict, List, Optional, Tuple


class ResponseStreamer:
    def __enter__(self) -> "ResponseStreamer":
        return self

    def on_next_token(self, token: str):
        pass

    def __exit__(self, *args):
        pass


class ChatListener:
    def on_chat_start(self):
        pass

    def on_chat_clear(self):
        pass

    def on_chat_rerun(self, success: bool):
        pass

    def on_error(self, error: Exception):
        pass

    def response_streamer(self) -> ResponseStreamer:
        return ResponseStreamer()

    def on_chat_message(self, message: Message):
        pass

    def on_chat_response(
        self,
        messages: List[Message],
        response: Message,
        overrides: ModelOverrides,
        usage: Optional[UsageEvent] = None,
    ):
        pass


class UserInputProvider:
    @abstractmethod
    def get_user_input(self) -> Tuple[str, Dict[str, Any]]:
        pass


class InvalidArgumentError(Exception):
    def __init__(self, message: str):
        self.message = message


COMMAND_CLEAR = (":clear", ":c")
COMMAND_QUIT = (":quit", ":q")
COMMAND_RERUN = (":rerun", ":r")
COMMAND_HELP = (":help", ":h", ":?")
ALL_COMMANDS = [*COMMAND_CLEAR, *COMMAND_QUIT, *COMMAND_RERUN, *COMMAND_HELP]
COMMANDS_HELP = """
Commands:
- `:clear` / `:c` / Ctrl+C - Clear the conversation.
- `:quit` / `:q` / Ctrl+D - Quit the program.
- `:rerun` / `:r` / Ctrl+R - Re-run the last message.
- `:help` / `:h` / `:?` - Show this help message.
"""


class ChatSession:
    def __init__(
        self,
        assistant: Assistant,
        listener: ChatListener,
    ):
        self.assistant = assistant
        self.messages: List[Message] = assistant.init_messages()
        self.user_prompts: List[Tuple[Message, ModelOverrides]] = []
        self.listener = listener

    def _clear(self):
        self.messages = self.assistant.init_messages()
        self.user_prompts = []
        self.listener.on_chat_clear()

    def _rerun(self):
        if len(self.user_prompts) == 0:
            self.listener.on_chat_rerun(False)
            return

        if self.messages[-1]["role"] == "assistant":
            self.messages = self.messages[:-1]

        self.listener.on_chat_rerun(True)
        _, args = self.user_prompts[-1]
        self._respond(args)

    def _respond(self, overrides: ModelOverrides) -> bool:
        """
        Respond to the user's input and return whether the assistant's response was saved.
        """
        next_response: str = ""
        usage: Optional[UsageEvent] = None
        try:
            completion_iter = self.assistant.complete_chat(
                self.messages, override_params=overrides
            )

            with self.listener.response_streamer() as stream:
                for event in completion_iter:
                    if event.type == "message_delta":
                        next_response += event.text
                        stream.on_next_token(event.text)
                    elif event.type == "usage":
                        usage = event

        except KeyboardInterrupt:
            # If the user interrupts the chat completion, we'll just return what we have so far
            pass
        except BadRequestError as e:
            self.listener.on_error(e)
            return False
        except CompletionError as e:
            self.listener.on_error(e)
            return True

        next_message: Message = {"role": "assistant", "content": next_response}
        self.listener.on_chat_message(next_message)
        self.listener.on_chat_response(self.messages, next_message, overrides, usage)

        self.messages = self.messages + [next_message]
        return True

    def _validate_args(self, args: Dict[str, Any]) -> TypeGuard[ModelOverrides]:
        for key in args:
            supported_overrides = self.assistant.supported_overrides()
            if key not in supported_overrides:
                self.listener.on_error(
                    InvalidArgumentError(
                        f"Invalid argument: {key}. Allowed arguments: {supported_overrides}"
                    )
                )
                return False
        return True

    def _add_user_message(self, user_input: str, args: ModelOverrides):
        user_message: Message = {"role": "user", "content": user_input}
        self.messages = self.messages + [user_message]
        self.listener.on_chat_message(user_message)
        self.user_prompts.append((user_message, args))

    def _rollback_user_message(self):
        self.messages = self.messages[:-1]
        self.user_prompts = self.user_prompts[:-1]

    def _print_help(self):
        with self.listener.response_streamer() as stream:
            stream.on_next_token(COMMANDS_HELP)

    def process_input(self, user_input: str, args: Dict[str, Any]):
        """
        Process the user's input and return whether the session should continue.
        """
        if not self._validate_args(args):
            return True

        if user_input in COMMAND_QUIT:
            return False
        elif user_input in COMMAND_CLEAR:
            self._clear()
            return True
        elif user_input in COMMAND_RERUN:
            self._rerun()
            return True
        elif user_input in COMMAND_HELP:
            self._print_help()
            return True

        self._add_user_message(user_input, args)
        response_saved = self._respond(args)
        if not response_saved:
            self._rollback_user_message()

        return True

    def loop(self, input_provider: UserInputProvider):
        self.listener.on_chat_start()
        while self.process_input(*input_provider.get_user_input()):
            pass
