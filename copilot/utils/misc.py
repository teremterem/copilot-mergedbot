from pathlib import Path
from typing import Iterable

from botmerger import InMemoryBotMerger, MergedMessage, MergedBot
from langchain.schema import BaseMessage

FAST_GPT_MODEL = "gpt-3.5-turbo-0613"
FAST_LONG_GPT_MODEL = "gpt-3.5-turbo-16k-0613"
SLOW_GPT_MODEL = "gpt-4-0613"
EMBEDDING_MODEL = "text-embedding-ada-002"

bot_merger = InMemoryBotMerger()


def get_openai_role_name(message: MergedMessage, this_bot: MergedBot) -> str:
    return "assistant" if message.sender == this_bot else "user"


def langchain_messages_to_openai(
    message: BaseMessage | Iterable[BaseMessage],
) -> dict[str, str] | list[dict[str, str]]:
    if isinstance(message, BaseMessage):
        if message.type == "human":
            role = "user"
        elif message.type == "ai":
            role = "assistant"
        elif message.type == "system":
            role = "system"
        else:
            raise ValueError(f"Unexpected message type: {message.type}")
        return {"role": role, "content": message.content}

    return [langchain_messages_to_openai(m) for m in message]


def sort_paths(paths: Iterable[Path], case_insensitive: bool = False) -> list[Path]:
    return sorted(paths, key=lambda p: (p.as_posix().lower(), p.as_posix()) if case_insensitive else p.as_posix())


async def reliable_chat_completion(**kwargs) -> str:
    # pylint: disable=import-outside-toplevel,no-name-in-module
    from promptlayer import openai

    response = await openai.ChatCompletion.acreate(**kwargs)
    completion = response.choices[0]
    if completion.finish_reason != "stop":
        raise RuntimeError(f"Incomplete chat completion (finish_reason: {completion.finish_reason})")
    return completion.message.content
