from pathlib import Path
from typing import Iterable

from botmerger import InMemoryBotMerger
from langchain.schema import BaseMessage

FAST_GPT_MODEL = "gpt-3.5-turbo-0613"
FAST_LONG_GPT_MODEL = "gpt-3.5-turbo-16k-0613"
SLOW_GPT_MODEL = "gpt-4-0613"
EMBEDDING_MODEL = "text-embedding-ada-002"

bot_merger = InMemoryBotMerger()


def convert_lc_message_to_openai(message: BaseMessage) -> dict[str, str]:
    if message.type == "human":
        role = "user"
    elif message.type == "ai":
        role = "assistant"
    elif message.type == "system":
        role = "system"
    else:
        raise ValueError(f"Unexpected message type: {message.type}")
    return {"role": role, "content": message.content}


def sort_paths(paths: Iterable[Path], case_insensitive: bool = False) -> list[Path]:
    return sorted(paths, key=lambda p: (p.as_posix().lower(), p.as_posix()) if case_insensitive else p.as_posix())
