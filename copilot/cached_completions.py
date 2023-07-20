# pylint: disable=no-name-in-module
from pathlib import Path
from typing import Iterable

from promptlayer import openai

from copilot.utils import SLOW_GPT_MODEL


async def chat_completion_for_repo_file(
    messages: Iterable[dict[str, str]],
    repo: Path,
    repo_file: Path,
    completion_name: str,
    model: str = SLOW_GPT_MODEL,
    **kwargs,
) -> str:
    gpt_response = await openai.ChatCompletion.acreate(messages=messages, model=model, **kwargs)
    return gpt_response.choices[0].message.content
