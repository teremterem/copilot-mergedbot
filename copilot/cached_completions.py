# pylint: disable=no-name-in-module
from pathlib import Path
from typing import Iterable

from promptlayer import openai

from copilot.utils import SLOW_GPT_MODEL


class RepoCompletions:
    def __init__(
        self,
        repo: Path,
        completion_name: str,
        model: str = SLOW_GPT_MODEL,
        temperature: float = 0.0,
        **kwargs,
    ):
        self.repo = repo
        self.completion_name = completion_name

        self.model = model
        self.temperature = temperature

        kwargs["model"] = model
        kwargs["temperature"] = temperature
        self.kwargs = kwargs

    async def chat_completion_for_file(self, messages: Iterable[dict[str, str]], repo_file: Path, **kwargs) -> str:
        if not repo_file.resolve().is_relative_to(self.repo.resolve()):
            raise ValueError(f"repo_file {repo_file} is not in repo {self.repo}")
        # update local kwargs with self.kwargs but make sure that local kwargs take precedence
        kwargs = {**self.kwargs, **kwargs}
        gpt_response = await openai.ChatCompletion.acreate(messages=messages, **kwargs)
        return gpt_response.choices[0].message.content
