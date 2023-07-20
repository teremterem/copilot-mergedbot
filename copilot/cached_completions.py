# pylint: disable=no-name-in-module
import json
from pathlib import Path
from typing import Iterable

from promptlayer import openai

from copilot.utils import SLOW_GPT_MODEL

COPILOT_MERGEDBOT_DIR_NAME = ".copilot-mergedbot"


class RepoCompletions:
    def __init__(
        self,
        repo: Path | str,
        completion_name: str,
        model: str = SLOW_GPT_MODEL,
        temperature: float = 0.0,
        **kwargs,
    ):
        if not isinstance(repo, Path):
            repo = Path(repo)
        self.repo = repo.resolve()
        self.completion_name = completion_name

        self.model = model
        self.temperature = temperature

        kwargs["model"] = model
        kwargs["temperature"] = temperature
        self.kwargs = kwargs

    async def chat_completion_for_file(
        self, messages: Iterable[dict[str, str]], repo_file: Path | str, **kwargs
    ) -> str:
        if not isinstance(repo_file, Path):
            repo_file = Path(repo_file)
        if not repo_file.is_absolute():
            repo_file = self.repo / repo_file
        repo_file = repo_file.resolve().relative_to(self.repo)

        # update local kwargs with self.kwargs but make sure that local kwargs take precedence
        kwargs = {**self.kwargs, **kwargs, "messages": messages}

        prompt_json_file = (
            self.repo / COPILOT_MERGEDBOT_DIR_NAME / f"{repo_file.as_posix()}.{self.completion_name}.prompt.json"
        )
        completion_str_file = (
            self.repo / COPILOT_MERGEDBOT_DIR_NAME / f"{repo_file.as_posix()}.{self.completion_name}.txt"
        )

        previous_prompt = json.loads(prompt_json_file.read_text(encoding="utf-8"))
        if previous_prompt == kwargs:
            # the prompt has not changed - return the cached completion
            return completion_str_file.read_text(encoding="utf-8")

        # either a completion for this file does not exist or the prompt has changed
        gpt_response = await openai.ChatCompletion.acreate(**kwargs)
        completion_str = gpt_response.choices[0].message.content

        completion_str_file.parent.mkdir(parents=True, exist_ok=True)
        completion_str_file.write_text(completion_str, encoding="utf-8")
        prompt_json_file.write_text(json.dumps(kwargs, indent=4), encoding="utf-8")

        return completion_str
