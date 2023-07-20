from pathlib import Path

from copilot.cached_completions import chat_completion_for_repo_file
from copilot.utils import FAST_GPT_MODEL


async def main() -> None:
    messages = [{"role": "assistant", "content": "Привіт world!"}]
    print(
        await chat_completion_for_repo_file(
            messages=messages,
            repo=Path(__file__).parent,
            repo_file=Path(__file__),
            completion_name="helloworld",
            model=FAST_GPT_MODEL,
        )
    )
