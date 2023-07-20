from pathlib import Path

from copilot.cached_completions import RepoCompletions
from copilot.utils import FAST_GPT_MODEL

repo_completions = RepoCompletions(
    repo=Path(__file__).parent,
    completion_name="helloworld",
    model=FAST_GPT_MODEL,
)


async def main() -> None:
    messages = [{"role": "assistant", "content": "Привіт world!"}]
    print(await repo_completions.chat_completion_for_file(messages=messages, repo_file=Path(__file__)))
