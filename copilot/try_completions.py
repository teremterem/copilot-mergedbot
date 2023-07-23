from pathlib import Path
from typing import Iterable

from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from copilot.utils.cached_completions import RepoCompletions
from copilot.utils.repo_access_utils import list_files_in_repo
from copilot.utils.misc import FAST_GPT_MODEL, FAST_LONG_GPT_MODEL, SLOW_GPT_MODEL

EXPLAIN_FILE_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("Here is the content of `{file_path}`:"),
        HumanMessagePromptTemplate.from_template("{file_content}"),
        SystemMessagePromptTemplate.from_template("Please explain the content of this file in plain English."),
    ]
)
REPO_PATH = Path(__file__).parents[2] / "langchain"

gpt3_explainer = RepoCompletions(
    repo=REPO_PATH,
    completion_name="gpt3-expl",
    model=FAST_GPT_MODEL,
)
gpt3_long_explainer = RepoCompletions(
    repo=REPO_PATH,
    completion_name="gpt3-long-expl",
    model=FAST_LONG_GPT_MODEL,
)
gpt4_explainer = RepoCompletions(
    repo=REPO_PATH,
    completion_name="gpt4-expl",
    model=SLOW_GPT_MODEL,
)


async def main() -> None:
    repo_files = [
        f for f in list_files_in_repo(REPO_PATH, additional_gitignore_content="tests/") if f.suffix.lower() == ".py"
    ]
    print()
    for file in repo_files:
        print(file)
    print()
    print(len(repo_files))
    print()

    # file = "autogpt/core/planning/simple.py"
    # messages = EXPLAIN_FILE_PROMPT.format_messages(
    #     file_path=file,
    #     file_content=(REPO_PATH / file).read_text(encoding="utf-8"),
    # )
    # messages = [convert_lc_message_to_openai(m) for m in messages]
    # await print_explanation(gpt3_explainer, messages, file)
    # await print_explanation(gpt3_long_explainer, messages, file)
    # await print_explanation(gpt4_explainer, messages, file)


async def print_explanation(explainer: RepoCompletions, messages: Iterable[dict[str, str]], file: Path | str) -> None:
    print("====================================================================================================")
    print()
    print(await explainer.chat_completion_for_file(messages=messages, repo_file=file))
    print()
    print(explainer.model)
    print()
