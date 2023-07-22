from pathlib import Path

from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from copilot.cached_completions import RepoCompletions
from copilot.repo_access_utils import list_files_in_repo
from copilot.utils import FAST_GPT_MODEL

repo_completions = RepoCompletions(
    repo=Path(__file__).parents[1],
    completion_name="helloworld",
    model=FAST_GPT_MODEL,
)

EXPLAIN_FILE_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("Here is the content of `{file_path}`:"),
        HumanMessagePromptTemplate.from_template("{file_content}"),
        SystemMessagePromptTemplate.from_template("Please explain the content of this file in plain English."),
    ]
)


async def main() -> None:
    # messages = [{"role": "assistant", "content": "Привіт world!"}]
    # print(await repo_completions.chat_completion_for_file(messages=messages, repo_file=Path(__file__)))

    repo_files = [
        f
        for f in list_files_in_repo(Path(__file__).parents[2] / "Auto-GPT", additional_gitignore_content="tests/")
        if f.suffix.lower() == ".py"
    ]
    print()
    for file in repo_files:
        print(file)
    print()
    print(len(repo_files))
    print()
