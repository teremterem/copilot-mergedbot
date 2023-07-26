import traceback
from pathlib import Path
from typing import Iterable

from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from copilot.specific_repo import REPO_PATH_IN_QUESTION, list_files_in_specific_repo_chunked
from copilot.utils.cached_completions import RepoCompletions
from copilot.utils.misc import FAST_GPT_MODEL, FAST_LONG_GPT_MODEL, SLOW_GPT_MODEL, convert_lc_message_to_openai

EXPLAIN_FILE_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "Here is the content of `{file_path}`, a file from the `{repo_name}` repo:"
        ),
        HumanMessagePromptTemplate.from_template("{file_content}"),
        SystemMessagePromptTemplate.from_template("Please explain the content of this file in plain English."),
    ]
)

gpt3_explainer = RepoCompletions(
    repo=REPO_PATH_IN_QUESTION,
    completion_name="gpt3-expl",
    model=FAST_GPT_MODEL,
)
gpt3_long_explainer = RepoCompletions(
    repo=REPO_PATH_IN_QUESTION,
    completion_name="gpt3-long-expl",
    model=FAST_LONG_GPT_MODEL,
)
gpt4_explainer = RepoCompletions(
    repo=REPO_PATH_IN_QUESTION,
    completion_name="gpt4-expl",
    model=SLOW_GPT_MODEL,
)


async def main() -> None:
    # pylint: disable=broad-exception-caught
    repo_files = list_files_in_specific_repo_chunked(reduced_list=True)[0]
    print()
    for file in repo_files:
        print(file)
    print()
    print(len(repo_files))
    print()

    failed_files = []
    for idx, file in enumerate(repo_files):
        try:
            print(idx, "-", file)
            messages = EXPLAIN_FILE_PROMPT.format_messages(
                repo_name=REPO_PATH_IN_QUESTION.name,
                file_path=file,
                file_content=(REPO_PATH_IN_QUESTION / file).read_text(encoding="utf-8"),
            )
            messages = [convert_lc_message_to_openai(m) for m in messages]
            await gpt3_explainer.chat_completion_for_file(messages=messages, repo_file=file)
        except Exception:
            traceback.print_exc()
            failed_files.append(file)

    if failed_files:
        print()
        print("FAILED FILES:")
        print()
        for file in failed_files:
            print(file)
        print()
        print(len(failed_files))
        print()

        files_that_failed_again = []
        for idx, file in enumerate(failed_files):
            try:
                print(idx, "-", file)
                messages = EXPLAIN_FILE_PROMPT.format_messages(
                    repo_name=REPO_PATH_IN_QUESTION.name,
                    file_path=file,
                    file_content=(REPO_PATH_IN_QUESTION / file).read_text(encoding="utf-8"),
                )
                messages = [convert_lc_message_to_openai(m) for m in messages]
                await gpt3_long_explainer.chat_completion_for_file(messages=messages, repo_file=file)
            except Exception:
                traceback.print_exc()
                files_that_failed_again.append(file)

        if files_that_failed_again:
            print()
            print("FAILED AGAIN FILES:")
            print()
            for file in files_that_failed_again:
                print(file)
            print()
            print(len(files_that_failed_again))
            print()

    print("DONE")


async def print_explanation(explainer: RepoCompletions, messages: Iterable[dict[str, str]], file: Path | str) -> None:
    print("====================================================================================================")
    print()
    print(await explainer.chat_completion_for_file(messages=messages, repo_file=file))
    print()
    print(explainer.model)
    print()
