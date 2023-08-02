# pylint: disable=no-name-in-module
from pathlib import Path

from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate

from copilot.specific_repo import REPO_PATH_IN_QUESTION
from copilot.utils.misc import (
    reliable_chat_completion,
    langchain_messages_to_openai,
    FAST_LONG_GPT_MODEL,
)

FILE_SNIPPETS_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "Here is the content of `{file_path}`, a file from the `{repo_name}` repo:"
        ),
        HumanMessagePromptTemplate.from_template("{file_content}"),
        SystemMessagePromptTemplate.from_template("And here is a request from the user."),
        HumanMessagePromptTemplate.from_template("{user_request}"),
        SystemMessagePromptTemplate.from_template(
            """\
Output the content of the file `{file_path}` verbatim omitting parts that are not relevant to the user's request. If \
nothing in this file is relevant to the conversation, output just one word: NONE

DO NOT COME UP WITH YOUR OWN CODE, ONLY OUTPUT THE CODE THAT YOU SEE IN THE FILE!\
"""
            # # The following piece seems to lead to hallucinations:
            #
            # Make sure to include the implementation details into the snippets, though, because those details might \
            # be important for answering the request.\
            # """
        ),
    ]
)


async def extract_relevant_snippets(file: Path | str, standalone_request: str) -> str:
    if not isinstance(file, Path):
        file = Path(file)

    messages = FILE_SNIPPETS_PROMPT.format_messages(
        repo_name=REPO_PATH_IN_QUESTION.name,
        file_path=file,
        file_content=(REPO_PATH_IN_QUESTION / file).read_text(encoding="utf-8"),
        user_request=standalone_request,
    )
    messages = langchain_messages_to_openai(messages)

    completion = await reliable_chat_completion(
        model=FAST_LONG_GPT_MODEL,
        temperature=0,
        pl_tags=["extract_snippets"],
        messages=messages,
    )
    return f"FILE: {file.as_posix()}\n\n{completion}"
