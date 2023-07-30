# pylint: disable=no-name-in-module
from pathlib import Path

from botmerger import MergedMessage, MergedBot
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate

from copilot.specific_repo import REPO_PATH_IN_QUESTION
from copilot.utils.history_processors import format_conversation_for_single_message, get_filtered_conversation
from copilot.utils.misc import (
    reliable_chat_completion,
    langchain_messages_to_openai,
    FAST_GPT_MODEL,
)

FILE_SNIPPETS_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "Here is the content of `{file_path}`, a file from the `{repo_name}` repo:"
        ),
        HumanMessagePromptTemplate.from_template("{file_content}"),
        SystemMessagePromptTemplate.from_template(
            "And here is a conversation between you as an AI assistant and the user."
        ),
        HumanMessagePromptTemplate.from_template("{chat_history}"),
        SystemMessagePromptTemplate.from_template(
            """\
Output the content of the file `{file_path}` verbatim omitting parts that are not relevant to the conversation. If \
nothing in this file is relevant to the conversation, output just one word: NONE\
"""
        ),
    ]
)


async def extract_relevant_snippets(file: Path | str, request: MergedMessage, this_bot: MergedBot) -> str:
    if not isinstance(file, Path):
        file = Path(file)

    conversation = await get_filtered_conversation(request, this_bot)
    chat_history = format_conversation_for_single_message(conversation, this_bot)

    messages = FILE_SNIPPETS_PROMPT.format_messages(
        repo_name=REPO_PATH_IN_QUESTION.name,
        file_path=file,
        file_content=(REPO_PATH_IN_QUESTION / file).read_text(encoding="utf-8"),
        chat_history=chat_history,
    )
    messages = langchain_messages_to_openai(messages)

    completion = await reliable_chat_completion(
        model=FAST_GPT_MODEL,
        temperature=0,
        pl_tags=["direct_answer"],
        messages=messages,
    )
    return completion
