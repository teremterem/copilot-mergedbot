# pylint: disable=no-name-in-module
import re
from typing import List

from botmerger import MergedMessage, MergedBot
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate

from copilot.utils.misc import (
    FAST_GPT_MODEL,
    reliable_chat_completion,
    langchain_messages_to_openai,
    get_openai_role_name,
)

CHAT_HISTORY_FILTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        HumanMessagePromptTemplate.from_template(
            """\
Here is a conversation history where each utterance has a number assigned to it (in brackets).

{chat_history}\
"""
        ),
        HumanMessagePromptTemplate.from_template(
            """\
And here is the current message (the one that goes right after the chat history).

{current_message}\
"""
        ),
        SystemMessagePromptTemplate.from_template(
            """\
Please select the numbers of the utterances which are important in relation to the current message and need to be \
kept. DO NOT EXPLAIN ANYTHING, JUST LIST THE NUMBERS.\
"""
        ),
    ]
)


async def get_relevant_history(
    request: MergedMessage, this_bot: MergedBot, include_request: bool = False, history_max_length: int = 20
) -> List[MergedMessage]:
    history = await request.get_conversation_history(max_length=history_max_length)

    if history:
        chat_history_parts = [
            f"[{i}] {get_openai_role_name(msg, this_bot).upper()}: {msg.content}"
            for i, msg in enumerate(history, start=1)
        ]
        chat_history = "\n\n".join(chat_history_parts)
        current_message = f"[CURRENT] {get_openai_role_name(request, this_bot).upper()}: {request.content}"
        filter_prompt = CHAT_HISTORY_FILTER_PROMPT.format_messages(
            chat_history=chat_history,
            current_message=current_message,
        )
        filter_prompt = langchain_messages_to_openai(filter_prompt)
        message_numbers_to_keep = await reliable_chat_completion(
            model=FAST_GPT_MODEL,
            temperature=0,
            pl_tags=["chat_history_filter"],
            messages=filter_prompt,
        )
        message_numbers_to_keep = [int(n) for n in re.findall(r"\d+", message_numbers_to_keep)]
        history = [msg for i, msg in enumerate(history, start=1) if i in message_numbers_to_keep]

    if include_request:
        history.append(request)
    return history
