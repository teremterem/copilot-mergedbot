# pylint: disable=no-name-in-module

from botmerger import MergedMessage, MergedBot
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate

from copilot.utils.misc import (
    FAST_GPT_MODEL,
    reliable_chat_completion,
    langchain_messages_to_openai,
    format_conversation_for_single_message,
)

CONDENSED_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            """\
Given the following conversation extract a standalone request that the user is trying to make to the AI assistant. \
Make sure the standalone request that you generate contains all the necessary details. Try to make sure it \
reflects what the user really wants (users may sometimes be somewhat indirect when they talk to AI assistants). \
Also, if possible, please try to formulate the request in the form of a question.\
"""
        ),
        HumanMessagePromptTemplate.from_template(
            """\
# CHAT HISTORY

{chat_history}

# STANDALONE REQUEST

USER:\
"""
        ),
    ]
)


async def get_standalone_request(request: MergedMessage, this_bot: MergedBot, history_max_length: int = 20) -> str:
    conversation = await request.get_full_conversation(max_length=history_max_length)

    if len(conversation) < 2:
        return request.content

    chat_history = format_conversation_for_single_message(conversation, this_bot)
    condenser_prompt = CONDENSED_QUESTION_PROMPT.format_messages(chat_history=chat_history)
    condenser_prompt = langchain_messages_to_openai(condenser_prompt)
    standalone_request = await reliable_chat_completion(
        model=FAST_GPT_MODEL,
        temperature=0,
        pl_tags=["question_condenser"],
        messages=condenser_prompt,
    )
    return standalone_request
