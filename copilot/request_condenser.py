# pylint: disable=no-name-in-module

from botmerger import SingleTurnContext
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate

from copilot.utils.misc import (
    FAST_GPT_MODEL,
    reliable_chat_completion,
    langchain_messages_to_openai,
    format_conversation_for_single_message,
    bot_merger,
    CHAT_HISTORY_MAX_LENGTH,
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


@bot_merger.create_bot("RequestCondenserBot")
async def request_condenser(context: SingleTurnContext) -> None:
    # TODO is relying on `original_message` to get the full conversation a good idea ?
    conversation = await context.concluding_request.original_message.get_full_conversation(
        max_length=CHAT_HISTORY_MAX_LENGTH
    )

    if len(conversation) < 2:
        await context.yield_final_response(context.concluding_request)
        return

    # TODO how to make it more clear why `this_bot` is `context.concluding_request.receiver` in this particular case ?
    chat_history = format_conversation_for_single_message(conversation, context.concluding_request.receiver)
    condenser_prompt = CONDENSED_QUESTION_PROMPT.format_messages(chat_history=chat_history)
    condenser_prompt = langchain_messages_to_openai(condenser_prompt)
    standalone_request = await reliable_chat_completion(
        model=FAST_GPT_MODEL,
        temperature=0,
        pl_tags=["question_condenser"],
        messages=condenser_prompt,
    )
    await context.yield_final_response(standalone_request)
