# pylint: disable=no-name-in-module
from typing import List

from botmerger import SingleTurnContext, MergedMessage, MergedBot
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate

from copilot.utils.misc import bot_merger, FAST_GPT_MODEL, reliable_chat_completion, langchain_messages_to_openai

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


def get_role_name(message: MergedMessage, this_bot: MergedBot) -> str:
    return "assistant" if message.sender == this_bot else "user"


async def get_relevant_history(
    request: MergedMessage, this_bot: MergedBot, include_request: bool = False, history_max_length: int = 20
) -> List[MergedMessage]:
    # TODO move this to `utils` package
    history = await request.get_conversation_history(max_length=history_max_length)

    if history:
        chat_history_parts = [
            f"[{i}] {get_role_name(msg, this_bot).upper()}: {msg.content}" for i, msg in enumerate(history, start=1)
        ]
        chat_history = "\n\n".join(chat_history_parts)
        current_message = f"[CURRENT] {get_role_name(request, this_bot).upper()}: {request.content}"
        filter_prompt = CHAT_HISTORY_FILTER_PROMPT.format_messages(
            chat_history=chat_history,
            current_message=current_message,
        )
        filter_prompt = langchain_messages_to_openai(filter_prompt)
        message_indices_to_keep = await reliable_chat_completion(
            model=FAST_GPT_MODEL,
            temperature=0,
            pl_tags=["chat_history_filter"],
            messages=filter_prompt,
        )
        # `message_indices_to_keep` contains a string with numbers in it. We need to extract those numbers.
        message_indices_to_keep = [int(s) - 1 for s in message_indices_to_keep.split() if s.isdigit()]

        message_indices_to_keep.sort(reverse=True)
        for idx in message_indices_to_keep:
            del history[idx]

    if include_request:
        history.append(request)
    return history


@bot_merger.create_bot("SimpleConversationBot")
async def simple_conversation(context: SingleTurnContext) -> None:
    conversation = [
        {
            "role": get_role_name(msg, context.this_bot),
            "content": msg.content,
        }
        for msg in await get_relevant_history(context.concluding_request, context.this_bot, include_request=True)
    ]
    completion = await reliable_chat_completion(
        model=FAST_GPT_MODEL,
        temperature=0,
        pl_tags=["simple_conversation"],
        messages=conversation,
    )
    await context.yield_final_response(completion)


main_bot = simple_conversation.bot
