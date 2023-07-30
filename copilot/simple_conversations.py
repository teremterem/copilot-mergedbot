# pylint: disable=no-name-in-module
from typing import List

from botmerger import SingleTurnContext, MergedMessage
from promptlayer import openai

from copilot.utils.misc import SLOW_GPT_MODEL, bot_merger


async def get_relevant_history(
    request: MergedMessage, include_request: bool = False, history_max_length: int = 20
) -> List[MergedMessage]:
    # TODO move this to `utils` package
    history = await request.get_conversation_history(max_length=history_max_length)
    if include_request:
        history.append(request)
    return history


@bot_merger.create_bot("SimpleConversationBot")
async def simple_conversation(context: SingleTurnContext) -> None:
    conversation = [
        {
            "role": "assistant" if msg.sender == context.this_bot else "user",
            "content": msg.content,
        }
        for msg in await get_relevant_history(context.concluding_request, include_request=True)
    ]
    gpt_response = await openai.ChatCompletion.acreate(
        model=SLOW_GPT_MODEL,
        temperature=0,
        pl_tags=["simple_conversation"],
        messages=conversation,
    )
    completion = gpt_response.choices[0]
    if completion.finish_reason != "stop":
        raise RuntimeError(f"Incomplete text completion (finish_reason: {completion.finish_reason})")
    await context.yield_final_response(completion.message.content)


main_bot = simple_conversation.bot
