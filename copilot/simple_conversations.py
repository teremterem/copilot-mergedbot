# pylint: disable=no-name-in-module
from botmerger import SingleTurnContext
from promptlayer import openai

from copilot.utils.misc import SLOW_GPT_MODEL, bot_merger


@bot_merger.create_bot("SimpleConversationBot")
async def simple_conversation(context: SingleTurnContext) -> None:
    conversation = [
        {
            "role": "user",
            "content": req.content,
        }
        for req in context.requests
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
