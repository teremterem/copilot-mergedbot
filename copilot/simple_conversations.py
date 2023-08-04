from botmerger import SingleTurnContext

from copilot.chat_history_filter import get_filtered_conversation
from copilot.utils.misc import bot_merger, FAST_GPT_MODEL, reliable_chat_completion, get_openai_role_name


@bot_merger.create_bot("SimpleConversationBot")
async def simple_conversation(context: SingleTurnContext) -> None:
    conversation = [
        {
            "role": get_openai_role_name(msg, context.this_bot),
            "content": msg.content,
        }
        for msg in await get_filtered_conversation(context.concluding_request, context.this_bot)
    ]
    completion = await reliable_chat_completion(
        model=FAST_GPT_MODEL,
        temperature=0,
        pl_tags=["simple_conversation"],
        messages=conversation,
    )
    await context.yield_final_response(completion)


main_bot = simple_conversation.bot
