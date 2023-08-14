from botmerger import SingleTurnContext

from copilot.chat_history_filter import chat_history_filter
from copilot.utils.misc import bot_merger, FAST_GPT_MODEL, reliable_chat_completion, get_openai_role_name


@bot_merger.create_bot
async def simple_conversation(context: SingleTurnContext) -> None:
    conversation = await chat_history_filter.bot.get_all_responses(context.concluding_request)
    conversation = [
        {
            "role": get_openai_role_name(msg.original_message, context.this_bot),
            "content": msg.content,
        }
        for msg in conversation
    ]
    completion = await reliable_chat_completion(
        model=FAST_GPT_MODEL,
        temperature=0,
        pl_tags=["simple_conversation"],
        messages=conversation,
    )
    await context.yield_final_response(completion)


main_bot = simple_conversation.bot
