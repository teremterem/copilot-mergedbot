from langchain.schema import BaseMessage

FAST_GPT_MODEL = "gpt-3.5-turbo-0613"
FAST_LONG_GPT_MODEL = "gpt-3.5-turbo-16k-0613"
SLOW_GPT_MODEL = "gpt-4-0613"


def convert_lc_message_to_openai(message: BaseMessage) -> dict[str, str]:
    if message.type == "human":
        role = "user"
    elif message.type == "ai":
        role = "assistant"
    elif message.type == "system":
        role = "system"
    else:
        raise ValueError(f"Unexpected message type: {message.type}")
    return {"role": role, "content": message.content}
