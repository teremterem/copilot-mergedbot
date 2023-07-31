# pylint: disable=no-name-in-module
import itertools

from botmerger import SingleTurnContext
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain.schema import HumanMessage

from copilot.explain_repo import explain_repo_file_in_isolation
from copilot.relevant_files import get_relevant_files
from copilot.specific_repo import REPO_PATH_IN_QUESTION
from copilot.utils.history_processors import get_filtered_conversation
from copilot.utils.misc import (
    SLOW_GPT_MODEL,
    bot_merger,
    langchain_messages_to_openai,
    reliable_chat_completion,
    get_openai_role_name,
)

DIRECT_ANSWER_PROMPT_PREFIX = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            """\
You are an AI assistant that is good at answering questions about the concepts that can be found in the repository \
by the name `{repo_name}`.

Below are the summaries of some source code files from `{repo_name}` repo which may or may not be relevant to the \
conversation that you are currently having with the user.\
"""
        ),
    ]
)
DIRECT_ANSWER_PROMPT_SUFFIX = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "Now, carry on with the conversation between you as an AI assistant and the user."
        ),
    ]
)


@bot_merger.create_bot("DirectAnswerBot")
async def direct_answer(context: SingleTurnContext) -> None:
    conversation = await get_filtered_conversation(context.concluding_request, context.this_bot)

    relevant_files = await get_relevant_files(conversation, context.this_bot)
    recalled_files_msg = "\n".join(f"{file}" for file in relevant_files)
    await context.yield_interim_response(f"```\n{recalled_files_msg}\n```", invisible_to_bots=True)

    prompt_prefix = DIRECT_ANSWER_PROMPT_PREFIX.format_messages(repo_name=REPO_PATH_IN_QUESTION.name)
    recalled_files = [HumanMessage(content=await explain_repo_file_in_isolation(file=file)) for file in relevant_files]
    prompt_suffix = DIRECT_ANSWER_PROMPT_SUFFIX.format_messages()

    prompt_openai = langchain_messages_to_openai(itertools.chain(prompt_prefix, recalled_files, prompt_suffix))
    prompt_openai.extend(
        {
            "role": get_openai_role_name(msg, context.this_bot),
            "content": msg.content,
        }
        for msg in conversation
    )

    completion = await reliable_chat_completion(
        model=SLOW_GPT_MODEL,
        temperature=0,
        pl_tags=["direct_answer"],
        messages=prompt_openai,
    )
    await context.yield_final_response(completion)


main_bot = direct_answer.bot
