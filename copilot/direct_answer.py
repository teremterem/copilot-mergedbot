# pylint: disable=no-name-in-module
import json

import faiss
import numpy as np
from botmerger import SingleTurnContext
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import HumanMessage
from promptlayer import openai

from copilot.explain_repo import explain_repo_file_in_isolation
from copilot.specific_repo import REPO_PATH_IN_QUESTION
from copilot.utils.misc import (
    SLOW_GPT_MODEL,
    bot_merger,
    EMBEDDING_MODEL,
    langchain_messages_to_openai,
    reliable_chat_completion,
)

DIRECT_ANSWER_PROMPT_PREFIX = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            """\
You are a chatbot that is good at answering questions about the concepts that can be found in the repository by the \
name `{repo_name}`.

Below are the summaries of the source code files from `{repo_name}` repo which may or may not be relevant to the \
request that came from the user (the request itself will be provided later).\
"""
        ),
    ]
)
DIRECT_ANSWER_PROMPT_SUFFIX = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("Now, respond to the following request that came from a user."),
        HumanMessagePromptTemplate.from_template("{request}"),
    ]
)

EXPLANATIONS_FAISS = faiss.read_index(str(REPO_PATH_IN_QUESTION / "explanations.faiss"))
INDEXED_EXPL_FILES = json.loads((REPO_PATH_IN_QUESTION / "explanation_files.json").read_text(encoding="utf-8"))


@bot_merger.create_bot("DirectAnswerBot")
async def direct_answer(context: SingleTurnContext) -> None:
    # pylint: disable=too-many-locals
    user_request = context.concluding_request.content
    result = await openai.Embedding.acreate(input=[user_request], model=EMBEDDING_MODEL, temperature=0)
    embedding = result["data"][0]["embedding"]
    scores, indices = EXPLANATIONS_FAISS.search(np.array([embedding], dtype=np.float32), 20)

    recalled_files_msg = "\n".join(
        f"{score:.2f} {INDEXED_EXPL_FILES[idx]}" for score, idx in zip(scores[0], indices[0])
    )
    await context.yield_interim_response(f"```\n{recalled_files_msg}\n```")

    prompt_prefix = DIRECT_ANSWER_PROMPT_PREFIX.format_messages(repo_name=REPO_PATH_IN_QUESTION.name)
    recalled_files = [
        HumanMessage(content=await explain_repo_file_in_isolation(file=INDEXED_EXPL_FILES[idx])) for idx in indices[0]
    ]
    prompt_suffix = DIRECT_ANSWER_PROMPT_SUFFIX.format_messages(request=user_request)

    prompt = [*prompt_prefix, *recalled_files, *prompt_suffix]
    prompt_openai = langchain_messages_to_openai(prompt)

    completion = await reliable_chat_completion(
        model=SLOW_GPT_MODEL,
        temperature=0,
        pl_tags=["direct_answer"],
        messages=prompt_openai,
    )
    await context.yield_final_response(completion)


main_bot = direct_answer.bot
