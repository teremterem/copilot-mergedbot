# pylint: disable=no-name-in-module
import json
from pathlib import Path

import faiss
import numpy as np
from botmerger import SingleTurnContext, BotResponses
from langchain import LLMChain
from langchain.chat_models import PromptLayerChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from promptlayer import openai

from copilot.specific_repo import REPO_PATH_IN_QUESTION, list_files_in_specific_repo_chunked
from copilot.utils.misc import SLOW_GPT_MODEL, bot_merger, EMBEDDING_MODEL

REWOO_PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            """\
You are a chatbot that is good at analysing the code in the following repository and answering questions about the \
concepts that can be found in this repository.

Repository name: {repo_name}
List of files in the repository:\
"""
        ),
        HumanMessagePromptTemplate.from_template("{file_list}"),
        # SystemMessagePromptTemplate.from_template(
        #     "And here are the outlines of the source code files in `{repo_name}` repo:"
        # ),
        # HumanMessagePromptTemplate.from_template("{file_outlines}"),
        SystemMessagePromptTemplate.from_template(
            """\
For the following tasks, make plans that can solve the problem step-by-step. For each plan, indicate which external \
tool together with tool input to retrieve evidence. You can store the evidence into a variable that can be called \
by later tools.

Here is the expected format of your response:\
"""
        ),
        SystemMessagePromptTemplate.from_template(
            """\
{{
    "evidence1": {{
        "plan": "explanation of a step of the plan",
        "tool": "Tool1",
        "tool_input": "free form text",
        "context": []
    }},
    "evidence2": {{
        "plan": "explanation of a step of the plan",
        "tool": "Tool2",
        "tool_input": "free form text",
        "context": []
    }},
    "evidence3": {{
        "plan": "explanation of a step of the plan",
        "tool": "Tool1",
        "tool_input": "free form text",
        "context": ["evidence2"]
    }},
    "evidence4": {{
        "plan": "explanation of a step of the plan",
        "tool": "Tool3",
        "tool_input": "free form text",
        "context": ["evidence1", "evidence3"]
    }}
}}\
"""
        ),
        #         SystemMessagePromptTemplate.from_template(
        #             """\
        # Tools can be one of the following:
        #
        # {tools}
        #
        # Begin! Describe your plans with rich details. RESPOND WITH VALID JSON ONLY AND NO OTHER TEXT.\
        # """
        #         ),
        SystemMessagePromptTemplate.from_template(
            "Begin! Describe your plans with rich details. RESPOND WITH VALID JSON ONLY AND NO OTHER TEXT."
        ),
        HumanMessagePromptTemplate.from_template("{request}"),
    ]
)

EXPLANATIONS_FAISS = faiss.read_index(str(REPO_PATH_IN_QUESTION / "explanations.faiss"))
INDEXED_EXPL_FILES = json.loads((REPO_PATH_IN_QUESTION / "explanation_files.json").read_text(encoding="utf-8"))


@bot_merger.create_bot("ReWOO")
async def rewoo(context: SingleTurnContext) -> None:
    # pylint: disable=too-many-locals
    user_request = context.concluding_request.content
    result = await openai.Embedding.acreate(input=[user_request], model=EMBEDDING_MODEL, temperature=0)
    embedding = result["data"][0]["embedding"]
    scores, indices = EXPLANATIONS_FAISS.search(np.array([embedding], dtype=np.float32), 10)

    recalled_files_msg = "\n".join(
        f"{score:.2f} {INDEXED_EXPL_FILES[idx]}" for score, idx in zip(scores[0], indices[0])
    )
    await context.yield_final_response(f"```\n{recalled_files_msg}\n```")
    return

    file_list: list[Path] = list_files_in_specific_repo_chunked()[0]
    file_list_str = "\n".join(f.as_posix() for f in file_list)

    chat_llm = PromptLayerChatOpenAI(
        model_name=SLOW_GPT_MODEL,
        temperature=0.5,
        pl_tags=["rewoo_planner"],
    )
    llm_chain = LLMChain(
        llm=chat_llm,
        prompt=REWOO_PLANNER_PROMPT,
    )
    # rewoo_tools = (
    #     explain_file_bot.bot,
    #     generate_file_outline.bot,
    #     read_file_bot.bot,
    #     rewoo.bot,
    #     simpler_llm.bot,
    # )
    generated_plan = json.loads(
        await llm_chain.arun(
            repo_name=REPO_PATH_IN_QUESTION.name,
            file_list=file_list_str,
            # file_outlines="\n\n\n".join(get_botmerger_outlines()),
            # tools="\n\n".join([f"{bot.alias}[input]: {bot.description}" for bot in rewoo_tools]),
            request=user_request,
        )
    )
    await context.yield_final_response(generated_plan)

    promises: dict[str, BotResponses] = {}
    for evidence_id, plan in generated_plan.items():
        bot = await bot_merger.find_bot(plan["tool"])
        plan_context = [promises[previous_evidence_id] for previous_evidence_id in plan["context"]]
        promises[evidence_id] = await bot.trigger(requests=[*plan_context, plan["tool_input"]])

    for idx, (evidence_id, responses) in enumerate(promises.items()):
        await context.yield_interim_response(f"```\n{evidence_id}\n```")
        await context.yield_from(responses, still_thinking=True if idx < len(promises) - 1 else None)


main_bot = rewoo.bot
