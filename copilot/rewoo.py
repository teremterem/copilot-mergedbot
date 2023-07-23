from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

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
        SystemMessagePromptTemplate.from_template(
            "And here are the outlines of the source code files in `{repo_name}` repo:"
        ),
        HumanMessagePromptTemplate.from_template("{file_outlines}"),
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
