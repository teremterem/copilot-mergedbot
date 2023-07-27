from copilot.explain_full_repo import EXPLAIN_FILE_PROMPT, gpt3_long_explainer
from copilot.specific_repo import REPO_PATH_IN_QUESTION, list_files_in_specific_repo_chunked
from copilot.utils.misc import convert_lc_message_to_openai


async def main() -> None:
    repo_files = list_files_in_specific_repo_chunked(reduced_list=True)[0]
    print()
    for file in repo_files:
        print(file)
    print()
    print(len(repo_files))
    print()

    file = "libs/experimental/langchain_experimental/cpal/base.py"
    messages = EXPLAIN_FILE_PROMPT.format_messages(
        repo_name=REPO_PATH_IN_QUESTION.name,
        file_path=file,
        file_content=(REPO_PATH_IN_QUESTION / file).read_text(encoding="utf-8"),
    )
    messages = [convert_lc_message_to_openai(m) for m in messages]
    explanation = await gpt3_long_explainer.chat_completion_for_file(messages=messages, repo_file=file)
    print(explanation)
    print()
