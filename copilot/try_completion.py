from copilot.specific_repo import REPO_PATH_IN_QUESTION
from copilot.utils.cached_completions import RepoCompletions
from copilot.utils.misc import EMBEDDING_MODEL

ada_embedder = RepoCompletions(
    repo=REPO_PATH_IN_QUESTION,
    completion_name="ada2",
    model=EMBEDDING_MODEL,
)


async def main() -> None:
    # repo_files = list_files_in_specific_repo_chunked(reduced_list=True)[0]
    # print()
    # for file in repo_files:
    #     print(file)
    # print()
    # print(len(repo_files))
    # print()

    # file = "libs/experimental/langchain_experimental/cpal/base.py"
    file = "libs/langchain/langchain/text_splitter.py"

    # explanation = await explain_repo_file_in_isolation(file)
    # print()
    # print(explanation)
    # print()

    embedding = await ada_embedder.file_related_embedding(
        content=(REPO_PATH_IN_QUESTION / file).read_text(encoding="utf-8"), repo_file=file
    )
    print()
    print(embedding)
    print()
