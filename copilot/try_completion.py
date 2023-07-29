from copilot.explain_repo import explain_repo_file_in_isolation


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

    result = await explain_repo_file_in_isolation(file)
    # result = await ada_embedder.file_related_embedding(result, repo_file=file)

    print()
    print(result)
    print()
