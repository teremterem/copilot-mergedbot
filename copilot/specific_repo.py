from pathlib import Path

from copilot.utils.repo_access_utils import list_files_in_repo

REPO_PATH_IN_QUESTION = Path(__file__).parents[2] / "langchain"


def list_files_in_specific_repo() -> list[Path]:
    return list_files_in_repo(REPO_PATH_IN_QUESTION)
