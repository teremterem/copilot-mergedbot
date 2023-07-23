from pathlib import Path
from random import Random

from copilot.utils.misc import sort_paths
from copilot.utils.repo_access_utils import list_files_in_repo

REPO_PATH_IN_QUESTION = Path(__file__).parents[2] / "langchain"
LIST_CHUNK_SIZE = 100

rnd = Random(42)


def list_files_in_specific_repo() -> list[Path]:
    return sort_paths(list_files_in_repo(REPO_PATH_IN_QUESTION))


def list_files_in_specific_repo_chunked(chunk_size: int = LIST_CHUNK_SIZE) -> list[list[Path]]:
    files = list_files_in_specific_repo()
    rnd.shuffle(files)

    chunks = []
    for i in range(0, len(files), chunk_size):
        chunks.append(sort_paths(files[i : i + chunk_size]))

    return chunks


def print_chunks() -> None:
    for chunk in list_files_in_specific_repo_chunked():
        print()
        for file in chunk:
            print(file.as_posix())
        print()
