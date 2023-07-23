from pathlib import Path
from random import Random

from copilot.utils.misc import sort_paths
from copilot.utils.repo_access_utils import list_files_in_repo

REPO_PATH_IN_QUESTION = Path(__file__).parents[2] / "langchain"

rnd = Random(42)


def list_files_in_specific_repo() -> list[Path]:
    return sort_paths(list_files_in_repo(REPO_PATH_IN_QUESTION))


def print_chunks() -> None:
    files = list_files_in_specific_repo()
    rnd.shuffle(files)

    chunk_size = 100
    for i in range(0, len(files), chunk_size):
        chunk = sort_paths(files[i : i + chunk_size])
        print()
        for file in chunk:
            print(file.as_posix())
        print()
