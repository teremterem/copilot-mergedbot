from pathlib import Path


def chat_completion_for_repo_file(prompt: str, repo: Path, repo_file: Path, completion_name: str) -> str:
    return prompt + " Hello World!"
