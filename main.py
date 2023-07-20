from pathlib import Path

from copilot.cached_completions import chat_completion_for_repo_file

if __name__ == "__main__":
    print(chat_completion_for_repo_file("Привіт world!", Path(__file__).parent, Path(__file__), "helloworld"))
