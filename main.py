# pylint: disable=wrong-import-position
from dotenv import load_dotenv

load_dotenv()

# from copilot.discord_connector import main
from copilot.specific_repo import print_chunks

if __name__ == "__main__":
    # main()
    print_chunks()
