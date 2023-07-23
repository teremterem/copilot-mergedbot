# pylint: disable=wrong-import-position
from dotenv import load_dotenv

load_dotenv()

from copilot.discord_connector import main

if __name__ == "__main__":
    main()
