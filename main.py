# pylint: disable=wrong-import-position
import asyncio

from dotenv import load_dotenv

load_dotenv()

# from copilot.explain_repo import main
from copilot.try_completion import main

if __name__ == "__main__":
    asyncio.run(main())
