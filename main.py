# pylint: disable=wrong-import-position
import asyncio

from dotenv import load_dotenv

load_dotenv()

from copilot.try_completions import main

if __name__ == "__main__":
    asyncio.run(main())
