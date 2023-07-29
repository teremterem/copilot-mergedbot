# pylint: disable=wrong-import-position
from dotenv import load_dotenv

load_dotenv()

# from copilot.try_completion import main
# from copilot.explain_repo import embed_everything
from copilot.discord_connector import main

if __name__ == "__main__":
    # asyncio.run(main())
    # asyncio.run(embed_everything())
    main()
