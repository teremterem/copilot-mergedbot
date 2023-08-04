# pylint: disable=wrong-import-position
"""Main file for running the Discord bot."""
import os

import discord
from botmerger.ext.discord_integration import attach_bot_to_discord

from copilot.direct_answer import main_bot

DISCORD_BOT_SECRET = os.environ["DISCORD_BOT_SECRET"]

discord_client = discord.Client(intents=discord.Intents.default())


@discord_client.event
async def on_ready() -> None:
    """Called when the client is done preparing the data received from Discord."""
    print("Logged in as", discord_client.user)
    print()


def main() -> None:
    attach_bot_to_discord(main_bot, discord_client)
    discord_client.run(DISCORD_BOT_SECRET)
