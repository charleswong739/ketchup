import os
import sys

from dotenv import load_dotenv

from discord import Intents, Message, Member
from discord.ext.commands import Bot, Context
from discord.user import User

from ketchup_model import KetchupModel

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

intents = Intents.default()
intents.message_content = True

bot = Bot(command_prefix="!", intents=intents)
model = KetchupModel()


def collate_messages(messages: list[Message]):
    messages_str = [f"{message.author.name}: {message.content}" for message in messages]

    # for message in messages:
    #     if message.content != "":
    #         print(type(message.author), file=sys.stderr)
    #         if type(message.author) is Member:
    #             name = message.author.nick
    #         elif type(message.author) is User:
    #             name = message.author.name
    #         else:
    #             name = "Anon"
    #
    #         messages_str.append(f"{name}: {message.content}")

    return "\n".join(messages_str)


@bot.command()
async def summarize(ctx: Context, arg: int=20):
    async with ctx.channel.typing():
        command_datetime = ctx.message.created_at
        messages = [message async for message in ctx.channel.history(limit=arg, before=command_datetime)]

        transcript = collate_messages(messages)
        print(transcript, file=sys.stderr)
        summary = model.summarize(transcript)
        print(summary, file=sys.stderr)

    await ctx.send(summary)

bot.run(DISCORD_TOKEN)