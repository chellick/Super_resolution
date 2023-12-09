import asyncio 
import logging
import sys
import glob
from aiogram import F
from aiogram import Bot, Dispatcher, Router, types
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.types import Message
from aiogram.utils.markdown import hbold
from aiogram.fsm.context import FSMContext


with open("tgbot/TOKEN.txt") as f:
    TOKEN = f.read()

dp = Dispatcher()
sp = glob.glob("tgbot/temp")

bot = Bot(TOKEN, parse_mode=ParseMode.HTML)


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    await message.answer(f"Hello, {hbold(message.from_user.full_name)}!")
    

@dp.message(F.photo)
async def download_photo(message: Message):
    print(message.photo[-1])
    await bot.download(
            message.photo[-1],
            destination=f"C:/python/GitHub/Super_resolution/tgbot/temp/x_train.png"
        )
    


async def main() -> None:
    bot = Bot(TOKEN, parse_mode=ParseMode.HTML)
    await dp.start_polling(bot)
    



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout) # for sys upds.
    asyncio.run(main())

