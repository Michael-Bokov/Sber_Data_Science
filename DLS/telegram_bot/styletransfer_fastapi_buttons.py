#import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import certifi
import os
os.environ['SSL_CERT_FILE'] = certifi.where()
if os.name == 'nt':
    os.system('chcp 65001')
else:
    sys.stdout.reconfigure(encoding='utf-8')
import torchvision.transforms as transforms
from torchvision import models

from PIL import Image
import copy
#import requests

import logging
import gdown
import time

from io import BytesIO

# Функция для загрузки изображения и преобразования его в тензор
def load_image(img_path, max_size=512, shape=None):
    image = Image.open(img_path).convert('RGB')
    size = max_size if max(image.size) > max_size else max(image.size)
    if shape is not None:
        size = shape
    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    h,w=image.size
    image = in_transform(image).unsqueeze(0)
    return image.requires_grad_(True),h,w

# Функция для вычисления матрицы грамма
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# Класс для потерь содержания
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = nn.functional.mse_loss

    def forward(self, input):
        self.loss_value = self.loss(input, self.target)
        return input

# Класс для потерь стиля
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = nn.functional.mse_loss

    def forward(self, input):
        G = gram_matrix(input)
        self.loss_value = self.loss(G, self.target)
        return input

# Модифицированная модель VGG19
class ModifiedVGG19(nn.Module):
    def __init__(self):
        super(ModifiedVGG19, self).__init__()
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        #vgg19 = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.features = nn.Sequential(*list(vgg19.children()))

    def forward(self, x):
        return self.features(x)

# Функция для создания модели и вычисления потерь
def get_style_model_and_losses(cnn, style_img, content_img, content_layers, style_layers):
    cnn = copy.deepcopy(cnn)
    model = nn.Sequential()
    content_losses = []
    style_losses = []

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

# Основная функция переноса стиля
def run_style_transfer(cnn, content_img, style_img, input_img, num_steps=50,
                      style_weight=1000, content_weight=100):
    #print('2')
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, style_img, content_img, content_layers=['conv_4'], style_layers=['conv_1', 'conv_2','conv_3','conv_4','conv_5']
    )
    input_img = input_img.clone().detach().requires_grad_(True)
    optimizer = optim.LBFGS([input_img])

    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} -Starting style transfer...")
    


    run = [0]
    while run[0] <= num_steps:
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = torch.tensor(0., device=input_img.device)
            content_score = torch.tensor(0., device=input_img.device)

            for sl in style_losses:
                style_score += sl.loss_value
            for cl in content_losses:
                content_score += cl.loss_value

            loss = style_score * style_weight + content_score * content_weight
            loss.backward()

            if run[0] % 50 == 0:
                print(f"Step {run[0]}:")
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} Style Loss: {style_score.item()} Content Loss: {content_score.item()}")

            run[0] += 1
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            input_img.clamp_(0, 1)
        #input_img.data.clamp_(0, 1)

    return input_img


from telegram import Bot, Update, InputMediaPhoto, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import Updater,Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes
from telegram.ext import ApplicationBuilder
import telegram.ext.filters as filters

import requests
from fastapi import FastAPI, Request
from pydantic import BaseModel
import threading
import os
import asyncio
import aiohttp
import aiofiles
from io import BytesIO
from PIL import Image
import uvicorn
from fastapi.responses import JSONResponse
from telegram import Bot, Update, InputMediaPhoto, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CommandHandler, CallbackQueryHandler, MessageHandler, filters, ApplicationBuilder
import telegram.ext.filters as filters

from telegram.error import NetworkError
import torch
import logging

from torchvision import transforms, models
import torchvision.transforms as transforms

# Global variables and configurations
global TOKEN, WEBHOOK_URL, bot, WEBHOOK_SET, style_image_path,bot_data
TOKEN = "TOKEN"
WEBHOOK_PATH = "/webhook"
WEBHOOK_URL = "https://YOUR DOMAIN NAME" + WEBHOOK_PATH
style_image_path = "styles/sty.jpg"
bot = None
WEBHOOK_SET = False
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO, handlers=[
        logging.FileHandler('bot_fastapi.log', encoding='utf-8')])
logger = logging.getLogger(__name__)

main_app = FastAPI()
update_queue = []
# Global state
bot_data = {'expecting_style_image': False}

# Function to preprocess image
async def preprocess(update, img_path, max_size=512, shape=None):
    logger.info("Preprocessing image")
    async with aiofiles.open(img_path, mode='rb') as file:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
    size = max_size if max(image.size) > max_size else max(image.size)
    if shape is not None:
        size = shape
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor.requires_grad_(True)

# Function to resize image
def resize_image(image_path, max_size=512):
    image = Image.open(image_path)
    image.thumbnail((max_size, max_size), Image.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer

# Function to handle /start command
async def start(update):
    logger.info('Получена команда /start')
    example_input_path = 'input.jpg'
    example_output_path = 'output_image.jpg'
    example_input = InputMediaPhoto(resize_image(example_input_path), caption='Input')
    example_output = InputMediaPhoto(resize_image(example_output_path), caption='Output')
    keyboard = [
        [InlineKeyboardButton("Set Style Image", callback_data='set_style')],
        [InlineKeyboardButton("Set Default Style", callback_data='set_default_style')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_media_group([example_input, example_output])
    await update.message.reply_text('И да, лучше конечно пейзажные файлы. примеры выше)', reply_markup=reply_markup)

# Function to handle /help command
async def help_command(update):
    logger.info('Получена команда /help')
    await update.message.reply_text('Абаут')

# Function to process and send image
async def process_and_send_image(update, img_path):
    global style_image_path, cnn,bot_data
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    input_tensor = await preprocess(update, img_path)
    logger.info(f"Image received and preprocessed")
    if input_tensor is None:
        logger.info(f"Ошибка при подготовке изображения.")
        return
    content_img = input_tensor.clone()
    style_image = await preprocess(update, style_image_path, shape=content_img.shape[-2:])
    logger.info(f"Image style received and preprocessed")
    output_image = run_style_transfer(cnn, content_img, style_image, input_tensor)
    final_image = transforms.ToPILImage()(output_image.squeeze(0).clamp(0, 1))
    sender_id = update.message.from_user.id
    output_path = f'output_{sender_id}_{img_name}.png'
    final_image.save(output_path)
    max_dimension = 1024
    final_image.thumbnail((max_dimension, max_dimension), Image.LANCZOS)
    try:
        await update.message.reply_photo(photo=open(output_path, 'rb'))
        logger.info(f"Image sent to user_{sender_id}")
    except NetworkError as e:
        logger.error(f"NetworkError: {e}")
        await update.message.reply_text("Еще пару сек.")
        time.sleep(5)
        try:
            await update.message.reply_photo(photo=open(output_path, 'rb'))
        except Exception as e:
            logger.error(f"Unexpected error during retry: {e}")
            await update.message.reply_text("Произошла непредвиденная ошибка. Пожалуйста, попробуйте еще раз.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        await update.message.reply_text("Произошла непредвиденная ошибка. Пожалуйста, попробуйте еще раз.")

# Function to setup bot
async def setup_bot():
    global bot, cnn
    bot = Bot(token=TOKEN)
    cnn = models.vgg19(weights='VGG19_Weights.IMAGENET1K_V1').features
    cnn = cnn.to('cuda' if torch.cuda.is_available() else 'cpu')

# Function to set webhook
async def set_webhook():
    await bot.setWebhook(WEBHOOK_URL)
    logger.info(f"Webhook set to {WEBHOOK_URL}")

async def send_welcome_message():
    chat_id = '5182590043'
    await bot.send_message(chat_id=chat_id, text="Бот запущен и готов к работе!")

async def handle_callback_query(update:Update):
    global bot_data
    logger.info("Entered handle_callback_query function")
    if update.callback_query:
        query = update.callback_query
        await query.answer()
        logger.info(f"Callback query received with data: {query.data}")

        if query.data == 'set_style':
            await query.message.reply_text("Please send the image you want to use as the new style.")
            bot_data['expecting_style_image'] = True
            logger.info("Set style mode activated. bot_data['expecting_style_image'] set to True")
        elif query.data == 'set_default_style':
            global style_image_path
            style_image_path = "styles/sty.jpg"
            await query.message.reply_text("Default style image has been set.")
            logger.info(f"Default style image set: {style_image_path}")
    else:
        logger.warning("update.callback_query is None")
# Function to process updates
async def handle_text_message(update:Update):
    text = update.message.text
    if text.startswith('/'):
        command = text.split()[0]
        if command == '/start':
            await start(update)
        if command == '/help':
            await help_command(update)

async def handle_photo(update:Update):
    logger.info("Handling photo")
    global style_image_path,bot_data
    if bot_data.get('expecting_style_image', False):
        logger.info("Expecting style image: True")
        photo = update.message.photo[-1]
        file = await bot.get_file(photo.file_id)
        file_url=file.file_path
        sender_id = update.message.from_user.id
        file_path = f"styles/{sender_id}_{photo.file_unique_id}.jpg"
        #await file.download(file_path)
        async with aiohttp.ClientSession() as session:
            async with session.get(file_url) as response:
                if response.status == 200:
                    photo_bytes = await response.read()
                    async with aiofiles.open(file_path, mode='wb') as f:
                        await f.write(photo_bytes)
        bot_data['expecting_style_image'] = False
        global style_image_path
        style_image_path = file_path
        await update.message.reply_text("Style image has been updated.")
        logger.info(f"Style image has been updated: {style_image_path}")
    else:
        logger.info("Handling photo")
        logger.info(f"Expecting style image: {bot_data['expecting_style_image']}")
    
        photo_size = update.message.photo[-1]
        photo_file = await photo_size.get_file()
        file_url = photo_file.file_path
        sender_id = update.message.from_user.id
        filename = photo_file.file_path.split("/")[-1]
          
        # if bot_data['expecting_style_image']: 
        #     photo_path = f"styles/{sender_id}_{filename}"
        # else:
        photo_path = f"downloads/{sender_id}_{filename}.jpg"
        async with aiohttp.ClientSession() as session:
            async with session.get(file_url) as response:
                if response.status == 200:
                    photo_bytes = await response.read()
                    async with aiofiles.open(photo_path, mode='wb') as f:
                        await f.write(photo_bytes)
        # if bot_data['expecting_style_image']:
        #     style_image_path = photo_path
        #     bot_data['expecting_style_image'] = False
        #     await update.message.reply_text("Style image has been updated.")
        #     logger.info(f"New style image set: {style_image_path}")
        # else:
        await update.message.reply_text('Обрабатываю фото...')
        await process_and_send_image(update, photo_path)
        await update.message.reply_text("Готово, давай следующее!")
        logging.info(f"Finished processing photo {filename}")

async def handle_document(update:Update):
    logger.info("Handling document")
    global style_image_path,bot_data
    if bot_data.get('expecting_style_image', False):
        logger.info("Expecting style image: True")
        document = update.message.document
        file_info = await bot.get_file(document.file_id)
        file_url = file_info.file_path
        sender_id = update.message.from_user.id
        filename = file_url.split("/")[-1]
        document_path = f"styles/{sender_id}_{document.file_unique_id}.jpg"
        #await document_file.download(file_path)
        async with aiohttp.ClientSession() as session:
            async with session.get(file_url) as response:
                if response.status == 200:
                    document_bytes = await response.read()
                    async with aiofiles.open(document_path, mode='wb') as f:
                        await f.write(document_bytes)
        bot_data['expecting_style_image'] = False
        
        style_image_path = document_path
        await update.message.reply_text("Style image has been updated.")
        logger.info(f"Style image has been updated: {style_image_path}")
    else:
        logger.info(f"Expecting style image: {bot_data['expecting_style_image']}")
        document = update.message.document
        document_file = await document.get_file()
        file_id = document_file.file_id
        file_info = await bot.get_file(file_id)
        file_url = file_info.file_path
        sender_id = update.message.from_user.id
        filename = file_url.split("/")[-1]
        # if bot_data['expecting_style_image']:
        #     document_path = f"styles/{sender_id}_{filename}"
        # else:
        document_path = f"downloads/{sender_id}_{filename}"
        async with aiohttp.ClientSession() as session:
            async with session.get(file_url) as response:
                if response.status == 200:
                    document_bytes = await response.read()
                    async with aiofiles.open(document_path, mode='wb') as f:
                        await f.write(document_bytes)
        # if bot_data['expecting_style_image']:
        #     style_image_path = document_path
        #     bot_data['expecting_style_image'] = False
        #     await update.message.reply_text("Style image has been updated.")
        #     logger.info(f"New style image set: {style_image_path}")
        # else:
        await update.message.reply_text('Обрабатываю фото...')
        await process_and_send_image(update, document_path)
        await update.message.reply_text("Готово, давай следующее!")
        logging.info(f"Finished processing document {filename}")

@main_app.post(WEBHOOK_PATH)
async def webhook_handler(request: Request):
    global bot_data
    try:
        update_json = await request.json()
        update = Update.de_json(update_json, bot)
        
        if bot_data.get('expecting_style_image', False):
            logger.info("Обрабатывается изображение стиля напрямую")
            if update.message.photo or update.message.document:
                # Обработка изображения или документа стиля напрямую
                if update.message.photo:
                    await handle_photo(update)
                elif update.message.document:
                    await handle_document(update)
        else:
            update_queue.append(update)
        
        logger.info("Получен вебхук")
        logger.info(f"expecting_style_image: {bot_data.get('expecting_style_image', False)}")
        return JSONResponse(status_code=200, content={"message": "Webhook received"})
    except Exception as e:
        logger.error(f"Ошибка при обработке вебхука: {e}")
        return JSONResponse(status_code=500, content={"message": "Webhook processing error"})
# async def webhook_handler(request: Request):
#     global bot_data
#     try:
#         update_json = await request.json()
#         update = Update.de_json(update_json, bot)
#         if not bot_data['expecting_style_image']:
#             update_queue.append(update)
#         logger.info("Получен вебхук")
#         logger.info(bot_data['expecting_style_image'])
#         return JSONResponse(status_code=200, content={"message": "Webhook received"})
#     except Exception as e:
#         logger.error(f"Ошибка при обработке вебхука: {e}")
#         return JSONResponse(status_code=500, content={"message": "Webhook processing error"})

async def process_updates():
    logger.info("process_updates запущен")
    while True:
        #if update_queue or bot_data.get('expecting_style_image', False):
            if update_queue:
                update = update_queue[0]
                logging.info(f"Processing update: {update}")
                if update.callback_query:
                    logger.info("Processing callback query")
                    await handle_callback_query(update)
                elif update.message:
                    logger.info("Processing message")
                    if update.message.text:
                        await handle_text_message(update)
                    elif update.message.photo:
                        await handle_photo(update)
                    elif update.message.document:
                        await handle_document(update)
                update_queue.pop(0)
                logging.info(f"Remaining items in queue: {len(update_queue)})")
            await asyncio.sleep(1)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    async def main():
        global WEBHOOK_SET
        await setup_bot()
        # Create an application
        application = ApplicationBuilder().token(TOKEN).build()
        # Register handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CallbackQueryHandler(handle_callback_query))
        application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
        application.add_handler(MessageHandler(filters.Document, handle_document))
        # Проверяем, установлен ли вебхук
        if not WEBHOOK_SET:
            await set_webhook()
            WEBHOOK_SET = True
        asyncio.create_task(process_updates())
        await send_welcome_message()
        config = uvicorn.Config(main_app, host="0.0.0.0", port=5000, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
    asyncio.run(main())
