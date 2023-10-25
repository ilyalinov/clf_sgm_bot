import telebot
import requests
from PIL import Image
from config import token
from clf_utils import *
from sgm_utils import *

bot = telebot.TeleBot(token)
DEVICE = get_device()
model_clf = get_clf_model()
model_sgm = get_sgm_model()

@bot.message_handler(commands=['start', 'help'])
def start_message(message):
    start_message = ('This bot can segment and classify skin lesion images. '
                     'Only nevus and melanoma lesion types are supported. '
                     'Send a photo of the skin lesion.')
    bot.send_message(message.chat.id, start_message)

@bot.message_handler(content_types='text')
def message_reply(message):
    answer = 'hi'
    bot.send_message(message.chat.id, answer)

@bot.message_handler(content_types=['photo'])
def process_image(message):
    file_id = message.photo[-1].file_id
    url = bot.get_file_url(file_id)
    img = Image.open(requests.get(url, stream=True).raw)
    size = img.size
    res_dict = classify_one_image(model_clf, img, DEVICE)
    bot.send_message(message.chat.id, 'Classification result\n' + dict_to_str(res_dict))
    mask = segment_one_sample(model_sgm, img, DEVICE)
    bio = prepare_mask(mask, size)
    bot.send_photo(message.chat.id, photo=bio, caption='Segmentation mask')

if __name__ == '__main__':
    bot.infinity_polling()