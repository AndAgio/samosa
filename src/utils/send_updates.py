import telebot
import argparse


def send_update_via_telegram(message):
    bot = telebot.TeleBot(token="5308162044:AAGS2BhKO3TOJ2VTZf881DpzgzRC9PzJvGw")
    bot.send_message(chat_id=287350003, text=message)


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--message', type=str, default='', help='message to send to bot')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    options = parse_options()
    send_update_via_telegram(message=options.message)
