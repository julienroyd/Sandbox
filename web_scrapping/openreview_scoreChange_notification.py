# import urllib.request, urllib.parse
# import os
# from pathlib import Path
# from datetime import datetime
#
# url = 'https://openreview.net/forum?id=BkggGREKvS'
#
# response = urllib.request.urlopen(url)
# webcontent = response.read().decode("utf-8")
#
# save_path = Path("openreview_files")
# os.makedirs(save_path, exist_ok=True)
#
# with open(save_path / f'iclr_submission_forum_{datetime.now().strftime("%d-%m-%Y-%Hh:%Mm:%Ss")}_v1', 'w+') as f:
#     f.write(webcontent)
#
# print('CANNOT GET THE REVIEWS BECAUSE THE PAGE IS DYNAMIC')


import openreview
from pathlib import Path
from datetime import datetime
import json
import os
from textwrap import wrap
import re
import time
import smtplib
from email.message import EmailMessage
import logging
import sys


def create_logger(name, loglevel, logfile=None, streamHandle=True):
    logger = logging.getLogger(name)
    logger.setLevel(loglevel)
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - {} - %(message)s'.format(name),
                                  datefmt='%d/%m/%Y %H:%M:%S', )

    handlers = []
    if logfile is not None:
        handlers.append(logging.FileHandler(logfile, mode='a'))
    if streamHandle:
        handlers.append(logging.StreamHandler(stream=sys.stdout))

    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


my_submission_forum = 'BkggGREKvS'  # the forum-id to "Promoting coordination through...." ICLR 2020
save_path = Path('openreview_files')
os.makedirs(save_path, exist_ok=True)
logger = create_logger(name="OpenreviewWatcher", loglevel=logging.INFO, logfile=save_path/"logger.out")

guest_client = openreview.Client(baseurl='https://openreview.net')
my_submission_notes = guest_client.get_notes(forum=my_submission_forum)


while True:

    # scrapping reviews

    my_submission_reviews = {}
    for note in my_submission_notes:
        if "Official Blind Review" in note.content['title']:
            my_submission_reviews[note.content['title']] = [note.content['rating'], wrap(note.content['review'], width=150)]

    logger.info(f"Just checked forum {my_submission_forum}")

    # loading previous reviews

    prev_reviews_files = sorted_nicely([str(review_file) for review_file in save_path.iterdir() if "logger" not in review_file.name])
    prev_review_file = prev_reviews_files[-1]

    with open(prev_review_file, 'r') as f:
        prev_reviews = json.load(f)

    # checking if reviews change since latest saved review

    if prev_reviews != my_submission_reviews:

        # saving reviews

        last_version = int(prev_review_file.strip('.json').split('v')[-1])

        new_file_name = f'iclr_submission_reviews_{datetime.now().strftime("%d-%m-%Y-%Hh:%Mm:%Ss")}_v{last_version + 1}.json'
        with open(save_path / new_file_name, 'w+') as f:
            json.dump(my_submission_reviews, f)

        logger.info(f"Reviews have changed! Saved new version at: {save_path / new_file_name}")

        # send an email

        # my_email = None
        # msg = EmailMessage()
        # msg.set_content("https://openreview.net/forum?id=BkggGREKvS")
        # msg['Subject'] = f'Your ICLR reviews have changed!'
        # msg['From'] = my_email
        # msg['To'] = my_email
        #
        # server = smtplib.SMTP('smtp.gmail.com', 587)
        # server.login("youremailusername", "password")
        #
        # with smtplib.SMTP(host='localhost', port=1025) as s:
        #     s.send_message(msg)
        #
        # logger.info(f"Email sent at: {my_email}")

    time.sleep(600)
