import urllib.request
import logging
import datetime
import time
import os
import sys


def create_logger(name, loglevel, logfile):
    logger = logging.getLogger(name)
    logger.setLevel(loglevel)
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - {} - %(message)s'.format(name),
                                  datefmt='%m/%d/%Y %H:%M:%S', )
    for handler in [logging.FileHandler(logfile, mode='w'), logging.StreamHandler(stream=sys.stdout)]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def internet_on():
    try:
        urllib.request.urlopen('http://216.58.192.142', timeout=1)
        return True
    except Exception:
        return False


def bool_to_internet(state):
    return "ON" if state == True else "OFF"


if __name__ == '__main__':
    today = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M')
    logger = create_logger("LOGGER", loglevel=logging.INFO, logfile=os.path.join("ConnTests", f"{today}.txt"))

    start = time.time()
    old_connection_state = internet_on()

    logger.info(f"Starting connection tester: Internet is {bool_to_internet(old_connection_state)}")
    while True:
        # Checks if internet connection is still ON every second or so
        time.sleep(0.9)
        connection_state = internet_on()

        # If connection_state has changed since last check
        if old_connection_state != connection_state:
            old_connection_state = connection_state

            # Connection is back ON
            if connection_state == True:
                logger.info(f"Connection is back ON. Downtime: {(time.time() - start):.2f}s")

            # Connection is back OFF
            else:
                logger.info(f"Connection has been LOST.")
                start = time.time()

        # Prints the connection_state every minute if nothing has changed
        else:
            if datetime.datetime.now().second in [0]:
                logger.info(bool_to_internet(connection_state))
