import argparse 
import logging
import multiprocessing
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import sys
import queue

DEFAULT_DURATION = 100
DEFAULT_LOGG_LEVEL = logging.INFO
DEFAULT_N_CLIENTS = 2
DEFAULT_N_COOKS = 8
DEFAULT_N_MAX_SIZE = 3


# ============================= MULTIPROCESS ==============================
# =========================================================================

class Restaurant(multiprocessing.Process):
    def __init__(self, log_level, kitchen_queues):
        super().__init__()

        self.log_level = log_level
        self.logger = None
        self.kitchen_queues = kitchen_queues
        self.exit = multiprocessing.Event()
        self.food_counter = 0

    def run(self):
        self.logger = create_logger('RESTAURANT', self.log_level, os.path.join('logs', 'RESTAURANT.log'))
        self.logger.info("Process initiated.")
        while not self.exit.is_set():
            try:
                self.work()
            except KeyboardInterrupt: pass
        self.logger.info("Shutting down.")

    def shutdown(self):
        self.exit.set()

    def work(self):
        for i in range(len(self.kitchen_queues)):
            # Takes food and put in on the table
            try:
                food = self.kitchen_queues[i].get(block=False)
                self.food_counter += 1
                self.logger.info(f'{food} incoming!')
                food_image = plt.imread(os.path.join('assets', f'{food}.jpg'))
                time.sleep(2)
                plt.imsave(os.path.join('table', f'{self.food_counter}_{food}.jpg'), food_image)

            # Yells at the cook 
            except queue.Empty:
                self.logger.info(f'There is no food in queue #{i}, hurry up cook_{i} !')
                time.sleep(1)


class Cook(multiprocessing.Process):
    def __init__(self, cook_id, log_level, kitchen_queue):
        super().__init__()

        self.id = cook_id
        self.name = f'cook_{cook_id}'
        
        self.log_level = log_level
        self.logger = None
        self.kitchen_queue = kitchen_queue
        
        self.exit = multiprocessing.Event()

        self.actions = ['Preparing a PIZZA',
                        'Peparing SPAGHETTI',
                        'Preparing a PAD-THAI',
                        'Taking a little BREAK',
                        'Going to the BATHROOM']

    def run(self):
        self.logger = create_logger(self.name.upper(), self.log_level, os.path.join('logs', f'{self.name.upper()}.log'))
        self.logger.info("Process initiated.")
        while not self.exit.is_set():
            try:
                self.work()
            except KeyboardInterrupt: pass  # TODO : Could that cause problems? Is that the best way to do it?
        self.logger.info("Shutting down.")

    def shutdown(self):
        self.exit.set()

    def work(self):

        # Chooses an action
        food = None
        action_id = np.random.randint(low=0, high=5)
        if action_id < 3:
            food = self.actions[action_id].split(' ')[-1]
            time_needed = np.random.randint(low=3, high=8)
        elif action_id in [3, 4]:
            time_needed = np.random.randint(low=10, high=15)
        else:
            raise ValueError(f'There are only 5 possible actions for a Cook. Got action_id={action_id}.')

        # Sleeps the required amount of time
        self.logger.info('\t{}\t({})'.format(self.actions[action_id], time_needed))
        time.sleep(time_needed)

        # Puts the food in queue (if any)
        if food is not None:
            try:
                self.kitchen_queue.put(food)
            except queue.Full:
                self.logger.info('My kitchen queue is full! Taking a break..')
                time.sleep(15)


class Client(multiprocessing.Process):
    def __init__(self, client_id, log_level):
        super().__init__()

        self.id = client_id
        self.name = f'client_{client_id}'
        
        self.log_level = log_level
        self.logger = None

        self.exit = multiprocessing.Event()

    def run(self):
        self.logger = create_logger(self.name.upper(), self.log_level, os.path.join('logs', f'{self.name.upper()}.log'))
        self.logger.info("Process initiated.")
        while not self.exit.is_set():
            try:
                self.eat()
            except KeyboardInterrupt: pass
        self.logger.info("Shutting down.")

    def shutdown(self):
        self.exit.set()

    def eat(self):
        all_food = sorted(os.listdir('table'))

        if len(all_food) != 0:
            try:
                os.remove(os.path.join('table', all_food[0]))
                self.logger.info(f'Just ate a {all_food[0]}')
            except FileNotFoundError:
                self.eat()
        else:
            self.logger.info('The table is empty, I am hungry!')

        time.sleep(5)


def shutdown_processes(logger, processes):
    logger.info('Ending all the processes')
    for process in processes:
        process.shutdown()


def kill_processes(logger, processes):
    logger.info('Killing all the processes')
    for process in processes:
        process.terminate()


# ============================= ARGS ======================================
# =========================================================================
def get_args():
    """
    Parse the args from command line and check them
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', default=DEFAULT_DURATION, type=int)
    parser.add_argument('--loglevel', default=DEFAULT_LOGG_LEVEL, type=parse_log_level)

    parser.add_argument('--ncooks', default=DEFAULT_N_COOKS, type=int)
    parser.add_argument('--nclients', default=DEFAULT_N_CLIENTS, type=int)
    parser.add_argument('--maxsize', default=DEFAULT_N_MAX_SIZE, type=int)

    parser.add_argument('--savelog', default='true', type=parse_bool)

    args = check_args(parser.parse_args())

    return args


def check_args(args):
    """
    Check the args from command line
    """
    if args.ncooks > 8:
        raise ArgsError(f'The argument "args.ncooks" must be between 0 and 8. Got {args.ncooks} instead.')

    if args.nclients > 8:
        raise ArgsError(f'The argument "args.nclients" must be between 0 and 8. Got {args.nclients} instead.')

    if args.maxsize < 1:
        raise ArgsError(f'The argument "args.maxsize" must be at least 1. Got {args.maxsize} instead.')

    if type(args.savelog) is not bool:
        raise ArgsError(f'The argument "savelog" should be a bool.')

    return args


def parse_log_level(level_arg):
    """
    Parse a string representing a log level.

    :param level_arg: The desired level as a string (ex: 'info').
    :return: The corresponding level as a `logging` member (ex: `logging.INFO`).
    """
    return getattr(logging, level_arg.upper())


def parse_bool(bool_arg):
    """
    Parse a string representing a boolean.

    :param bool_arg: The string to be parsed
    :return: The corresponding boolean
    """
    if bool_arg.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif bool_arg.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgsError(f'Boolean argument expected. Got {bool_arg} instead.')


class ArgsError(Exception):
    pass


# ============================= LOGGING ===================================
# =========================================================================

def create_logger(name, loglevel, logfile):
    logger = logging.getLogger(name)
    logger.setLevel(loglevel)
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - {} - %(message)s'.format(name),
                                  datefmt='%d/%m/%Y %H:%M:%S',)
    for handler in [logging.FileHandler(logfile, mode='w'), logging.StreamHandler(stream=sys.stdout)]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# ============================= MAIN ======================================
# =========================================================================

if __name__ == '__main__':
    # Create table folder if it doesnt exists
    if not os.path.exists('table'):
        os.mkdir('table')
    if not os.path.exists('logs'):
        os.mkdir('logs')

    # .. or delete food if it does
    else:
        food_on_table = [f for f in os.listdir('table') if f.endswith(".jpg")]
        for f in food_on_table:
            os.remove(os.path.join('table', f))

    # Parse args
    args = get_args()

    # Create logger for master process
    master_logger = create_logger('MASTER', args.loglevel, os.path.join('logs', 'MASTER.log'))

    # Create queues
    kitchen_queues = []
    for _ in range(args.ncooks):
        kitchen_queues.append(multiprocessing.Queue(maxsize=args.maxsize))

    # Creates sub-processes
    master_logger.info('Creating 1 Restaurant process')
    restaurant_process = Restaurant(args.loglevel, kitchen_queues)

    master_logger.info(f'Creating {args.ncooks} Cook processes')
    cook_processes = []
    for i in range(args.ncooks):
        cook_processes.append(Cook(i, args.loglevel, kitchen_queues[i]))

    master_logger.info(f'Creating {args.nclients} Client processes')
    client_processes = []
    for i in range(args.nclients):
        client_processes.append(Client(i, args.loglevel))

    # Start all processes.
    all_processes = [restaurant_process] + cook_processes + client_processes
    master_logger.info('Starting {} processes (#restaurant = 1, #cooks = {}, #clients = {})'.format(
        len(all_processes), len(cook_processes), len(client_processes)))
    
    start_time = time.time()
    for i, process in enumerate(all_processes):
        process.start()

    dead_process = False
    while True:
        try:
            # Check if some processes are dead
            for process in all_processes:
                if process.is_alive():
                    master_logger.debug('{process.name} is still alive.')
                else:
                    master_logger.debug('{process.name} has crashed.')
                    dead_process = True
                    break

            if dead_process:
                master_logger.info("A dead process has been found. Leaving the main loop.")
                break

            # Check if we exceeded duration
            current_time = time.time()
            time_exceeded = True if (current_time - start_time) > args.duration else False

            if time_exceeded:
                master_logger.info("Time's up. Leaving the main loop.")
                break

            time.sleep(3)

        except KeyboardInterrupt:
            master_logger.info("Catched a KEYBOARD INTERRUPT")
            break

    master_logger.info('Out of the main loop.')

    # Waits for all processes to shutdown
    shutdown_processes(master_logger, all_processes)

    while any([p.is_alive() for p in all_processes]):
        master_logger.info([p.is_alive() for p in all_processes])
        time.sleep(2)

    # Without the calls below the process may hang while waiting for queues to be flushed
    master_logger.info('Ensuring queues do not prevent the process from exiting')
    for the_q in kitchen_queues:
        the_q.cancel_join_thread()

    master_logger.info('Everything done. Shutting down.')



    
