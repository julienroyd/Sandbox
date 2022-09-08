from multiprocessing import Process
import logging
import random
import numpy as np
import time
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


def _worker(worker_id, logger):
    logger.info(f"In process {worker_id}")
    time.sleep(2)  # do stuff


def launch_processes(n_processes):

    # Creates logger

    master_logger = create_logger(name=f'ID:{str(random.randint(1, 999999)).zfill(6)} - MASTER',
                                  loglevel=logging.INFO,
                                  logfile=None,
                                  streamHandle=True)

    # Launches multiple processes

    if n_processes > 1:

        processes = []

        for i in range(1, n_processes + 1):

            # Creates process logger

            logger = create_logger(name=f'ID:{str(random.randint(1, 999999)).zfill(6)} - SUBPROCESS_{i}',
                                   loglevel=logging.INFO,
                                   logfile=None,
                                   streamHandle=True)

            # Creates process

            processes.append(Process(target=_worker, args=(i, logger)))

        try:
            # start processes

            for p in processes:
                p.start()
                # time.sleep(0.5)

            # waits for all processes to end

            dead_processes = []
            while any([p.is_alive() for p in processes]):

                # check if some processes are dead

                for i, p in enumerate(processes):
                    if not p.is_alive() and i not in dead_processes:
                        master_logger.info(f'PROCESS_{i} has died.')
                        dead_processes.append(i)

                time.sleep(0.5)

        except KeyboardInterrupt:
            master_logger.info("KEYBOARD INTERRUPT. Killing all processes")

            # terminates all processes

            for process in processes:
                process.terminate()

        master_logger.info("All processes are done. Closing '__main__'")

    # No additional processes

    else:
        _worker(0, master_logger)

    return


if __name__ == "__main__":
    print("Hello Multiprocessing\n")
    for n in range(1, 10):
        print(f"Running {n} simultaneous processes")
        t = time.time()
        launch_processes(n_processes=n)
        print(f"\t-> {time.time() - t:.2f}s elapsed.\n\n")
