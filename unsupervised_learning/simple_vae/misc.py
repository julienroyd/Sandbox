import math
import torch
import numpy as np
import random
from itertools import zip_longest


def wandb_watch(wandb, learners):
    to_watch = []
    for learner in learners:
        to_watch += learner.wandb_watchable()

    if len(to_watch) > 0:
        wandb.watch(to_watch, log="all")

def standardize(x, mean, std):
    return (x - mean) / (std + 1e-5)
