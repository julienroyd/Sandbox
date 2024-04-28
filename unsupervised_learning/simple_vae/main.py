import argparse
import random
import numpy as np
import itertools
import logging
from nop import NOP
from copy import deepcopy
import time
import torch
import torchvision

from utils import *
from vae_model import VAE
from ae_model import AE
from data.GMIX.gmix_dataset import GMIXdataset
from misc import wandb_watch, standardize

import alfred
from alfred.utils.misc import create_management_objects
from alfred.utils.config import parse_bool, parse_log_level, save_config_to_json, load_config_from_json
from alfred.utils.recorder import Recorder
from alfred.utils.directory_tree import *
import alfred.defaults


def set_up_alfred():
    alfred.defaults.DEFAULT_DIRECTORY_TREE_ROOT = "./storage"
    alfred.defaults.DEFAULT_DIRECTORY_TREE_GIT_REPOS_TO_TRACK['sandbox'] = str(Path(__file__).absolute().parents[2])
    alfred.defaults.DEFAULT_DIRECTORY_TREE_GIT_REPOS_TO_TRACK['alfred'] = str(Path(alfred.__file__).parents[1])

    alfred.defaults.DEFAULT_PLOTS_ARRAYS_TO_MAKE = [
        ('epoch_i', 'train_loss', (None, None), (None, None)),
        ('epoch_i', 'valid_loss', (None, None), (None, None))
    ]

    alfred.defaults.DEFAULT_BENCHMARK_X_METRIC = "epoch_i"
    alfred.defaults.DEFAULT_BENCHMARK_Y_METRIC = "valid_perf"
    alfred.defaults.DEFAULT_BENCHMARK_PERFORMANCE_METRIC = "valid_perf"


def get_main_args(overwritten_args=None):
    parser = argparse.ArgumentParser()

    # alfred's arguments

    parser.add_argument('--alg_name', type=str, default='vae', choices=['vae', 'ae'])
    parser.add_argument('--task_name', type=str, default="gmix_3", choices=['mnist', 'gmix_1', 'gmix_3'])
    parser.add_argument("--desc", type=str, default="", help="Description of the experiment to be run")
    parser.add_argument("--seed", default=1, type=int, help="Random seed")
    parser.add_argument("--root_dir", default="./storage", type=str, help="Root directory")
    parser.add_argument("--log_level", default=logging.INFO, type=parse_log_level, help="Logging level")

    # common arguments

    parser.add_argument('--lr', type=float, default=0.001, metavar='G',
                        help='learning rate')

    parser.add_argument('--lr_decay_per_update', type=float, default=1., metavar='G',
                        help='learning rate decay coefficient')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='batch size')

    parser.add_argument('--hidden_dims', type=int, default=[512, 256, 128], metavar='N',
                        help='size of hidden layers')

    parser.add_argument('--num_steps', type=int, default=5000, metavar='N',
                        help='maximum number of steps (default: 1000000)')

    parser.add_argument('--steps_bw_save', type=int, default=1000, metavar='N',
                        help='Number of environment steps between save')

    parser.add_argument('--steps_bw_eval', type=int, default=1000, metavar='N',
                        help='Number of environment episodes between evaluations')

    parser.add_argument('--use_cuda', type=parse_bool, default=False,
                        help='run on CUDA (default: False)')

    # VAE arguments

    parser.add_argument('--z_dim', type=int, default=10, metavar='N',
                        help='Dimensionality for continuous latent variables')

    parser.add_argument('--beta', type=float, default=1.,
                        help="Beta-parameter for beta-VAE (coefficient for the prior loss)")

    # wandb arguments

    parser.add_argument("--use_wandb", default=True, type=parse_bool)
    parser.add_argument("--sync_wandb", default=False, type=parse_bool)
    parser.add_argument("--save_best_model_to_wandb", default=False, type=parse_bool)
    return parser.parse_args(overwritten_args)


def validate_config(args, logger):
    original_args = deepcopy(args)

    # Re-validate args if they have been changed

    if original_args != args:
        args = validate_config(args, logger)

    return args


def main(config, dir_tree=None, logger=None, pbar="default_pbar"):
    # Setting up alfred's configs

    set_up_alfred()

    # Management objects

    dir_tree, logger, pbar = create_management_objects(dir_tree=dir_tree, logger=logger, pbar=None, config=config)

    # Scan arguments for errors

    config = validate_config(config, logger)

    # Determines which device will be used

    config.device = str(torch.device("cuda" if config.use_cuda and torch.cuda.is_available() else "cpu"))
    logger.info(f"Device to be used: {config.device}")

    # Saving config

    save_config_to_json(config, filename=str(dir_tree.seed_dir / "config.json"))
    if (dir_tree.seed_dir / "config_unique.json").exists():
        config_unique = load_config_from_json(filename=str(dir_tree.seed_dir / "config_unique.json"))
        for k in config_unique.__dict__.keys():
            config_unique.__dict__[k] = config.__dict__[k]
        save_config_to_json(config_unique, filename=str(dir_tree.seed_dir / "config_unique.json"))

    # Dataset

    if config.task_name == 'mnist':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            torchvision.transforms.Lambda(lambda x: torch.flatten(x))
        ])
        dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
        n_eval_generations = 10

    elif config.task_name == 'gmix_3':
        dataset = GMIXdataset(path="data/GMIX/data_0.npy")
        # dataset.data = standardize(dataset.data, mean=0.1210, std=4.6956)
        n_eval_generations = 100

    elif config.task_name == 'gmix_1':
        dataset = GMIXdataset(path="data/GMIX/data_1.npy")
        # dataset.data = standardize(dataset.data, mean=5., std=0.9985)
        n_eval_generations = 100

    else:
        raise NotImplementedError

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Set aside a fixed set of x's (to evaluate reconstruction)

    eval_x = next(iter(data_loader))[0][:n_eval_generations].detach()  # TODO: deal with batch_labels later (should not even be returned by __getitem__() for unlabeled datasets)

    # Seeding

    set_seeds(config.seed)

    # Agent

    if config.alg_name == "vae":
        algorithm = VAE(x_dim=eval_x.shape[1], config=config, logger=logger)
    elif config.alg_name == "ae":
        algorithm = AE(x_dim=eval_x.shape[1], config=config, logger=logger)
    else:
        raise NotImplementedError

    # Recorder

    os.makedirs(dir_tree.recorders_dir, exist_ok=True)
    train_recorder = Recorder(algorithm.metrics_to_record)

    # Wandb

    if not config.sync_wandb:
        os.environ['WANDB_MODE'] = 'dryrun'
        os.environ['WANDB_DISABLE_CODE'] = 'true'

    if config.use_wandb:
        import wandb
        os.environ["WANDB_DIR"] = str(dir_tree.seed_dir.absolute())
        wandb.init(name=dir_tree.get_run_name(), project='behavioral_preferences', reinit=True)
        wandb.config.update(config, allow_val_change=True)
        wandb_save_dir = Path(wandb.run.dir) if config.save_best_model_to_wandb else None
    else:
        wandb = NOP()
        wandb_save_dir = None

    wandb_watch(wandb, [algorithm])

    # First evaluation before initiating training
    # Set aside a fixed set of z's (to evaluate generation) and x's (to evaluate reconstruction)

    eval_z = algorithm.model.prior.sample(sample_shape=(n_eval_generations,)).detach()
    evaluation_step(algorithm, eval_z, eval_x, config.task_name, save_path=dir_tree.seed_dir / "eval_imgs" / f"Epoch0")

    # Training loop

    update_i = 0
    stop_training = False
    last_eval_step = 0

    for epoch_i in itertools.count(1):
        time_start = time.time()
        epoch_losses = []

        # Episode loop

        for (batch_data, batch_labels) in data_loader:  # TODO: deal with batch_labels later (should not even be returned by __getitem__() for unlabeled datasets)

            new_recordings = algorithm.update_parameters(batch_data, update_i)

            # Save loss recordings

            wandb.log(new_recordings)
            train_recorder.write_to_tape(new_recordings)
            update_i += 1
            epoch_losses.append(new_recordings['total_loss'])

            # Save plots and models

            if update_i % config.steps_bw_save == 0:
                save(algorithm, train_recorder, dir_tree, logger)

            # Decide whether to continue training

            if update_i >= config.num_steps:
                stop_training = True
                break

        if stop_training:
            break

        time_stop = time.time()

        # Save return recordings

        new_recordings = {
            'epoch_i': epoch_i,
            'epoch_time': time_stop - time_start,
            'wallclock_time': time.time(),
        }

        train_recorder.write_to_tape(new_recordings)
        wandb.log(new_recordings)
        logger.info(f"Epoch: {epoch_i}, total numsteps: {update_i} ({time_stop - time_start:.2f}s), total_loss: {np.mean(epoch_losses):.2f}")

        # Evaluate the model

        if update_i - last_eval_step > config.steps_bw_eval:
            logger.info("Saving plots for visualising reconstruction and generation performance.")
            evaluation_step(algorithm, eval_z, eval_x, config.task_name, save_path=dir_tree.seed_dir / "eval_imgs" / f"Epoch{epoch_i}")
            last_eval_step = update_i

    # Final evaluation and save

    evaluation_step(algorithm, eval_z, eval_x, config.task_name, save_path=dir_tree.seed_dir / "eval_imgs" / f"Epoch{epoch_i}")
    save(algorithm, train_recorder, dir_tree, logger)

    # finishes logging before exiting training script
    wandb.join()


def save(model, train_recorder, dir_tree, logger):
    train_recorder.save(dir_tree.recorders_dir / 'train_recorder.pkl')
    model.create_plots(train_recorder, save_dir=dir_tree.seed_dir)
    model.save_model(path=dir_tree.seed_dir, logger=logger)


def evaluation_step(algorithm, eval_z, eval_x, task_name, save_path):
    with torch.no_grad():
        generated_x = algorithm.model.decode(eval_z)

        encoded_z, _, _ = algorithm.model.encode(eval_x)
        reconstructed_x = algorithm.model.decode(encoded_z)

    if task_name == "mnist":
        eval_plot_mnist(x=eval_x, x_hat=reconstructed_x, x_new=generated_x, save_path=save_path)

    elif "gmix" in task_name:
        eval_plot_gmix(x=eval_x, x_hat=reconstructed_x, x_new=generated_x, save_path=save_path)


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    config = get_main_args()
    main(config)