import numpy as np
from tqdm import tqdm


# The Coin Flipper Monopyly Experiment
# ------------------------------------
# The setup is as follows:
# 1. We spawn a population of N players with M coins each.
# 2. At each round, players are paired up randomly without replacement.
#    (If N is odd, one player is left out for this round.)
# 3. A coin is flipped for each pair, and the winner takes a coin from the loser.
#    (If a player has no coins, they are eliminated from the game.)
# 4. The game ends when one player has all the coins.
#
# The question is: how many rounds does it take for the game to end?
# We will run this experiment many times and look at the distribution of
# the number of rounds it takes to end the game.
#
# Parameters:
# - N: number of players
# - M: number of coins per player
# - num_experiments: number of times to run the experiment
#
# Returns:
# - A list of the number of rounds it took to end the game for each experiment.


def coin_flipper_monopoly_experiment(N, M, max_rounds, num_experiments):
    n_rounds = []
    for _ in tqdm(range(num_experiments)):
        n_rounds.append(coin_flipper_monopoly(N, M, max_rounds))
    return n_rounds


def coin_flipper_monopoly(N, M, max_rounds):
    players_money = np.ones(N).astype(np.uint) * M
    for round in range(max_rounds):
        players_money = coin_flipping_round(players_money)
        if np.sum(players_money != 0.) == 1:
            return round + 1


def coin_flipping_round(players_money):
    # get idxs of players that have at least one coin
    players_with_coins = np.where(players_money > 0.)[0]
    # randomly pair up players
    np.random.shuffle(players_with_coins)
    # if there is an odd number of players, one player is left out
    if len(players_with_coins) % 2 == 1:
        players_with_coins = players_with_coins[:-1]
    # reshape the players idxs per pair
    players_with_coins = players_with_coins.reshape(-1, 2)
    # flip coins un matrix form
    coin_flips = np.random.randint(0, 2, size=(len(players_with_coins),))
    # if the coin is 0, the first player gets a coin from the second, and viceversa
    players_money[players_with_coins[np.arange(len(players_with_coins)), coin_flips]] += 1
    players_money[players_with_coins[np.arange(len(players_with_coins)), 1 - coin_flips]] -= 1

    return players_money


if __name__ == '__main__':
    N = 10
    M = 20
    max_rounds = 100_000
    num_experiments = 100
    rounds = coin_flipper_monopoly_experiment(N, M, max_rounds, num_experiments)
    print(f'Average number of rounds: {np.mean(rounds)}')
    print(f'Median number of rounds: {np.median(rounds)}')
    print(f'Std number of rounds: {np.std(rounds)}')
    print(f'Min number of rounds: {np.min(rounds)}')
    print(f'Max number of rounds: {np.max(rounds)}')
