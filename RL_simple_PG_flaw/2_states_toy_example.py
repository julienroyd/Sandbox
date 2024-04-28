import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn
seaborn.set() #make the plots look pretty

AGENT_NAME = "PG"
EPS = 0.01

LR = 0.01
N_STEPS = int(1e4)

R_00 = 1.
R_01 = 0.99
R_10 = 0.
R_11 = 1.

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

class PGagent(object):
    def __init__(self, lr=0.01):

        class SimplestModel(object):
            def __init__(self):
                self.w = np.random.uniform(low=-0.1, high=0.1)
                self.b = 0.

            def forward(self, s):
                return sigmoid(self.w * s + self.b)

            def backward(self, s, a):
                sign = 1 if a == 0 else -1
                dw = sign * self.forward(s) * (1 - self.forward(s)) * s
                db = sign * self.forward(s) * (1 - self.forward(s)) * 1.
                return dw, db

        self.model = SimplestModel()
        self.lr = lr

    def select_action(self, s):
        prob_a_0 = self.model.forward(s)
        return np.random.binomial(n=1, p=1.-prob_a_0)

    def learning_step(self, s, a, r):
        prob_action_0 = self.model.forward(s)
        prob_action_1 = 1. - prob_action_0
        denominator = prob_action_0 if a == 0 else prob_action_1

        dw, db = self.model.backward(s, a)
        self.model.w += self.lr * r * (dw / denominator)
        self.model.b += self.lr * r * (db / denominator)


class Qagent(object):
    def __init__(self, lr=0.01, eps=0.05):

        class Qmodel(object):
            def __init__(self):
                self.w0 = np.random.uniform(low=-0.1, high=0.1)
                self.w1 = np.random.uniform(low=-0.1, high=0.1)
                self.b0 = 0.
                self.b1 = 0.

            def forward(self, s):
                return np.array([self.w0 * s + self.b0,
                                 self.w1 * s + self.b1])

            def backward(self, s, a):
                if a == 0:
                    dw0 = s
                    db0 = 1.
                    dw1 = 0.
                    db1 = 0.

                elif a == 1:
                    dw0 = 0.
                    db0 = 0.
                    dw1 = s
                    db1 = 1.

                else:
                    raise ValueError('There are only two possible actions in any givem state on this problem')

                return dw0, db0, dw1, db1


        self.model = Qmodel()
        self.lr = lr
        self.eps = eps

    def select_action(self, s):
        if np.random.rand() < self.eps:
            return np.random.binomial(n=1, p=0.5)
        else:
            return np.argmax(self.model.forward(s))

    def learning_step(self, s, a, r):
        """ We use a MSE loss """
        td_error = r - self.model.forward(s)[a]
        dw0, db0, dw1, db1 = self.model.backward(s, a)

        self.model.w0 += self.lr * 2 * td_error * dw0
        self.model.b0 += self.lr * 2 * td_error * db0
        self.model.w1 += self.lr * 2 * td_error * dw1
        self.model.b1 += self.lr * 2 * td_error * db1



class Env(object):
    def __init__(self):
        self.initial_s = np.random.binomial(n=1, p=0.5)
        self.reward_function = np.array([[R_00, R_01],  # s = 0
                                         [R_10, R_11]]) # s = 1

    def reset(self):
        self.initial_s = np.random.binomial(n=1, p=0.5)

    def step(self, a):
        # Only returns a reward
        return self.reward_function[self.initial_s, a]


class Recorder(object):
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.outputs = []

    def save_episode(self, s, a, r):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)

    def plot_rewards(self, ax, smooth_it_up=False, title="Reward over time"):

        data =self.rewards
        if smooth_it_up:
            def smooth(data_list, last_n=25):
                mean = np.mean(data_list[0:last_n])
                new_array = []
                for element in data_list:
                    mean = (1. - 1. / last_n) * mean + (1. / last_n) * element
                    new_array.append(mean)
                return new_array

            data = smooth(data)


        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("Episodes")
        ax.plot(data, label="Reward")
        ax.legend(loc='best')
        ax.grid(True, color="lightgrey")

        return ax

    def plot_outputs(self, ax, data, title="Policy"):
        ind = np.arange(1,3)  # location of ticks
        width = 0.35  # width of ticks

        rects1 = ax.bar(ind, data[:, 0], width, color='blue', label="a=0")
        rects2 = ax.bar(ind + width, data[:, 1], width, color='orange', label="a=1")
        ax.legend(loc='best')

        ax.set_title(title, fontweight='bold')
        ax.set_ylim(0., 1.02)
        ax.set_xticks(ind + width / 2.)
        ax.set_xticklabels(['s=0', 's=1'])

        def autolabel(rects):
            # Attach a text label above each bar displaying its height
            for rect in rects:
                ax.text(x=rect.get_x() + rect.get_width() / 2.,
                        y=0.90 * rect.get_height(),
                        s='{:.2f}'.format(rect.get_height()),
                        ha='center',
                        va='bottom',
                        fontweight='bold')

        autolabel(rects1)
        autolabel(rects2)

        return ax


if __name__ == '__main__':

    # Initialization
    agent = None
    if AGENT_NAME == "PG":
        agent = PGagent(lr=LR)
    elif AGENT_NAME == "QN":
        agent = Qagent(lr=LR, eps=EPS)
    env = Env()
    recorder = Recorder()

    # Records the initial policy
    data = None
    if AGENT_NAME == "PG":
        data = np.array([[agent.model.forward(s=0), 1. - agent.model.forward(s=0)],
                         [agent.model.forward(s=1), 1. - agent.model.forward(s=1)]])
    elif AGENT_NAME == "QN":
        data = np.array([agent.model.forward(0),
                         agent.model.forward(1)])
    recorder.outputs.append(data)

    # Training loop
    for i in tqdm(range(N_STEPS)):

        env.reset()
        s = env.initial_s
        a = agent.select_action(s)
        r = env.step(a)
        agent.learning_step(s, a, r)

        recorder.save_episode(s,a,r)

    # Records the final policy
    data = None
    if AGENT_NAME == "PG":
        data = np.array([[agent.model.forward(s=0), 1. - agent.model.forward(s=0)],
                         [agent.model.forward(s=1), 1. - agent.model.forward(s=1)]])
    elif AGENT_NAME == "QN":
        data = np.array([agent.model.forward(0),
                         agent.model.forward(1)])
    recorder.outputs.append(data)

    # Plots
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    title_init = "Initial Policy (PG)" if AGENT_NAME == "PG" else "Initial Q-values (QN)"
    title_end = "Final Policy (PG)" if AGENT_NAME == "PG" else "Final Q-values (QN)"
    ax1 = recorder.plot_outputs(ax1, recorder.outputs[0], title=title_init)
    ax2 = recorder.plot_rewards(ax2, smooth_it_up=True, title="Smoothed reward over time")
    ax3 = recorder.plot_outputs(ax3, recorder.outputs[1], title=title_end)

    plt.show()
