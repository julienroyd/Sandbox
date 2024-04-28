import numpy as np
import matplotlib.pyplot as plt

# Define reward distributions for each action (e.g., mean and standard deviation)
reward_distributions = [
    {"mean": 0.5, "std": 0.2},
    {"mean": 0.7, "std": 0.3}
]
colors = ['blue', 'orange', 'green']

# Violing plots of the true reward distributions
fig, axes = plt.subplots(1, 2, figsize=(20, 5))
ax = axes[0]
for i, dist in enumerate(reward_distributions):
    ax.violinplot(np.random.normal(dist["mean"], dist["std"], 1000), positions=[i])
    ax.scatter(i, dist["mean"], color=colors[i])
ax.set_xlabel('Actions')
ax.set_ylabel('Reward')
ax.set_title('True Reward Distributions')

# Define parameters
C = 1.0
total_pulls = 100

# Initialize arrays to store number of pulls and mean rewards
N_a = len(reward_distributions) 
N = np.zeros(N_a) 
Q = np.zeros(N_a) 

# Initialize array to store confidence terms over time
confidence_terms = np.zeros((N_a, total_pulls))
Q_estimates = np.zeros((N_a, total_pulls))
chosen_actions = np.zeros(total_pulls)

# Main loop
for t in range(1, total_pulls):
    confidence_terms[:, t - 1] = C * np.sqrt(np.log(t) / np.max(np.stack([np.ones_like(N), N]), axis=0))
    
    # Choose action with highest upper confidence bound
    chosen_action = np.argmax(Q_estimates[:, t - 1] + confidence_terms[:, t - 1])
    
    # Simulate pulling the chosen action and observe reward
    reward = np.random.normal(reward_distributions[chosen_action]["mean"],
                              reward_distributions[chosen_action]["std"])
    
    # Update estimates
    N[chosen_action] += 1
    chosen_actions[t] = chosen_action
    Q_estimates[chosen_action, t] = Q_estimates[chosen_action, t - 1] + ((reward - Q_estimates[chosen_action, t - 1]) / N[chosen_action])
    Q_estimates[np.arange(N_a) != chosen_action, t] = Q_estimates[np.arange(N_a) != chosen_action, t - 1]

# Plot confidence terms over time
ax = axes[1]
for i in range(N_a):
    ax.hlines(reward_distributions[i]["mean"], 1, total_pulls, linestyles='dashed', label=f'True mean of action {i+1}', color=colors[i])
    ax.plot(range(1, total_pulls + 1), Q_estimates[i], label=f'Q-estimate {i+1}', color=colors[i])
    ax.fill_between(range(1, total_pulls + 1), y1=Q_estimates[i] + confidence_terms[i], y2=Q_estimates[i], label=f'Confidence term {i+1}', alpha=0.2, color=colors[i])

# vertical thin blue line for t where action 1 is selected
chosen_action_is_one = chosen_actions == 0
ax.vlines(np.arange(total_pulls)[chosen_action_is_one] + 1, ymin=-0.2, ymax=1.5, color='blue', alpha=0.5, label='Chosen action is 1', linewidth=0.5)

# plots
ax.set_xlabel('Number of pulls')
ax.set_ylabel('Confidence term')
ax.set_title('UCB Confidence Terms over Time')
fig.legend()
plt.tight_layout()
plt.show()
