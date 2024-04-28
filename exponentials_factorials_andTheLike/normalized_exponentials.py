import numpy as np
import matplotlib.pyplot as plt

Ts = [10, 25, 50, 100, 200]
transition_weights_coeff = [1.01, 1.03, 1.05, 1.1]
norm = 10.

fig, axes = plt.subplots(1, len(transition_weights_coeff), figsize=(8 * len(transition_weights_coeff),5))
cm = plt.cm.get_cmap('viridis')
colors = [np.array(cm(float(k) / float(len(transition_weights_coeff)))) for k, _ in enumerate(Ts)]

# EXPONENTIAL WEIGHTING

for i, coeff in enumerate(transition_weights_coeff):
    for j, T in enumerate(Ts):
        time_weights = np.power(coeff, np.arange(T))
        normalised_time_weights = (norm - 1.) * ((time_weights - np.min(time_weights)) / (np.max(time_weights) - np.min(time_weights))) + 1.
        axes[i].plot(normalised_time_weights, color=colors[j], label=f"T={T}")

    handles, labels = axes[i].get_legend_handles_labels()
    axes[i].legend(handles=reversed(handles), labels=reversed(labels), loc='best')

    axes[i].set_xlabel('t')
    axes[i].set_ylim(0, norm + 3)
    axes[i].set_title(f"C={coeff}")

    # plt.tight_layout()

plt.show()
