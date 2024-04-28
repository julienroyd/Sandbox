def softmax(x, T):
    x = x - np.max(x)  # for numerical stability
    return np.exp(x / T) / np.sum(np.exp(x / T))


x1 = [1, 2, 3]
x2 = [1, 1.1, 1.2]
x3 = [100, 101, 102]

xs = [x1, x2, x3]

Ts = [0.1, 0.2, 0.5, 1., 2., 5., 10.]

fig, axes = plt.subplots(1, len(xs), figsize=(8 * len(xs), 5))
cm = plt.cm.get_cmap('viridis')
colors = [np.array(cm(float(k) / float(len(Ts)))) for k, _ in enumerate(Ts)]

for i, x in enumerate(xs):
    for j, T in enumerate(Ts):
        axes[i].bar(np.arange(len(x)) + 0.1 * j, softmax(x, T), width=0.1, color=colors[j], label=f"T={T}")

    handles, labels = axes[i].get_legend_handles_labels()
    axes[i].legend(handles=reversed(handles), labels=reversed(labels), loc='upper left')

    axes[i].set_ylim(0, 1)
    axes[i].set_title(f"x={x}")

# plt.tight_layout()
fig.savefig('test.png')
plt.show()