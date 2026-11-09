import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
a1 = 1.5
a2 = -2.0
b = 3
n = 100

x_compact = np.random.uniform(1, 10, n)
x_extins = np.random.uniform(1, 50, n)

noise_mic = np.random.normal(0, 1, n)
noise_mare = np.random.normal(0, 50, n)

scenarii = {
    "Varianta mare pe X1": ((x_extins, x_compact), a1 * x_extins + a2 * x_compact + b + noise_mic),
    "Varianta mare pe X2": ((x_compact, x_extins), a1 * x_compact + a2 * x_extins + b + noise_mic),
    "Varianta mare pe Y": ((x_compact, x_compact), a1 * x_compact + a2 * x_compact + b + noise_mare),
    "Varianta mare pe Toate": ((x_extins, x_extins), a1 * x_extins + a2 * x_extins + b + noise_mare)
}

fig, axes = plt.subplots(2, 2, figsize=(14, 12), subplot_kw={'projection': '3d'})
axes = axes.ravel()

for i, (tip, ((x1, x2), y)) in enumerate(scenarii.items()):
    X_matrice = np.column_stack((np.ones(n), x1, x2))
    U, _, _ = np.linalg.svd(X_matrice, full_matrices=False)
    levier = np.sum(U ** 2, axis=1)

    ax = axes[i]
    scatter = ax.scatter(x1, x2, y, c=levier)

    xx1, xx2 = np.meshgrid(np.linspace(x1.min(), x1.max(), 15),
                           np.linspace(x2.min(), x2.max(), 15))
    ax.plot_surface(xx1, xx2, a1 * xx1 + a2 * xx2 + b, color='red')

    ax.set_title(tip)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("Y")
    plt.colorbar(scatter, ax=ax, label='Levier')

plt.tight_layout()
plt.show()