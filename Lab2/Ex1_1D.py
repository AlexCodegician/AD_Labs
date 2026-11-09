import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
a = 2
b = 3
n = 100

x_compact = np.random.uniform(1, 10, n)
x_extins = np.random.uniform(1, 50, n)

scenarii = {
    "Regulare": (x_compact, a * x_compact + b + np.random.normal(0, 1, n)),
    "Varianta mare pe X": (x_extins, a * x_extins + b + np.random.normal(0, 1, n)),
    "Varianta mare pe Y": (x_compact, a * x_compact + b + np.random.normal(0, 100, n)),
    "Varianta mare pe X si Y": (x_extins, a * x_extins + b + np.random.normal(0, 100, n))
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, (tip, (x, y)) in enumerate(scenarii.items()):
    X_matrice = np.column_stack((np.ones(n), x))
    U, _, _ = np.linalg.svd(X_matrice, full_matrices=False)
    levier = np.sum(U ** 2, axis=1)

    ax = axes[i]
    scatter = ax.scatter(x, y, c = levier)

    x_linie = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_linie, a * x_linie + b, color = 'red')

    ax.set_title(tip)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.colorbar(scatter, ax=ax, label='Levier')

plt.tight_layout()
plt.show()