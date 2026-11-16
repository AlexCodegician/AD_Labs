import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA
import os
import warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings("ignore", message=".*pin_memory.*")

date_antrenament, _ = make_blobs(
    n_samples=[500, 500],
    centers=[[0, 10, 0], [10, 0, 10]],
    cluster_std=1.0,
    n_features=3,
    random_state=42
)

np.random.seed(42)
date_test = np.random.uniform(low=-10, high=20, size=(1000, 3))

model_iforest = IForest(contamination=0.02, random_state=42)
model_iforest.fit(date_antrenament)
scoruri_iforest = model_iforest.decision_function(date_test)

vector_cazuri_dif = [[4], [32, 16], [64, 32, 16]]

for neuroni in vector_cazuri_dif:
    fig, axe = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'projection': '3d'})

    s0 = axe[0].scatter(date_test[:, 0], date_test[:, 1], date_test[:, 2], c=scoruri_iforest, edgecolor='k')
    fig.colorbar(s0, ax=axe[0], label='Scor', shrink=0.6)
    axe[0].set_title("IForest (Referință 3D)")

    model_dif = DIF(contamination=0.02, random_state=42, hidden_neurons=neuroni)
    model_dif.fit(date_antrenament)
    scoruri_dif = model_dif.decision_function(date_test)

    s1 = axe[1].scatter(date_test[:, 0], date_test[:, 1], date_test[:, 2], c=scoruri_dif, edgecolor='k')
    fig.colorbar(s1, ax=axe[1], label='Scor', shrink=0.6)
    axe[1].set_title(f"DIF (Neuroni: {neuroni})")

    for ax in axe:
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_zlabel("X3")

    plt.tight_layout()
    plt.show()

vector_cazuri_loda = [5, 20, 60]

for b in vector_cazuri_loda:
    fig, axe = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'projection': '3d'})

    s0 = axe[0].scatter(date_test[:, 0], date_test[:, 1], date_test[:, 2], c=scoruri_iforest, edgecolor='k')
    fig.colorbar(s0, ax=axe[0], label='Scor', shrink=0.6)
    axe[0].set_title("IForest (Referință 3D)")

    model_loda = LODA(contamination=0.02, random_state=42, n_bins=b)
    model_loda.fit(date_antrenament)
    scoruri_loda = model_loda.decision_function(date_test)

    s2 = axe[1].scatter(date_test[:, 0], date_test[:, 1], date_test[:, 2], c=scoruri_loda, edgecolor='k')
    fig.colorbar(s2, ax=axe[1], label='Scor', shrink=0.6)
    axe[1].set_title(f"LODA (n_bins: {b})")

    for ax in axe:
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_zlabel("X3")

    plt.tight_layout()
    plt.show()