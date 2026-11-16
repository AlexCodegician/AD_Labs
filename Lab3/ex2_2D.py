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
    centers=[[10, 0], [0, 10]],
    cluster_std=1.0,
    n_features=2,
    random_state=42
)

np.random.seed(42)
date_test = np.random.uniform(low=-10, high=20, size=(1000,2))

model_iforest = IForest(contamination=0.02, random_state=42)
model_iforest.fit(date_antrenament)
scoruri_iforest = model_iforest.decision_function(date_test)

vector_cazuri_dif = [[4], [32, 16], [64, 32, 16]]

for neuroni in vector_cazuri_dif:
    fig, axe = plt.subplots(1, 2, figsize=(12, 6))

    s0 = axe[0].scatter(date_test[:, 0], date_test[:, 1], c=scoruri_iforest, edgecolor='k')
    fig.colorbar(s0, ax=axe[0], label='Scor')
    axe[0].set_title("IForest (Referință)")

    model_dif = DIF(contamination=0.02, random_state=42, hidden_neurons=neuroni)
    model_dif.fit(date_antrenament)
    scoruri_dif = model_dif.decision_function(date_test)

    s1 = axe[1].scatter(date_test[:, 0], date_test[:, 1], c=scoruri_dif, edgecolor='k')
    fig.colorbar(s1, ax=axe[1], label='Scor')
    axe[1].set_title(f"DIF (Neuroni: {neuroni})")

    for ax in axe:
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

vector_cazuri_loda = [5, 20, 60]

for b in vector_cazuri_loda:
    fig, axe = plt.subplots(1, 2, figsize=(12, 6))

    s0 = axe[0].scatter(date_test[:, 0], date_test[:, 1], c=scoruri_iforest, edgecolor='k')
    fig.colorbar(s0, ax=axe[0], label='Scor')
    axe[0].set_title("IForest (Referință)")

    model_loda = LODA(contamination=0.02, random_state=42, n_bins=b)
    model_loda.fit(date_antrenament)
    scoruri_loda = model_loda.decision_function(date_test)

    s2 = axe[1].scatter(date_test[:, 0], date_test[:, 1], c=scoruri_loda, edgecolor='k')
    fig.colorbar(s2, ax=axe[1], label='Scor')
    axe[1].set_title(f"LODA (n_bins: {b})")

    for ax in axe:
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

## Cum IForest face doar taieturi verticale si orizontale, iar cum cele doua clustere sunt puse fix pe diagonala
## se creeaza inca 2 clustere "artificiale", iar punctele din jurul acestora primesc scoruri mai mici (par mai normale)
## decat ar trebui, desi sunt departe de date.

## DIF proiecteaza datele prin rețele neuronale aleatoare, iar acele taieturi paralele cu axele nu mai exista,
## iar artefactele sunt atenuate.

## LODA foloseste vectori de proiectie pentru a proiecta datele pe directii aleatoare si estimeaza o histograma 1D
## pe fiecare dintre acestea. Scorul unui punct e media (log-)probabilitatilor din aceste histograme.
## Cu putine binuri avem underfitting (bias mare), in timp ce  cu multe binuri tindem spre overfitting
## pentru ca unele binuri au putine puncte. Punctele din afara range-ului histogramelor au probabilitate 0,
## deci scor maxim (anomalii).