import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

date_antrenament, _ = make_blobs(n_samples=500,
                                 centers=[[0, 0]],
                                 cluster_std=1.0,
                                 n_features=2)
np.random.seed(42)
date_test = np.random.uniform(low=-3, high=3, size=(500, 2))

vectori_proiectie = []
for _ in range(5):
    vector_aleator = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]])
    vector_normat = vector_aleator / np.linalg.norm(vector_aleator)
    vectori_proiectie.append(vector_normat)

date_ant_proiectate = [np.dot(date_antrenament, vector) for vector in vectori_proiectie]
#print(date_ant_proiectate)

valori_binuri = [5, 10, 30, 50]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, nr_binuri in enumerate(valori_binuri):
    lista_probabilitati = []
    lista_margini = []

    for proiectie in date_ant_proiectate:
        numar_elemente, margini = np.histogram(proiectie, bins=nr_binuri, range=(-4, 4))
        probabilitati = numar_elemente / len(date_antrenament)
        lista_probabilitati.append(probabilitati)
        lista_margini.append(margini)

    scoruri_anomalie = []
    for punct in date_test:
        probabilitati_punct = []
        for i, vector in enumerate(vectori_proiectie):
            valoare_proiectata = np.dot(punct, vector)
            margini = lista_margini[i]
            probabilitati = lista_probabilitati[i]

            index_bin = np.digitize(valoare_proiectata, margini) - 1
            if 0 <= index_bin < len(probabilitati):
                prob = probabilitati[index_bin]
            else:
                prob = 0.0
            probabilitati_punct.append(prob)

        scor_anomalie = 1.0 - np.mean(probabilitati_punct)
        scoruri_anomalie.append(scor_anomalie)

    scoruri_anomalie = np.array(scoruri_anomalie)

    ax = axes[idx]
    scatter = ax.scatter(date_test[:, 0], date_test[:, 1], c=scoruri_anomalie, edgecolor='k')
    ax.set_title(f'LODA cu {nr_binuri} bin-uri')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    fig.colorbar(scatter, ax=ax, label='Scor')

plt.tight_layout()
plt.show()

## Se observa in grafic ca la un nr mic de binuri (5) tindem sa facem underfitting, in schimb
## pe un nr mai mare (30, 50) facem overfitting. Cumva valoare optima e in jurul a 10 binuri.