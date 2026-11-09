import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from pyod.models.knn import KNN
from pyod.models.lof import LOF

x, y = make_blobs(n_samples = [200, 100], centers = [(-10, -10), (10, 10)],
                  cluster_std = [2.0, 6.0], n_features = 2, random_state = 42)

rata_contaminare = 0.07
nr_vecini = [1, 3, 10, 25]

for k in nr_vecini:
    clf_knn = KNN(n_neighbors = k, contamination = rata_contaminare)
    clf_knn.fit(x)
    y_pred_knn = clf_knn.labels_

    clf_lof = LOF(n_neighbors = k, contamination = rata_contaminare)
    clf_lof.fit(x)
    y_pred_lof = clf_lof.labels_

    fig, axes = plt.subplots(1, 2, figsize=(12,6))
    fig.suptitle(f"Comparatie KNN vs LOF (n_neighbors = {k})", fontsize=16, fontweight='bold')

    axes[0].scatter(x[:, 0], x[:, 1], c = y_pred_knn)
    axes[0].set_title("Predictii KNN")
    axes[0].set_xlabel("X1")
    axes[0].set_ylabel("X2")

    axes[1].scatter(x[:, 0], x[:, 1], c = y_pred_lof)
    axes[1].set_title("Predictii LOF")
    axes[1].set_xlabel("X1")
    axes[1].set_xlabel("X1")
    axes[1].set_ylabel("X2")

plt.tight_layout()
plt.show()

##Indiferent de valoare lui k, KNN isi pastreaza aceeasi limitare majora din cauza
##ca se bazeaza pe distante globale. In schimb, in cazul LOF, cresterea numarului de
##vecini are un impact benefic, obtinand o delimitare precisa a anomaliilor.