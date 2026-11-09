import matplotlib.pyplot as plt
from pyod.models.knn import KNN
from pyod.utils.data import generate_data_clusters
from sklearn.metrics import balanced_accuracy_score

X_ant, X_test, y_ant, y_test = generate_data_clusters(
    n_train=400,
    n_test=200,
    n_clusters=2,
    n_features=2,
    contamination=0.1
)

nr_vecini = [1, 3, 10, 25]

for k in nr_vecini:
    clf = KNN(n_neighbors = k, contamination = 0.1)
    clf.fit(X_ant)

    y_ant_pred = clf.labels_
    y_test_pred = clf.predict(X_test)

    scor_train = balanced_accuracy_score(y_ant, y_ant_pred)
    scor_test = balanced_accuracy_score(y_test, y_test_pred)

    print(k, ":", scor_train)
    print(k, ":", scor_test)

    fig, axes = plt.subplots(2, 2, figsize=(12,10))
    fig.suptitle(f"Evaluare KNN cu n_neighbors (k) = {k}", fontsize = 16, fontweight= 'bold')

    axes[0, 0].scatter(X_ant[:, 0], X_ant[:, 1], c = y_ant)
    axes[0, 0].set_title("Date reale antrenare")

    axes[0, 1].scatter(X_ant[:, 0], X_ant[:, 1], c = y_ant_pred)
    axes[0, 1].set_title("Predictii antrenare")

    axes[1, 0].scatter(X_test[:, 0], X_test[:, 1], c = y_test)
    axes[1, 0].set_title("Date reale test")

    axes[1, 1].scatter(X_test[:, 0], X_test[:, 1], c = y_test_pred)
    axes[1, 1].set_title("Predictii test")

plt.tight_layout()
plt.show()

##Cu cat crestem nr. de vecini, preciziile devin mai mari deoarece atunci cand avem
##un nr mai mic de vecini, modelul e mai sensibil la zgomot si la mici variatii


