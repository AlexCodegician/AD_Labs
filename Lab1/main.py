import pyod
import matplotlib.pyplot as plt
from pyod.models.knn import KNN
from sklearn.metrics import confusion_matrix, roc_curve, balanced_accuracy_score
import numpy as np

### Ex 1
dataset = pyod.utils.data.generate_data(n_train=400, n_test=100, contamination=0.1)
#print(dataset)
train_data = dataset[0]
test_data = dataset[1]
train_labels = dataset[2]
test_labels = dataset[3]

indices_inliers = [i for i,val in enumerate(train_labels) if not val]
indices_outliers = [i for i, val in enumerate(train_labels) if val]
inliers = train_data[indices_inliers]
outliers = train_data[indices_outliers]

plt.scatter(inliers[:, 0], inliers[:, 1], c = 'orange')
plt.scatter(outliers[:, 0], outliers[:, 1], c = 'yellow')
#plt.show()

### Ex 2
clf = KNN(contamination=0.1)
clf.fit(train_data)

train_data_pred = clf.predict(train_data)
test_data_pred = clf.predict(test_data)

cm_train = confusion_matrix(train_labels, train_data_pred)
cm_test = confusion_matrix(test_labels, test_data_pred)

ba_train = balanced_accuracy_score(train_labels, train_data_pred)
ba_test = balanced_accuracy_score(test_labels, test_data_pred)

#print(cm_train)
#print(ba_train)
#print(cm_test)
#print(ba_test)

decision_function = clf.decision_function(test_data)
#print(decision_function)
fpr, tpr, thresholds = roc_curve(test_labels, decision_function)
plt.plot(fpr, tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
#plt.show()

#When we increase the contamination rate, we catch nearly all or all of the real anomalies, but also more unreal ones,
#so the number of FP's increases, and on contrary case, when we decrease the contamination rate, the model will mark as
#anomalies only the most isolated points from the rest, so the number of FN's will increase.
#I think the 0.1 contamination rate is a "sweet spot"!

###Ex 3
unidimensional_dataset = pyod.utils.data.generate_data(n_train=1000, n_test=0, n_features=1, contamination=0.1)
#print(unidimensional_dataset)
ud_train_data = unidimensional_dataset[0]
ud_train_data_labels = unidimensional_dataset[2]

mean = np.mean(ud_train_data)
std = np.std(ud_train_data)
z_scores = (ud_train_data - mean) / std
#print(z_scores)

z_scores = np.abs(z_scores)
threshold = np.quantile(z_scores, 0.9)

anomalies = (z_scores > threshold).astype(int)
ba_ud = balanced_accuracy_score(ud_train_data_labels, anomalies)

print(ba_ud)
