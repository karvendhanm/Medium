import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random

from architectural_priors import utils
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

# cache the downloaded dataset into 'data_home'
mnist = fetch_openml('mnist_784', version=1, cache=True)

data = mnist['data']
target = mnist['target']

X_train, X_test, y_train, y_test = data[:5000], data[5000:], target[:5000], target[5000:]
X_train = X_train.sample(frac=1, random_state=42).reset_index(drop=True)
y_train = y_train.sample(frac=1, random_state=42).reset_index(drop=True)

# initiate sgd classifier.
sgd_clf = SGDClassifier(random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)

cl_a, cl_b = '3', '5'
X_aa = X_train.loc[(y_train == cl_a) & (y_train_pred == cl_a), :]
X_bb = X_train.loc[(y_train == cl_b) & (y_train_pred == cl_b), :]
X_ab = X_train.loc[(y_train == cl_a) & (y_train_pred == cl_b), :]
X_ba = X_train.loc[(y_train == cl_b) & (y_train_pred == cl_a), :]

plt.figure(figsize=(8, 8))
plt.subplot(221)
utils.plot_digits(X_aa.iloc[:23, :], images_per_row=5)
plt.subplot(222)
utils.plot_digits(X_ab.iloc[:23, :], images_per_row=5)
plt.subplot(223)
utils.plot_digits(X_ba.iloc[:23, :], images_per_row=5)
plt.subplot(224)
utils.plot_digits(X_bb.iloc[:23, :], images_per_row=5)
plt.show()
