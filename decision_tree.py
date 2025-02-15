# -*- coding: utf-8 -*-
"""decision tree.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Hq339zyER936plVpFTU2fgr6YQcWdb4P
"""

#### decision tree
! pip install mglearn
import mglearn


mglearn.plots.plot_animal_tree()

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)

tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
tree.score(X_test, y_test), tree.score(X_train, y_train)

### pre pruning example

tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)
tree.score(X_test, y_test), tree.score(X_train, y_train)

result_range = []
result_test = []
result_train = []

for x in range(1, 10):
  tree = DecisionTreeClassifier(max_depth=x, random_state=0)
  tree.fit(X_train, y_train)
  print(x, tree.score(X_test, y_test), tree.score(X_train, y_train))
  result_range.append(x)
  result_test.append(tree.score(X_test, y_test))
  result_train.append(tree.score(X_train, y_train))

import matplotlib.pyplot as plt

plt.plot(result_range, result_test, label='test')
plt.plot(result_range, result_train, label='train')
plt.legend()
plt.show()

import pandas as pd

from sklearn.tree import DecisionTreeClassifier

data_train = ra