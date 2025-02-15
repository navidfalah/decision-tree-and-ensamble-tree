# 🌳 Decision Trees and Ensembles

Welcome to the **Decision Trees and Ensembles** repository! This project explores the use of **Decision Trees**, **Random Forests**, and **Gradient Boosting Machines (GBMs)** for classification tasks. It also covers feature importance and model evaluation.

---

## 📂 **Project Overview**

This repository demonstrates how to build and evaluate decision trees and ensemble models using **Scikit-learn**. It includes:

- **Decision Trees**: Building and pruning decision trees.
- **Random Forests**: Combining multiple trees for improved performance.
- **Gradient Boosting Machines (GBMs)**: Sequential building of trees to minimize errors.
- **Feature Importance**: Identifying the most important features in a dataset.

---

## 🛠️ **Tech Stack**

- **Python**
- **Scikit-learn**
- **mglearn**
- **NumPy**
- **Matplotlib**

---

## 📊 **Datasets**

The project uses the following datasets:
- **Breast Cancer Dataset**: For binary classification with decision trees and ensembles.
- **Moons Dataset**: For visualizing decision boundaries.

---

## 🧠 **Key Concepts**

### 1. **Decision Trees**
- A tree-like model for classification and regression.
- Can overfit if not pruned properly.
- Pruning techniques like `max_depth` help control overfitting.

### 2. **Random Forests**
- An ensemble of decision trees.
- Reduces overfitting by averaging predictions from multiple trees.
- Provides feature importance scores.

### 3. **Gradient Boosting Machines (GBMs)**
- Builds trees sequentially to correct errors from previous trees.
- Sensitive to hyperparameters like `learning_rate` and `max_depth`.

### 4. **Feature Importance**
- Measures the contribution of each feature to the model's predictions.
- Helps in understanding the dataset and model behavior.

---

## 🚀 **Code Highlights**

### Building a Decision Tree
```python
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("Test score:", tree.score(X_test, y_test))
```

### Pruning a Decision Tree
```python
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)
print("Test score:", tree.score(X_test, y_test))
```

### Building a Random Forest
```python
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)
print("Test score:", forest.score(X_test, y_test))
```

### Building a Gradient Boosting Machine
```python
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)
print("Test score:", gbrt.score(X_test, y_test))
```

### Visualizing Feature Importance
```python
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.show()

plot_feature_importances_cancer(forest)
```

---

## 🛠️ **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/navidfalah/decision-trees-ensembles.git
   cd decision-trees-ensembles
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   ```bash
   jupyter notebook decision_tree.ipynb
   ```

---

## 🤝 **Contributing**

Feel free to contribute to this project! Open an issue or submit a pull request.

---

## 📧 **Contact**

- **Name**: Navid Falah
- **GitHub**: [navidfalah](https://github.com/navidfalah)
- **Email**: navid.falah7@gmail.com
