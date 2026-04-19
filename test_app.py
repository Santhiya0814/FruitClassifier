import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# 1. LOAD DATASET
df = pd.read_csv("data.csv")

# Clean column names (safe step)
df.columns = df.columns.str.strip()

print("Columns:", df.columns)
X = df[['weight', 'size', 'sweetness']]  
y = df['label']                    

# Convert to numpy
X = X.values
y = y.values

# ==============================
# 3. TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
# ==============================
# 4. TRAIN MODELS
# ==============================

# KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_acc = accuracy_score(y_test, knn.predict(X_test))
print("KNN Accuracy:", round(knn_acc, 2))

# Logistic Regression
lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test))
print("Logistic Regression Accuracy:", round(lr_acc, 2))

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_acc = accuracy_score(y_test, dt.predict(X_test))
print("Decision Tree Accuracy:", round(dt_acc, 2))

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_acc = accuracy_score(y_test, nb.predict(X_test))
print("Naive Bayes Accuracy:", round(nb_acc, 2))

# ==============================
# 5. BEST MODEL
# ==============================
accuracies = {
    "KNN": knn_acc,
    "Logistic Regression": lr_acc,
    "Decision Tree": dt_acc,
    "Naive Bayes": nb_acc
}

best_model = max(accuracies, key=accuracies.get)

print("\nBest Algorithm:", best_model)
print("Best Accuracy:", round(accuracies[best_model], 2))

# ==============================
# 6. TEST NEW DATA
# ==============================
new_fruit = [[160, 7.2, 7]]

if best_model == "KNN":
    prediction = knn.predict(new_fruit)
elif best_model == "Logistic Regression":
    prediction = lr.predict(new_fruit)
elif best_model == "Decision Tree":
    prediction = dt.predict(new_fruit)
else:
    prediction = nb.predict(new_fruit)

print("\nPredicted Fruit:", prediction[0])