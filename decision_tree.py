import pandas as pd
from sklearn.tree import DecisionTreeClassifier
X_train = pd.read_csv("data/X_train.csv")
X_val   = pd.read_csv("data/X_val.csv")
X_test  = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv")
y_val   = pd.read_csv("data/y_val.csv")
y_test  = pd.read_csv("data/y_test.csv")
y_train = y_train.values.ravel()
y_val   = y_val.values.ravel()
y_test  = y_test.values.ravel()
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
print(f"Validation Accuracy for unimproved model: {model.score(X_val, y_val)*100:.2f}%")
max_depth = [ 5, 10, 15, 20]
min_samples_split = [2, 5, 10, 20]
min_samples_leaf = [1, 5, 10, 20]
best_accuracy = 0
best_parameters = []
for depth in max_depth:
    for split in min_samples_split:
        for leaf in min_samples_leaf:
            model = DecisionTreeClassifier(
                max_depth=depth,
                min_samples_split=split,
                min_samples_leaf=leaf,
                random_state=42
            )
            model.fit(X_train, y_train)
            accuracy = model.score(X_val, y_val)
            print(f"max depth: {depth}, min samples split: {split},min samples leaf: {leaf}, Accuracy: {accuracy*100:.2f}%")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_parameters = [depth,split,leaf]
print(f"best accuracy: {best_accuracy*100:.2f}%")
print(f"best parameters: max depth: {best_parameters[0]}, min samples split: {best_parameters[1]},min samples leaf: {best_parameters[2]}")
improved_model= DecisionTreeClassifier(
    max_depth=best_parameters[0],
    min_samples_split=best_parameters[1],
    min_samples_leaf=best_parameters[2],
    random_state=42
)
improved_model.fit(X_train, y_train)
print(f"Validation Accuracy for improved model: {improved_model.score(X_val, y_val)*100:.2f}%")
unimproved_model = DecisionTreeClassifier(random_state=42)
unimproved_model.fit(X_train, y_train)
print(f"Validation Accuracy for unimproved model: {unimproved_model.score(X_val, y_val)*100:.2f}%")
print(f"Test Accuracy for unimproved model:  {unimproved_model.score(X_test, y_test)*100:.2f}%")
print(f"Test Accuracy for improved model:  {improved_model.score(X_test, y_test)*100:.2f}%")
