import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


X_train = pd.read_csv("data/X_train.csv")
X_val   = pd.read_csv("data/X_val.csv")
X_test  = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv")
y_val   = pd.read_csv("data/y_val.csv")
y_test  = pd.read_csv("data/y_test.csv")

y_train = y_train.values.ravel()
y_val   = y_val.values.ravel()
y_test  = y_test.values.ravel()

unimproved_model = LogisticRegression(random_state=42, max_iter=1000)
unimproved_model.fit(X_train, y_train)
print(f"Validation Accuracy for unimproved model: {unimproved_model.score(X_val, y_val)*100:.2f}%\n")

C_values = [0.001, 0.01, 0.1, 1, 10, 100]
l1_ratios = [0.0, 1.0] 

best_accuracy = 0
best_parameters = []

for C in C_values:
    for l1_ratio in l1_ratios:
        model = LogisticRegression(
            C=C,
            l1_ratio=l1_ratio,
            solver='saga',
            random_state=42,
            max_iter=1000
        )
        model.fit(X_train, y_train)
        accuracy = model.score(X_val, y_val)
        
        penalty_label = 'l1' if l1_ratio == 1.0 else 'l2'
        print(f"C: {C}, penalty: {penalty_label}, Accuracy: {accuracy*100:.2f}%")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_parameters = [C, l1_ratio]

print(f"\nbest accuracy: {best_accuracy*100:.2f}%")
best_penalty_label = 'l1' if best_parameters[1] == 1.0 else 'l2'
print(f"best parameters: C: {best_parameters[0]}, penalty: {best_penalty_label}")

improved_model = LogisticRegression(
    C=best_parameters[0],
    l1_ratio=best_parameters[1],
    solver='saga',
    random_state=42,
    max_iter=1000
)
improved_model.fit(X_train, y_train)
print(f"\nValidation Accuracy for improved model: {improved_model.score(X_val, y_val)*100:.2f}%")

unimproved_model = LogisticRegression(random_state=42, max_iter=1000)
unimproved_model.fit(X_train, y_train)

print(f"Validation Accuracy for unimproved model: {unimproved_model.score(X_val, y_val)*100:.2f}%")
print(f"Test Accuracy for unimproved model:  {unimproved_model.score(X_test, y_test)*100:.2f}%")
print(f"Test Accuracy for improved model:  {improved_model.score(X_test, y_test)*100:.2f}%")