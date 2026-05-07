import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
X_train = pd.read_csv('data/X_train.csv')
y_train = pd.read_csv('data/y_train.csv').values.ravel()
X_val = pd.read_csv('data/X_val.csv')
y_val = pd.read_csv('data/y_val.csv').values.ravel()
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv').values.ravel()
best_k = 1
best_score = 0
for k in [1, 3, 5, 7, 9, 11]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    score = knn.score(X_val, y_val) 
    print(f"k={k} | Accuracy: {score:.4f}")
    if score > best_score:
        best_score = score
        best_k = k
print(f"--- Best k found is: {best_k} with accuracy {best_score:.4f} ---")
final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train, y_train)
final_accuracy = final_model.score(X_test, y_test)
print(f"Final Accuracy on Test Set: {final_accuracy:.4f}")
