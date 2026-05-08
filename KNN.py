import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

X_train = pd.read_csv('data/X_train.csv')
y_train = pd.read_csv('data/y_train.csv').values.ravel()
X_val = pd.read_csv('data/X_val.csv')
y_val = pd.read_csv('data/y_val.csv').values.ravel()
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv').values.ravel()


best_k = 1
best_weight = 'uniform'
best_score = 0


for k in [3, 5, 7, 9, 11]:
    for w in ['uniform', 'distance']:
      
        knn = KNeighborsClassifier(n_neighbors=k, weights=w)
        
     
        knn.fit(X_train, y_train)
        
       
        score = knn.score(X_val, y_val)
        
        print(f"Testing: k={k}, weights={w} | Accuracy: {score:.4f}")
        
    
        if score > best_score:
            best_score = score
            best_k = k
            best_weight = w

print("-" * 30)
print(f"DONE! Best Hyperparameters: k={best_k}, weights='{best_weight}'")
print(f"Best Validation Accuracy: {best_score:.4f}")
print("-" * 30)



final_model = KNeighborsClassifier(n_neighbors=best_k, weights=best_weight)
final_model.fit(X_train, y_train)


final_acc = final_model.score(X_test, y_test)
print(f"Final Model Accuracy on Test Set: {final_acc:.4f}")