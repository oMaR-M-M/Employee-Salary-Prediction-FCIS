import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

X_train = pd.read_csv("data/X_train.csv")
X_val = pd.read_csv("data/X_val.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").values.ravel()
y_val = pd.read_csv("data/y_val.csv").values.ravel()
y_test = pd.read_csv("data/y_test.csv").values.ravel()

max_depth = [2, 4, 6]
n_estimators = [100, 500, 700, 1000] #num of the trees
learning_rate = [0.01, 0.05, 0.1]

accuracy = 0
ratio = 0
best_param = {}

print('-'*100)
print('-'*100)


for mx_d in max_depth:
    print(f'at max_depth = {mx_d}: ')
    for n_est in n_estimators:
        print(f'    at number of estimators = {n_est}: ')
        for lr in learning_rate:
            print(f'        at learning rate = {lr}: ')
            XGboost = XGBClassifier(
                max_depth = mx_d,
                n_estimators = n_est,
                learning_rate = lr,
                verbosity = 0,
                random_state = 433,
                subsample = 0.7,
                colsample_bytree = 0.7,
                eval_metric='auc'
            )
            XGboost.fit(X_train, y_train)
            accuracy_train = XGboost.score(X_train, y_train)
            accuracy_val = XGboost.score(X_val, y_val)
            print(f'            accuracy of train= {accuracy_train} while accuracy of val= {accuracy_val}')
            if accuracy_val > accuracy:
                accuracy = accuracy_val
                ratio = accuracy_val / accuracy_train
                best_param = {
                    'max_depth' : mx_d,
                    'n_estimators' : n_est,
                    'learning_rate' : lr
                }
    print('-'*100)

print(f'the val accuracy  = {accuracy} with ratio = {ratio}')
print(f'with paramitars {best_param['max_depth']}, {best_param['n_estimators']}, {best_param['learning_rate']}')
print('-'*100)
print('-'*100)

XGboost_model = XGBClassifier(
                max_depth = best_param['max_depth'],
                n_estimators = best_param['n_estimators'],
                learning_rate = best_param['learning_rate'],
                verbosity = 0,
                random_state = 433,
                subsample = 0.7,
                colsample_bytree = 0.7,
                eval_metric='auc'
)

y_train = pd.read_csv("data/y_train.csv")
y_val = pd.read_csv("data/y_val.csv")
X = pd.concat([X_train, X_val])
y = pd.concat([y_train, y_val])
y = y.values.ravel()
XGboost_model.fit(X, y)
print(f'the accuracy of the model = {XGboost_model.score(X_test, y_test)}')
print('-'*100)
print('-'*100)

y_hat = XGboost_model.predict(X_test)

y_prob = XGboost_model.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_hat)
auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy: {acc * 100:.2f}%")
print(f"AUC: {auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_hat))