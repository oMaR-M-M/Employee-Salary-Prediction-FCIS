import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


df_train = pd.read_csv("data/catboost_data_train.csv")
df_ = pd.read_csv("data/catboost_data_test.csv")

X_train = df_train.drop(columns='salary')
y_train = df_train['salary'].values.ravel()

X_ = df_.drop(columns='salary')
y_ = df_['salary'].values.ravel()

X_test , X_val, y_test, y_val = train_test_split(X_, y_, train_size=0.5, random_state=42)
print('-'*100)


cat_features = X_train.select_dtypes(include=['object']).columns.tolist()

model = CatBoostClassifier(
    iterations=4000,
    learning_rate=0.005,
    depth=8,
    loss_function='Logloss',
    eval_metric='AUC',
    verbose=1, 
)


model.fit(X_train,
          y_train,
          cat_features=cat_features,
          eval_set=(X_val, y_val),
          early_stopping_rounds=50
)

print(model.score(X_train, y_train))
print(model.score(X_val, y_val))

print('-'*100)
print('-'*100)
print('-'*100)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy: {acc * 100:.2f}%")
print(f"AUC: {auc:.2f}")
print('-'*100)


print("\nClassification Report:")
print(classification_report(y_test, y_pred))