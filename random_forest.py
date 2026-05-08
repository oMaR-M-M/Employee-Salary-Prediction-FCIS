# hyperparameters chosen to try:
# n_estimators (number of trees)
# max_depth (max depth of trees)
# min_samples_leaf (how many samples minimum per leaf)
# max_features (how many features to consider per tree)

from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# --- read data ---

# .read_csv() returns a pandas DataFrame
# .to_numpy() or .values converts a DataFrame to a numpy array
# .ravel() flattens a numpy array to a 1D numpy array

X_train = read_csv("data/X_train.csv")
y_train = read_csv("data/y_train.csv").to_numpy().ravel()

X_val   = read_csv("data/X_val.csv")
y_val   = read_csv("data/y_val.csv").to_numpy().ravel()

X_test  = read_csv("data/X_test.csv")
y_test  = read_csv("data/y_test.csv").to_numpy().ravel()

# --- select hyperparameters' range ---

# truncated for training time
# 4 * 4 * 3 * 3 = 144

params = {
    "n_estimators": [1, 55, 100, 200],
    "max_depth": list(range(5, 36, 15)) + [None],
    "min_samples_leaf": list(range(1, 12, 5)),
    "max_features": ["sqrt", "log2", None],
    "random_state": [42],
    "n_jobs": [-1]
}

# --- tune model ---

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

tuned_model = RandomForestClassifier()

best_score = -1
best_params = [1, 1, 1, None]

# grid = GridSearchCV(
#     RandomForestClassifier(),
#     params,
#     n_jobs=-1,
# )

# grid.fit(X_train, y_train)
# tuned_model = grid.best_estimator_
# best_score = tuned_model.score(X_val, y_val)
# best_params = [tuned_model.n_estimators, tuned_model.max_depth, tuned_model.min_samples_leaf, tuned_model.max_features]

for chosen_n_estimators in params["n_estimators"]:
    for chosen_max_depth in params["max_depth"]:
        for chosen_min_samples_leaf in params["min_samples_leaf"]:
            for chosen_max_features in params["max_features"]:
                tuned_model = RandomForestClassifier(
                    n_estimators=chosen_n_estimators,
                    max_depth=chosen_max_depth,
                    min_samples_leaf=chosen_min_samples_leaf,
                    max_features=chosen_max_features,
                    random_state=params["random_state"][0],
                    n_jobs=params["n_jobs"][0]
                )
                tuned_model.fit(X_train, y_train)

                score = tuned_model.score(X_val, y_val)

                print(f"n_estimators: {chosen_n_estimators}, max_depth: {chosen_max_depth}, min_samples_leaf: {chosen_min_samples_leaf}, max_features: {chosen_max_features}, score: {score*100:.2f}%")

                if score > best_score:
                    best_score = score
                    best_params = [chosen_n_estimators, chosen_max_depth, chosen_min_samples_leaf, chosen_max_features]

tuned_model = RandomForestClassifier(
    n_estimators=best_params[0],
    max_depth=best_params[1],
    min_samples_leaf=best_params[2],
    max_features=best_params[3],
    random_state=params["random_state"][0],
    n_jobs=params["n_jobs"][0]
)
tuned_model.fit(X_train, y_train)

print("\nValidation")
print("  Default model")
print(f"    Best score: {model.score(X_val, y_val)*100:.2f}%")
print(f"    Using parameters: n_estimators: {model.n_estimators}, max_depth: {model.max_depth}, min_samples_leaf: {model.min_samples_leaf}, max_features: {model.max_features}")

print("  Tuned model")
print(f"    Best score: {best_score*100:.2f}%")
print(f"    Using parameters: n_estimators: {best_params[0]}, max_depth: {best_params[1]}, min_samples_leaf: {best_params[2]}, max_features: {best_params[3]}")

# --- test model ---

print("\nTesting")
print("  Default model")
print(f"    Best score: {model.score(X_test, y_test)*100:.2f}%")
print(f"    Using parameters: n_estimators: {model.n_estimators}, max_depth: {model.max_depth}, min_samples_leaf: {model.min_samples_leaf}, max_features: {model.max_features}")

print("  Tuned model")
print(f"    Best score: {tuned_model.score(X_test, y_test)*100:.2f}%")
print(f"    Using parameters: n_estimators: {best_params[0]}, max_depth: {best_params[1]}, min_samples_leaf: {best_params[2]}, max_features: {best_params[3]}")