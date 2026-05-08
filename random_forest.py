from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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

# hyperparameters chosen to try:
# n_estimators (number of trees)
# max_depth (max depth of trees)
# min_samples_leaf (how many samples minimum per leaf)
# max_features (how many features to consider per tree)

# truncated for training time
# 4 * 4 * 3 * 3 = 144 combinations

RANDOM_STATE = 42
N_JOBS = -1

params: dict = {
    "n_estimators": [1, 55, 100, 200],
    "max_depth": list(range(5, 36, 15)) + [None],
    "min_samples_leaf": list(range(1, 12, 5)),
    "max_features": ["sqrt", "log2", None],
}

# --- tune model ---

default_model = RandomForestClassifier(
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS
)
default_model.fit(X_train, y_train)

tuned_model = RandomForestClassifier()

best_score = -1
best_params: dict = {
    "n_estimators": 1,
    "max_depth": 1,
    "min_samples_leaf": 1,
    "max_features": None
}

for chosen_n_estimators in params["n_estimators"]:
    for chosen_max_depth in params["max_depth"]:
        for chosen_min_samples_leaf in params["min_samples_leaf"]:
            for chosen_max_features in params["max_features"]:
                tuned_model = RandomForestClassifier(
                    n_estimators=chosen_n_estimators,
                    max_depth=chosen_max_depth,
                    min_samples_leaf=chosen_min_samples_leaf,
                    max_features=chosen_max_features,
                    random_state=RANDOM_STATE,
                    n_jobs=N_JOBS
                )
                tuned_model.fit(X_train, y_train)

                predictions = tuned_model.predict(X_val)
                score = accuracy_score(y_val, predictions)

                print(f"n_estimators: {chosen_n_estimators}, max_depth: {chosen_max_depth}, min_samples_leaf: {chosen_min_samples_leaf}, max_features: {chosen_max_features}, score: {score*100:.2f}%")

                if score > best_score:
                    best_score = score
                    best_params.update({
                        "n_estimators": chosen_n_estimators,
                        "max_depth": chosen_max_depth,
                        "min_samples_leaf": chosen_min_samples_leaf,
                        "max_features": chosen_max_features
                    })

tuned_model = RandomForestClassifier(
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    min_samples_leaf=best_params["min_samples_leaf"],
    max_features=best_params["max_features"],
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS
)
tuned_model.fit(X_train, y_train)

default_validation_predictions = default_model.predict(X_val)
default_validation_score = accuracy_score(y_val, default_validation_predictions)

tuned_validation_predictions = tuned_model.predict(X_val)
tuned_validation_score = accuracy_score(y_val, tuned_validation_predictions)

print('\n------------------------------------------')

print(f'\nDefault model\'s validation accuracy: {default_validation_score*100:.2f}%')
print(f'  Using parameters: n_estimators: {default_model.n_estimators}, max_depth: {default_model.max_depth}, min_samples_leaf: {default_model.min_samples_leaf}, max_features: {default_model.max_features}')

print(f'Tuned model\'s validation accuracy: {tuned_validation_score*100:.2f}%')
print(f'  Using parameters: n_estimators: {best_params["n_estimators"]}, max_depth: {best_params["max_depth"]}, min_samples_leaf: {best_params["min_samples_leaf"]}, max_features: {best_params["max_features"]}')

# --- test model ---

default_test_predictions = default_model.predict(X_test)
default_test_score = accuracy_score(y_test, default_test_predictions)

tuned_test_predictions = tuned_model.predict(X_test)
tuned_test_score = accuracy_score(y_test, tuned_test_predictions)

print(f'\nDefault model\'s test accuracy: {default_test_score*100:.2f}%')
print(f'  Using parameters: n_estimators: {default_model.n_estimators}, max_depth: {default_model.max_depth}, min_samples_leaf: {default_model.min_samples_leaf}, max_features: {default_model.max_features}')

print(f'Tuned model\'s test accuracy: {tuned_test_score*100:.2f}%')
print(f'  Using parameters: n_estimators: {best_params["n_estimators"]}, max_depth: {best_params["max_depth"]}, min_samples_leaf: {best_params["min_samples_leaf"]}, max_features: {best_params["max_features"]}')
