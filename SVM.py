import pandas as pd
from sklearn.svm import SVC

X_train = pd.read_csv('data/X_train.csv')
y_train = pd.read_csv('data/y_train.csv').values.ravel()
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv').values.ravel()
X_val = pd.read_csv('data/X_val.csv')
y_val = pd.read_csv('data/y_val.csv').values.ravel()
# kernels = ['linear', 'poly', 'rbf', 'sigmoid']
# c_values = [0.1, 1, 10]
# degrees = [1,2, 3, 4]
# for kernel in kernels:
#     for c in c_values:
#         if kernel == 'poly':
#             for degree in degrees:
#                 SVM = SVC(kernel=kernel, C=c, degree=degree, random_state=42)
#                 SVM.fit(X_train, y_train)
#                 accuracy = SVM.score(X_val, y_val)
#                 print(f"Kernel: {kernel}, C: {c}, Degree: {degree}, Accuracy: {accuracy * 100}%")
#         else:
#             SVM = SVC(kernel=kernel, C=c, random_state=42)
#             SVM.fit(X_train, y_train)
#             accuracy = SVM.score(X_val, y_val)
#             print(f"Kernel: {kernel}, C: {c}, Accuracy: {accuracy * 100}%")
SVM = SVC(kernel='linear', C=0.1, random_state=42)
SVM.fit(X_test, y_test)
accuracy = SVM.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100}%")