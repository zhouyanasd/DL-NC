# ----------------------------------------
# Logistic Regression for class balance using scikit-learn
# ----------------------------------------

from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=4)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

X_train = X[:-200]
X_test = X[-200:]
y_train = y[:-200]
y_test = y[-200:]

lr.fit(X_train, y_train)
y_train_predictions = lr.predict(X_train)
y_test_predictions = lr.predict(X_test)

print ("train ACC:",(y_train_predictions == y_train).sum().astype(float) / y_train.shape[0])
print ("test ACC:",(y_test_predictions == y_test).sum().astype(float) / y_test.shape[0])

X, y = make_classification(n_samples=5000, n_features=4, weights=[.95])
print("check the unbalance:", sum(y) / (len(y)*1.)) #检查不平衡的类型

X_train = X[:-500]
X_test = X[-500:]
y_train = y[:-500]
y_test = y[-500:]

lr.fit(X_train, y_train)
y_train_predictions = lr.predict(X_train)
y_test_predictions = lr.predict(X_test)

print ("train ACC:",(y_train_predictions == y_train).sum().astype(float) / y_train.shape[0])
print ("test ACC:",(y_test_predictions == y_test).sum().astype(float) / y_test.shape[0])

print ((y_test[y_test==1] == y_test_predictions[y_test==1]).sum().astype(float) / y_test[y_test==1].shape[0])

lr = LogisticRegression(class_weight={0: .15, 1: .85})#多重采样(oversample)
lr.fit(X_train, y_train)

# LogisticRegression(C=1.0, class_weight={0: 0.15, 1: 0.85}, dual=False,
#           fit_intercept=True, intercept_scaling=1, max_iter=100,
#           multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
#           solver='liblinear', tol=0.0001, verbose=0, warm_start=False)

y_train_predictions = lr.predict(X_train)
y_test_predictions = lr.predict(X_test)

print ((y_test[y_test==1] == y_test_predictions[y_test==1]).sum().astype(float) / y_test[y_test==1].shape[0])
print ("test ACC:",(y_test_predictions == y_test).sum().astype(float) / y_test.shape[0])