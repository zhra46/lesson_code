from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

mnist = fetch_mldata('MNIST original', data_home='test_data_home')
print(mnist)

X, y = mnist['data'], mnist['target']
print(X.shape, y.shape)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
rf_clf = RandomForestClassifier()
tree_clf = DecisionTreeClassifier()
logi_clf = LogisticRegression(max_iter=100,solver='sag')
# tree_clf.fit(X_train,y_train)
# rf_clf.fit(X_train,y_train)
# y_hat_rf = rf_clf.predict(X_test)
# logi_clf.fit(X_train[:2000],y_train[:2000])
# print('fitted')
# y_hat_logi = logi_clf.predict(X_test[:2000])
gbdt_clf = GradientBoostingClassifier()
gbdt_clf.fit(X_train[:2000],y_train[:2000])
y_hat_gbdt = gbdt_clf.predict(X_test[:2000])
# y_hat_tree = tree_clf.predict(X_test)

# print(accuracy_score(y_test,y_hat_rf))
# print(accuracy_score(y_test,y_hat_tree))
# print(accuracy_score(y_test[:2000],y_hat_logi))
# print(accuracy_score(y_test[:2000],y_hat_gbdt))
# print(f1_score(y_test[:2000],y_hat_gbdt))
# print(precision_score(y_test[:2000],y_hat_gbdt))
# print(recall_score(y_test[:2000],y_hat_gbdt))

