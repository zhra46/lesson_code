import xgboost as xgb
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
xg_sk = xgb.XGBClassifier()