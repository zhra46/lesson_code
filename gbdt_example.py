from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn import datasets
from sklearn.metrics import mean_squared_error
boston = datasets.load_boston()
# 读取数据
X_train,X_test,y_train,y_test = train_test_split(boston['data'],boston['target'],test_size=0.1,random_state=42)
print(len(X_train))
print(len(X_test))
params = {'n_estimators': 1000, 'max_depth': 6, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls','random_state':42}
# 设置初始化参数

# mse_list = []
# for i in range(1,11):
#     params['max_depth'] = i
#     clf = ensemble.GradientBoostingRegressor(**params)
#     clf.fit(X_train, y_train)
#     mse_list.append(mean_squared_error(y_test, clf.predict(X_test)))
# print('mse_list',mse_list)



clf = ensemble.GradientBoostingRegressor(**params)
# 初始化gbdt分类器
clf_tree = DecisionTreeRegressor(random_state=42)
# 初始化回归树分类器
clf_tree.fit(X_train,y_train)

# mse_tree_list = []
# for i in range(1,20):
#
#     clf_tree = DecisionTreeRegressor(random_state=42,max_depth=i)
#     clf_tree.fit(X_train, y_train)
#     mse_tree_list.append(mean_squared_error(y_test, clf_tree.predict(X_test)))
# print('mse_tree_list',mse_tree_list)

clf.fit(X_train, y_train)
mse = mean_squared_error(y_test, clf.predict(X_test))
# gbdt预测结果与真实结果在验证集上的损失
mse_tree = mean_squared_error(y_test,clf_tree.predict(X_test))
print (clf.loss_(y_test,clf.predict(X_test)))
# clf.loss_函数用来根据clf对象定义的损失函数计算两组数据的损失(在此例子中是平方均值损失)
print("MSE: %.4f" % mse)
print("mse_tree: %.4f" % mse_tree)
# print(len(clf.train_score_))
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
# 创建一个长度=迭代次数的全0一维数组
test_tree_score = np.zeros((params['n_estimators'],),dtype=np.float64)
print(len(list(clf.staged_predict(X_test))))
# 将之前的全0数组依次赋值
for i, y_pred in enumerate(clf.staged_predict(X_test)):
	# staged_predict函数将模型每一次迭代的中间模型结果保留下来, 利用该函数可以查看每一次迭代的中间结果对于X_test的预测值
	# enumerate函数将一个序列与序号组合后返回
    test_score[i] = clf.loss_(y_test, y_pred)
    test_tree_score[i] = mse_tree
print(test_score)

plt.figure(figsize=(12, 6))
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1,test_tree_score,'g-',label = 'Tree Test Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')

plt.show()

