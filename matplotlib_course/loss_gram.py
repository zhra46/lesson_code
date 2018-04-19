from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
iris = load_iris()
X,y = iris['data'][:,:2],(iris['target']==1)
lr = LogisticRegression(C=1000000,fit_intercept=False)
lr.fit(X,y)
theta1 = lr.coef_[0,0]
theta2 = lr.coef_[0,1]
print(theta1,theta2)
# h_function = 1/(1+np.exp(-(np.dot(lr.coef_,X[0]))))
def h_function(data,theta1,theta2):
    return 1/(1+np.exp(-(theta1*data[0]+theta2*data[1])))
print(h_function(X[0],theta1,theta2))
def loss_function(datas,theta1,theta2,labels):
    result = 0
    for data,label in zip(datas,labels):
        h_result = h_function(data,theta1,theta2)
        # print(h_result)
        lossfunction = -1*label*np.log(h_result)-(1-label)*np.log(1-h_result)
        result += lossfunction
    return result
print(loss_function(X,theta1-1,theta2,y))

theta1_space = np.linspace(theta1-3,theta1+3,10)
theta2_space = np.linspace(theta2-3,theta2+3,10)
result2_ = np.array([loss_function(X,theta1,i,y) for i in theta2_space])
result1_ = np.array([loss_function(X,i,theta2,y) for i in theta1_space])
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(theta2_space,result2_)
plt.subplot(2,1,2)
plt.plot(theta1_space,result1_)

# plt.show()

theta1_space = np.linspace(theta1-3,theta1+3,500)
theta2_space = np.linspace(theta2-3,theta2+3,500)
theta1_grid,theta2_grid = np.meshgrid(theta1_space,theta2_space)
result_ = np.zeros((500,500))

for i in range(len(theta1_space)):
    for j in range(len(theta2_space)):
        result_[i,j]=loss_function(X,theta1_space[i],theta2_space[j],y)

print(result_)
# plt.contourf(theta1_space,theta2_space,result_,10,alpha=0.6,cmap=plt.cm.hot)
plt.contour3D(theta1_space,theta2_space,result_,10,alpha=0.6,cmap=plt.cm.hot)
plt.show()
