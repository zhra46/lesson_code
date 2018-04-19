import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0,10,100)
# plt.plot(x,np.sin(x))
# plt.plot(x,np.cos(x))
# plt.show()
#plot.show在背后完成了很多事情， 与不同操作系统的图形显示接口进行交互，虽然具体操作细节非常复杂，但是matplotlib隐藏了所有的细节，很省心


# fig = plt.figure()
# plt.plot(x,np.sin(x))
# plt.plot(x,np.cos(x))
# fig.savefig('1.png')
# 支持多种图片格式 如 jpg jpeg pdfraw tif等

# 创建子图
# plt.figure()
# plt.subplot(2,2,1)#总共绘制2行2列的图，绘制在第一号位置
# plt.plot(x,np.sin(x))
# plt.plot(x,np.cos(x))
# plt.subplot(2,2,2)#遇到这行代码 matplotlib会开始在这个图上进行绘制
# plt.plot(x,np.cos(x))
# plt.show()
# 这种接口最重要的特性是有状态的，会持续跟踪当前的图形和坐标轴，所有plt命令都会在当前图形上进行操作

# #面向对象接口,相比之前有状态的方式，显式指明要在那个子图上绘制
# fig,ax = plt.subplots(2)
# ax[0].plot(x,np.sin(x))
# ax[1].plot(x,np.cos(x))

# figure对象是一个容纳所有图像文字坐标轴的容器
# plt.style.use('seaborn-whitegrid')
# fig = plt.figure()
# ax = plt.axes()
# ax.plot(x,np.sin(x))
# plt.show()

# #调整坐标轴上下界
# # plt.plot(x,np.sin(x))
# # plt.xlim(-1,11)
# # plt.ylim(-1.5,1.5)
# # plt.show()
# #可以通过逆序传入参数使坐标轴逆序绘制
# #可以通过axis 注意不是axes绘制图像
# plt.plot(x,np.sin(x))
# # plt.axis([5,11,-1.5,1.5])
# plt.axis('tight')
# plt.axis('equal')
# plt.show()

#设置图形标签
# fig =plt.figure()
# plt.plot(x,np.sin(x))
# plt.title("a sin curve")
# plt.xlabel('x axis')
# plt.ylabel('sin(x)')
# # plt.plot(x,np.cos(x),':b',label='cos(x)')
# plt.plot(x,np.cos(x),'o',label='cos(x)')
# #scatter命令在创造散点时可以和每个数据匹配，让每个散点具有不同的属性（大小，表面颜色，边框颜色等）
# plt.scatter(x,np.cos(x)+1,marker='o')
# plt.legend()
# plt.show()


# # 每个散点不同的改变大小
# rng = np.random.RandomState(0)
# x = rng.randn(100)
# y = rng.randn(100)
# colors = rng.randn(100)
# sizes = 1000*rng.rand(10)
# plt.scatter(x,y,c=colors,s=sizes,alpha=0.3)
# plt.colorbar()
# plt.show()
# # 效率对比： 当大数据量时 plot的效率要远高于scatter

# 密度图与等高线图
def f(x,y):
    return np.sin(x)**10 + np.cos(10+y*x)*np.cos(x)
x = np.linspace(0,5,50)
y = np.linspace(0,5,40)
X,Y = np.meshgrid(x,y)
Z=f(X,Y)
plt.contour(X,Y,Z,colors='black')
plt.contourf(X,Y,Z,cmap='RdGy')
plt.imshow(Z,extent=[0,5,0,5],origin='lower',cmap='RdGy')
plt.show()