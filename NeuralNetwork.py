'''
非线性可分数据集上的神经网络实践
隐层使用tanh函数，输出层用softmax函数处理，用交叉熵损失函数
作者：胡亦磊
'''

import numpy as np
import sklearn
from sklearn import datasets
from matplotlib import pyplot as plt

class Config:

    eta=0.01#学习率

    reg_lambda=0.1#正则化常数
    
    input_dim=2#输入层神经元个数，即样本的特征数，这里是横纵坐标
    
    hidden_dim=10#隐层神经元个数
    
    output_dim=3#输出层神经元个数
    
def generate_data():
    
    np.random.seed(0)
    
    N=100#number of points per class
    
    D=2#dimensionality
    
    K=3#number of classes
    
    X=np.zeros((N*K,D))
    
    Y=np.zeros(N*K,dtype='uint8')
    
    for j in range(K):
        
      ix=range(N*j,N*(j+1))
      
      r=np.linspace(0.0,1,N)
      
      t=np.linspace(j*4,(j+1)*4,N)+np.random.randn(N)*0.2
      
      X[ix]=np.c_[r*np.sin(t),r*np.cos(t)]
      
      Y[ix]=j
      
    fig=plt.figure()
    
    plt.scatter(X[:,0],X[:,1],c=Y,s=40,cmap=plt.cm.Spectral)
    
    plt.xlim([-1,1])
    
    plt.ylim([-1,1])
    
    return X,Y

def softmax(z):#输出层softmax函数，输出为yhat(概率)。为了数值稳定防止溢出，进行平移。

    z-=np.max(z)
    
    return np.exp(z)/np.sum(np.exp(z))
        
def predict(x,model):
    '''
    得到预测输出
    '''
    p=np.dot(x,model['v'])+model['b']#隐层输入

    a=np.tanh(p)#隐层输出

    z=np.dot(a,model['w'])+model['c']#输出层输入

    yhat=softmax(z)#输出层输出(概率)
    
    return np.argmax(yhat,axis=1),yhat#选概率大的作为分类
    
def CrossEntropyLoss(y,yhat):#单个样本交叉熵损失函数
    
    return -np.log(yhat[0][y])

def totalLoss(X,Y,model):#训练集总损失
    
    Loss=0.0
    
    precision=0.0
    
    for i in range(len(X)):
        
        ypred,yhat=predict(X[i],model)
        
        if(Y[i]==ypred):
            
            precision+=1

        Loss+=CrossEntropyLoss(Y[i],yhat)

    precision/=len(X)
    
    print("精确度:",precision)

    reg_loss=0.5*Config.reg_lambda*np.sum(np.square(model['w']))+0.5*Config.reg_lambda*np.sum(np.square(model['v']))
    
    Loss=Loss/len(X)+reg_loss
    
    return Loss
            

def build_model(X,Y,iters):
    
    '''
    在[0,1)区间内随机初始化连接权重与阈值
    '''
    v=np.random.random((Config.input_dim,Config.hidden_dim))#输入层到隐层的连接权重
    
    w=np.random.random((Config.hidden_dim,Config.output_dim))#隐层到输出层的连接权重
    
    b=np.random.random((1,Config.hidden_dim))#隐层的阈值

    c=np.random.random((1,Config.output_dim))#输出层的阈值

    model={}
    
    for t in range(iters):#iters次epoch
        
        for i in range(len(X)):#一个训练样本就更新一次参数 
            
            #forward propagation
            
            p=np.array([X[i]]).dot(v)+b#隐层输入

            a=np.tanh(p)#隐层输出

            z=np.dot(a,w)+c#输出层输入

            yhat=softmax(z)#输出层输出(概率)

            #back propagation
            
            g=yhat
            
            g[range(len(yhat)),np.array([Y[i]])]-=1#输出层梯度

            w+=-Config.eta*np.dot(a.T,g)#更新隐层到输出层的连接权重

            c+=-Config.eta*g#更新输出层的阈值

            h=np.dot(g,w.T)*(1-np.power(a,2))#隐层梯度

            v+=-Config.eta*np.array([X[i]]).T.dot(h)#更新输入层到隐层的连接权重

            b+=-Config.eta*h#更新隐层的阈值
        
        w+=-Config.eta*Config.reg_lambda*w
        
        v+=-Config.eta*Config.reg_lambda*v
        
        model={'v': v, 'w': w, 'b': b, 'c': c}

        print("after",t,"iteration:",totalLoss(X,Y,model))
        
    return model    

def plot_decision_boundary(pred_func,X,Y):
    
    #设置最小最大值, 加上一点外边界
    
    x_min,x_max=X[:,0].min()-.5,X[:,0].max()+.5
    
    y_min,y_max=X[:,1].min()-.5,X[:,1].max()+.5

    h=0.01
    
    # 根据最小最大值和一个网格距离生成整个网格
    
    xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
    
    # 对整个网格预测边界值
    
    Z,Yhat=pred_func(np.c_[xx.ravel(),yy.ravel()])
    
    Z=Z.reshape(xx.shape)
    
    # 绘制边界和数据集的点
    
    plt.contourf(xx,yy,Z,cmap=plt.cm.Spectral)
    
    plt.scatter(X[:,0],X[:,1],c=Y,cmap=plt.cm.Spectral)        

def NN():
    
    X,Y=generate_data()#产生数据集
    
    model=build_model(X,Y,2500)#建立神经网络模型,iters为迭代次数

    plot_decision_boundary(lambda x: predict(x,model),X,Y)

    plt.title("Decision Boundary for hidden layer size %d" %Config.hidden_dim)

    plt.show()
        
if __name__ == "__main__":
    
    NN()
