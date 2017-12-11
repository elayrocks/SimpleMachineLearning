'''
高斯混合模型的实践（高斯一元分布）。

对于由参数未知的K个高斯混合模型生成的数据集，利用EM算法可以对这K个高斯分布进行参数估计，并且可以知道两个模型的各自比重。因此还可以用来聚类。
'''

import numpy as np

import matplotlib.pyplot as plt

def Normal(x,mu,sigma):#一元正态分布概率密度函数
    
    return np.exp(-(x-mu)**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)

'''
下面给出K=2，即由两个高斯分布组成的混合模型，分别是男女生身高分布。

已经给出了各自分布的比重、参数。用来检验算法生成的参数估计是否准确。
'''

N_boys=77230#比重77.23%

N_girls=22770#比重22.77%

N=N_boys+N_girls#观测集大小

K=2#高斯分布模型的数量

np.random.seed(1)

#男生身高数据

mu1=1.74#均值

sig1=0.0865#标准差

BoyHeights=np.random.normal(mu1,sig1,N_boys)#返回随机数

BoyHeights.shape=N_boys,1

#女生身高数据

mu2=1.63

sig2=0.0642

GirlHeights=np.random.normal(mu2,sig2,N_girls)#返回随机数

GirlHeights.shape=N_girls,1

data=np.concatenate((BoyHeights,GirlHeights))#合并身高数据，N行1列

#随机初始化模型参数

Mu=np.random.random((1,2))#平均值向量

#Mu[0][0]#Mu[0][1]

SigmaSquare=np.random.random((1,2))#模型迭代用Sigma平方

#SigmaSquare[0][0]#SigmaSquare[0][1]

#随机初始化各模型比重系数（大于等于0，且和为1）

a=np.random.random()

b=1-a

Alpha=np.array([[a,b]])

#Alpha[0][0]#Alpha[0][1]

i=0#迭代次数

while(True):#用EM算法迭代求参数估计
    
    i+=1
    
    #Expectation
    
    gauss1=Normal(data,Mu[0][0],np.sqrt(SigmaSquare[0][0]))#第一个模型
    
    gauss2=Normal(data,Mu[0][1],np.sqrt(SigmaSquare[0][1]))#第二个模型
    
    Gamma1=Alpha[0][0]*gauss1

    Gamma2=Alpha[0][1]*gauss2

    M=Gamma1+Gamma2

    #Gamma=np.concatenate((Gamma1/m,Gamma2/m),axis=1) 元素(j,k)为第j个样本来自第k个模型的概率，聚类时用来判别样本分类

    #Maximization
    
    #更新SigmaSquare
    
    SigmaSquare[0][0]=np.dot((Gamma1/M).T,(data-Mu[0][0])**2)/np.sum(Gamma1/M)
    
    SigmaSquare[0][1]=np.dot((Gamma2/M).T,(data-Mu[0][1])**2)/np.sum(Gamma2/M)

    #更新Mu       

    Mu[0][0]=np.dot((Gamma1/M).T,data)/np.sum(Gamma1/M)

    Mu[0][1]=np.dot((Gamma2/M).T,data)/np.sum(Gamma2/M)

    #更新Alpha

    Alpha[0][0]=np.sum(Gamma1/M)/N
    
    Alpha[0][1]=np.sum(Gamma2/M)/N
    
    if(i%1000==0):
        print("第",i,"次迭代:")
        print("Mu:",Mu)
        print("Sigma:",np.sqrt(SigmaSquare))
        print("Alpha",Alpha)

    #当参数估计不再有显著变化时，退出即可，代码略
    

    

