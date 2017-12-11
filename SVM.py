'''
支持向量机，代码参考了《机器学习实战》

即寻找超平面：w*x+b=0，使得二分类有最大间隔

利用SMO算法，快速求解对偶问题，得到最优解w，b

作者：胡亦磊
'''

'''
在ε精度范围内违反KKT条件的点：

1、alpha[i]=0   ⇔ yiEi<-ε

2、0<alpha[i]<c ⇔ |yiEi|>ε

3、alpha[i]=c   ⇔ yiEi>ε

alpha[i]=0或c时，称为边界点；0<alpha[i]<c，称为非边界点（nonbound）
'''

import numpy as np

import scipy.io as scio

from matplotlib import pyplot as plt

def loadDataSet():#载入数据集
    
    dataFile='nonlinearData.mat'#读取mat数据集
    
    data=scio.loadmat(dataFile)
    
    dataset=data['nonlinear'].T#N*3的格式，N为样本个数，平面坐标X1,X2，标签Y
    
    dataset=dataset[2300:2500,:]#选200个样本进行训练
    
    positive=np.array([[0,0,0]])#+1的样本集
    
    negative=np.array([[0,0,0]])#-1的样本集
    
    for i in range(dataset.shape[0]):
        
        if(dataset[i][2]==1):
            
            positive=np.row_stack((positive,np.array([dataset[i]])))
            
        else:
            
            negative=np.row_stack((negative,np.array([dataset[i]])))

    return positive[1:,:],negative[1:,:],dataset

'''
def kernel(xi,xj):#核函数（xi，xj为向量）

    return xi.dot(xj.T)#内积(数据集线性可分或近似线性可分）
'''

sigma=10.0

def kernel(xi,xj):#高斯核函数（数据集线性不可分）

    M=xi.shape[0]

    K=np.zeros((M,1))

    for l in range(M):

        A=np.array([xi[l]])-xj

        K[l]=[np.exp(-0.5*float(A.dot(A.T))/(sigma**2))]

    return K
            
def findNonBound(alpha,C):#寻找非边界点
    
    nonbound=[]
    
    for i in range(len(alpha)):
        
        if(0<alpha[i] and alpha[i]<C):
            
            nonbound.append(i)
            
    return nonbound

def selectJrand(i,N):#随机选择j

    j=i

    while(j==i):

        j=int(np.random.uniform(0,N))#左闭右开，类型转化后0<=j<=N-1

    return j
    
class SVM(object):

    def __init__(self,X,Y,C,epsilon):
        
        self.X=X#数据集N*D（D维特征）
        
        self.Y=Y#标签（1，-1）

        self.N=X.shape[0]#数据集大小
        
        self.C=C#惩罚系数

        self.epsilon=epsilon#精度

        self.alpha=np.zeros((self.N,1))#拉格朗日乘子N*1

        self.b=0#位移

        self.E=np.zeros((self.N,2))#误差缓存表N*2，第一列为更新状态（0-未更新，1-已更新），第二列为缓存值

    def computeEk(self,k):#计算缓存项Ek

        xk=np.array([self.X[k]])

        y=np.array([self.Y]).T

        gxk=float(self.alpha.T.dot(y*kernel(self.X,xk)))+self.b
        
        Ek=gxk-self.Y[k]
        
        return Ek

    def updateEk(self,k):#更新缓存项Ek包括计算Ek和设置对应的更新状态为1

        Ek=self.computeEk(k)

        self.E[k]=[1,Ek]

    def selectJ(self,i,Ei):#内循环，根据i选择j

        self.E[i]=[1,Ei]#更新Ei

        validE=np.nonzero(self.E[:,0])[0]#validE保存更新状态为1的缓存项的行指标
        
        if(len(validE)>1):

            j=0

            maxDelta=0

            Ej=0
            
            for k in validE:#寻找最大的|Ei-Ej|

                if(k==i):   continue
                
                Ek=self.computeEk(k)

                if(abs(Ei-Ek)>maxDelta):

                    j=k

                    maxDelta=abs(Ei-Ek)

                    Ej=Ek

        else:#随机选择

            j=selectJrand(i,self.N)

            Ej=self.computeEk(j)
                
        return j,Ej
        
    def inner(self,i):

        Ei=self.computeEk(i)
        
        if((self.Y[i]*Ei>self.epsilon and float(self.alpha[i])>0) or\
           (self.Y[i]*Ei<-self.epsilon and float(self.alpha[i])<self.C)):#alpha[i]违反了KKT条件

            j,Ej=self.selectJ(i,Ei)#选择对应的alpha[j]

            alphaiold=float(self.alpha[i])
            
            alphajold=float(self.alpha[j])
            
            if(self.Y[i]!=self.Y[j]):

                L=max(0,alphajold-alphaiold)
                
                H=min(self.C,self.C+alphajold-alphaiold)

            else:
                
                L=max(0,alphajold+alphaiold-self.C)
                
                H=min(self.C,alphajold+alphaiold)

            if(L==H): return 0

            xi=np.array([self.X[i]])

            xj=np.array([self.X[j]])

            eta=float(kernel(xi,xi)+kernel(xj,xj)-2*kernel(xi,xj))

            if(eta<=0): return 0

            alphajnewunc=alphajold+self.Y[j]*(Ei-Ej)/eta#未剪辑的alphajnew

            #更新alphaj
            
            if(alphajnewunc>H): self.alpha[j]=[H]

            elif(alphajnewunc<L): self.alpha[j]=[L]

            else: self.alpha[j]=[alphajnewunc]

            #更新Ej
            
            self.updateEk(j)

            if(abs(float(self.alpha[j])-alphajold)<0.00001): return 0
            
            #更新alphai
            
            self.alpha[i]=[alphaiold+Y[i]*Y[j]*(alphajold-float(self.alpha[j]))]

            #更新Ei

            self.updateEk(i)

            #更新b

            bi=-Ei-self.Y[i]*float(kernel(xi,xi))*(float(self.alpha[i])-alphaiold)-\
                self.Y[j]*float(kernel(xj,xi))*(float(self.alpha[j])-alphajold)+self.b
            
            bj=-Ej-self.Y[i]*float(kernel(xi,xj))*(float(self.alpha[i])-alphaiold)-\
                self.Y[j]*float(kernel(xj,xj))*(float(self.alpha[j])-alphajold)+self.b
            
            if(0<float(self.alpha[i]) and float(self.alpha[i])<self.C): self.b=bi
            
            elif(0<float(self.alpha[j]) and float(self.alpha[j])<self.C): self.b=bj
            
            else: self.b=0.5*(bi+bj)

            return 1

        else: return 0

    def visualize(self,positive,negative):

        plt.xlabel('X1')#横坐标
    
        plt.ylabel('X2')#纵坐标
    
        plt.scatter(positive[:,0],positive[:,1],c = 'r',marker = 'o')#+1样本红色标出
    
        plt.scatter(negative[:,0],negative[:,1],c = 'g',marker = 'o')#-1样本绿色标出

        nonZeroAlpha=self.alpha[np.nonzero(self.alpha)]#非零的alpha
        
        supportVector=X[np.nonzero(self.alpha)[0]]#支持向量
        
        y=np.array([self.Y]).T[np.nonzero(self.alpha)]#支持向量对应的标签
        
        plt.scatter(supportVector[:,0],supportVector[:,1],s=100,c='y',alpha=0.5,marker='o')#标出支持向量

        print("支持向量个数:",len(nonZeroAlpha))

        X1=np.arange(-50.0,50.0,0.1)
    
        X2=np.arange(-50.0,50.0,0.1)
 
        x1,x2=np.meshgrid(X1,X2)

        g=self.b
        
        for i in range(len(nonZeroAlpha)):

            #g+=nonZeroAlpha[i]*y[i]*(x1*supportVector[i][0]+x2*supportVector[i][1])

            g+=nonZeroAlpha[i]*y[i]*np.exp(-0.5*((x1-supportVector[i][0])**2+(x2-supportVector[i][1])**2)/(sigma**2))

        plt.contour(x1,x2,g,0,colors='b')#画出超平面

        plt.title("sigma: %f" %sigma)
        
        plt.show()

def SMO(X,Y,C,epsilon,maxIters):#SMO的主程序

    SVMClassifier=SVM(X,Y,C,epsilon)
    
    iters=0
    
    iterEntire=True#由于alpha被初始化为零向量，所以先遍历整个样本集
    
    while(iters<maxIters):#循环在整个样本集与非边界点集上切换，达到最大循环次数时退出

        iters+=1
        
        if(iterEntire):#循环遍历整个样本集

            alphaPairChanges=0
            
            for i in range(SVMClassifier.N):#外层循环

                alphaPairChanges+=SVMClassifier.inner(i)
                    
            if(alphaPairChanges==0):    break#整个样本集上无alpha对变化时退出循环
            
            else:   iterEntire=False#有alpha对变化时遍历非边界点集
            
        else:#循环遍历非边界点集
            
            alphaPairChanges=0
            
            nonbound=findNonBound(SVMClassifier.alpha,SVMClassifier.C)#非边界点集
            
            for i in nonbound:#外层循环

                alphaPairChanges+=SVMClassifier.inner(i)
                    
            if(alphaPairChanges==0):

                iterEntire=True#非边界点全满足KKT条件，则循环遍历整个样本集
            
    return SVMClassifier

if __name__ == "__main__":
    
    positive,negative,dataset=loadDataSet()#返回+1与-1的样本集，总训练集
    
    X=dataset[:,0:2]#X1,X2
    
    Y=dataset[:,2]#Y
    
    SVMClassifier=SMO(X,Y,1,0.001,40)

    SVMClassifier.visualize(positive,negative)

    
