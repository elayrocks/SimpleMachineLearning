from sympy import *
import numpy as np

epsilon=1e-6#精度

x=Symbol("x")
y=Symbol("y")
z=Symbol("z")
k=Symbol('k')#步长

z=x**4-8*x*y+2*y**2-3#目标函数

w0=np.array([-17.3,-20.5])#初始点

i=0#迭代次数

while (True):
    gx=diff(z,x).subs([(x,w0[0]),(y,w0[1])])#对x求偏导
    gy=diff(z,y).subs([(x,w0[0]),(y,w0[1])])#对y求偏导
    p=-np.array([gx,gy])#负梯度方向
    if((np.sum(np.square(p)))**0.5<epsilon):#负梯度的模长达到精度要求时退出
        break
    
    #用求导的方式，得到步长k
    w1=p*k+w0
    f=z.subs([(x,w1[0]),(y,w1[1])])#用w1替换原目标函数z中的x,y。得到关于步长k的一元函数f(k),即求f(k)的最小值
    a=solve(diff(f,k),k)#对f(k)求导,并使其等于0，求出k
    w1=p*a[0]+w0
    
    #用固定步长k=0.03。比用求导得到的步长收敛慢很多，而且很容易发生振荡，无法收敛。不推荐。
    #w1=p*0.03+w0
    
    print(w1[0].evalf(),w1[1].evalf(),z.subs([(x,w1[0]),(y,w1[1])]).evalf())#观察迭代
    
    if((np.sum(np.square(w1-w0)))**0.5<epsilon):#经过迭代后无明显变化时退出
        break
    
    w0=w1#继续迭代
    i=i+1
    
print(i,"次迭代完成")
print(round(w1[0].evalf()),round(w1[1].evalf()),round(z.subs([(x,w1[0]),(y,w1[1])]).evalf()))
