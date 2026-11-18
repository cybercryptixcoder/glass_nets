import numpy as np
D="2 attention layers. R=E[tok]/s0+P/s1. A=softmax(R M R^T/s2 over keys); Z=R+A R V/s3; A1=softmax((Z N/s4)Z^T over keys); out=(Z+A1 Z W/s5)[4] U/s6. Integer tables."
E=np.array([[1,-2,1,1],[1,-1,1,1],[1,1,1,1],[1,1,1,-1],[-1,1,2,-1]])
P=np.array([[1,-1,2,1],[1,-1,1,-1],[-1,1,2,-1],[-1,-1,1,2],[-1,-1,1,2]])
M=np.array([[-1,1,-1,-1],[-1,1,-1,1],[1,1,1,1],[1,-1,1,-1]])
V=np.array([[1,1,-1,1],[1,1,-1,1],[-1,1,-1,-1],[1,1,-1,1]])
N=np.array([[-1,-1,1,1],[-1,-1,1,1],[-1,-1,1,1],[-1,-1,1,1]])
W=np.array([[-1,1,-1,-1],[1,-1,-1,1],[-1,1,-1,-1],[-1,-1,1,-1]])
U=np.array([[1,-1,-1,-1,-1],[-1,1,1,-1,-1],[1,-1,1,-1,-1],[-1,-1,1,1,1]])
s=[100,100,100,100,60,20,30]
def f(a):
 a=np.exp(a-a.max(-1,keepdims=True));return a/a.sum(-1,keepdims=True)
def model(x,iv):
 B=len(x)
 R=iv('e',(E[x]/s[0]+P/s[1]).reshape(B,-1)).reshape(B,5,4)
 A=f(R@M@R.transpose(0,2,1)/s[2])
 A0=iv('a0',A.reshape(B,-1)).reshape(B,5,5)
 Z=iv('z',(R+A0@R@V/s[3]).reshape(B,-1)).reshape(B,5,4)
 A=f((Z@N/s[4])@Z.transpose(0,2,1))
 A1=iv('a1',A.reshape(B,-1)).reshape(B,5,5)
 r=iv('r1',(Z+A1@(Z@(W/s[5]))).reshape(B,-1)).reshape(B,5,4)
 return r[:,4]@U/s[6]
ALIGN={'e':['embed'],'a0':['attn0.pattern'],'z':['resid0.mid'],'a1':['attn1.pattern'],'r1':['resid1.mid']}
