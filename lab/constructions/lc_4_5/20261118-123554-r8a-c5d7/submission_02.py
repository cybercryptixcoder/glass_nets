import numpy as np
D="2 attention layers. R=E[tok]/s0+P/s1. A=softmax(R M R^T/s2 over keys); Z=R+A R V/s3; A1=softmax((Z N/s4)Z^T over keys); out=(Z+A1 Z W/s5)[4] U/s6. Integer tables."
E=np.array([[-55,-203,142,30],[102,-109,68,185],[100,-20,114,140],[177,190,109,-52],[-64,-2,288,-7]])
P=np.array([[93,-53,225,3],[191,-115,59,-167],[-130,196,255,-47],[-159,-45,-7,206],[-157,-115,17,261]])
M=np.array([[-837,171,-1071,-768],[-1029,541,-1027,429],[43,287,661,537],[581,-389,289,-207]])
V=np.array([[169,94,-49,45],[52,58,-14,76],[-60,106,-15,-84],[46,133,-26,104]])
N=np.array([[-9,-16,14,37],[-5,-10,9,15],[-47,-158,191,279],[-8,-23,32,31]])
W=np.array([[-27,577,-151,-8],[106,-81,-24,70],[-8,802,-97,-197],[-27,-377,160,-135]])
U=np.array([[5,-3,-1,-1,-5],[-1,3,3,-4,-12],[21,-13,6,-16,-44],[-9,-11,18,11,0]])
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
