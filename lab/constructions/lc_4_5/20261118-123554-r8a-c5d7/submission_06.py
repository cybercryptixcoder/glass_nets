import numpy as np
D="Two attention layers: R=E[tok]/s0+P/s1; A0=softmax(R M R^T/s2 over keys); Z=R+A0 R V/s3; A1=softmax((Z N/s4)Z^T over keys); out=(Z+A1 Z W/s5)[4] U/s6. Integer tables x1/s."
E=np.array([[-83,-304,214,45],[153,-163,102,277],[150,-30,172,209],[265,284,164,-78],[-96,-2,432,-11]])
P=np.array([[140,-79,338,4],[286,-173,89,-251],[-195,294,382,-71],[-239,-68,-10,309],[-236,-173,25,391]])
M=np.array([[-1256,257,-1607,-1152],[-1543,812,-1541,643],[64,431,991,806],[872,-583,434,-310]])
V=np.array([[85,47,-25,23],[26,29,-7,38],[-30,53,-7,-42],[23,66,-13,52]])
N=np.array([[-15,-27,23,62],[-8,-17,14,25],[-78,-264,319,466],[-13,-38,53,51]])
W=np.array([[-11,231,-60,-3],[42,-32,-9,28],[-3,321,-39,-79],[-11,-151,64,-54]])
U=np.array([[4,-3,-1,-1,-4],[-1,3,3,-3,-10],[17,-10,5,-13,-37],[-8,-9,15,9,0]])
s=[150,150,150,50,100,20,25]
def f(a):
 a=np.exp(a-a.max(-1,keepdims=True));return a/a.sum(-1,keepdims=True)
def model(x,iv):
 B=len(x)
 R=E[x]/s[0]+P/s[1]
 A=f(R@M@R.transpose(0,2,1)/s[2])
 A=iv('a0',A.reshape(B,-1)).reshape(B,5,5)
 Z=iv('z',(R+A@R@V/s[3]).reshape(B,-1)).reshape(B,5,4)
 A=f((Z@N/s[4])@Z.transpose(0,2,1))
 return (Z+A@(Z@(W/s[5])))[:,4]@U/s[6]
ALIGN={'a0':['attn0.pattern'],'z':['resid0.mid']}
