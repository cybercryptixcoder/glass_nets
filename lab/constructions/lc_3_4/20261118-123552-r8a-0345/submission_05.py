import numpy as np
DESCRIPTION="2 attention layers,width3;pos3 gauge0;softmax(hMh^T) then +(a@h)@VO;last pos read by WU."
v=np.array("0.205 0.453 0.153 2.828 2.243 -0.029 2.114 2.358 0.323 3.102 1.586 1.264 -0.931 -2.115 -0.550 -1.865 -1.547 3.097 -5.973 -0.123 1.308 0.187 -0.394 0.098 -0.448 -1.263 1.446 -2.828 -1.654 2.650 0.030 -0.774 0.130 1.064 -0.102 0.246 0.746 1.114 -0.107 0.258 2.938 0.266 -0.195 -2.235 -0.212 -0.278 1.273 3.810 -1.879 1.571 -1.812 0.045 0.442 -0.405 -0.169 -9.847 5.846 37.446 35.604 36.886 6.303 3.634 4.531 1.064 -4.666 -2.586".split(),float)
E=v[:12].reshape(4,3);P=v[12:21].reshape(3,3);M0=v[21:30].reshape(3,3);V0=v[30:39].reshape(3,3);M1=v[39:48].reshape(3,3);V1=v[48:57].reshape(3,3);U=v[57:66].reshape(3,3)
P=np.vstack([P,np.zeros(3)]);U=np.insert(U,1,0,1)
def model(x,iv):
 B=len(x);h=iv('e',(E[x]+P).reshape(B,-1)).reshape(B,4,3)
 for l,(M,W) in enumerate([(M0,V0),(M1,V1)]):
  s=h@M@h.transpose(0,2,1);s=s-s.max(-1,keepdims=True);a=np.exp(s);a/=a.sum(-1,keepdims=True)
  a=iv('p%d'%l,a.reshape(B,-1)).reshape(B,4,4)
  o=iv('o%d'%l,np.einsum('bij,bjk,kl->bil',a,h,W).reshape(B,-1)).reshape(B,4,3)
  h=iv('r%d'%l,(h+o).reshape(B,-1)).reshape(B,4,3)
 return h[:,3]@U
ALIGN={'e':['embed'],'p0':['attn0.pattern'],'o0':['attn0.out'],'r0':['resid0.mid'],'p1':['attn1.pattern'],'o1':['attn1.out'],'r1':['resid1.mid']}
