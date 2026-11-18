import numpy as np
DESCRIPTION="Two exact-softmax heads on R=W_E[tok]+W_pos: score R.M.R^T/2, softmax over the 5 keys, add attended values; layer 1 same; logits = last residual times W_U. Matrices (weights*150) in a base-81 string."
s="JIG_N.L'MBIJL`NmM?K-MUN)NaO#MMJNJ<KIPfK@M5JMOYKOO%I@LSHCI*O-P4JUHOJXKAO<HRI@KdP=,KQg#f/'%C_N%G[AM)V-cq_Ca%=+V3CgNVM6JRL>LHLQK6LlJBMHK5IoL?MpK$MEJpJJL@MbK2JiL%LFHUA]W;\\_K$J)MHMCJL`iEnK3OEHKJVN+K2i3GlD3JL=OQDFJK`K<KGKGKDK\\K\\K6LaJ]KiJMJmJfLUL0"
T=np.frombuffer(s.encode(),'uint8').astype(int)-35
M=(T[0::2]*81+T[1::2]-3280).reshape(30,4)
E,P,Q,V,W,G,U=np.split(M,[5,10,14,18,22,26])
k=150
def f(S):e=np.exp(S-S.max(-1,keepdims=True));return e/e.sum(-1,keepdims=True)
def model(x,iv):
 B=len(x)
 def h(n,v):return iv(n,v.reshape(B,-1)).reshape(v.shape)
 r=h('R',(E[x]+P)/k)
 z=h('Z',r+f(r@Q@r.transpose(0,2,1)/2/k)@r@V/k)
 a=h('a',f(z@W@z.transpose(0,2,1)/2/k))
 o=h('o',a@z@G/k)
 t=iv('t',(z+o)[:,4])
 return np.c_[t@U/k,t[:,:1]*0]
ALIGN={'R':['embed'],'Z':['resid0.mid'],'a':['attn1.pattern'],'o':['attn1.out'],'t':['resid1.mid']}
