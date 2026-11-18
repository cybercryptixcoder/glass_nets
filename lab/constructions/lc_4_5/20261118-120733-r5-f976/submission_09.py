import numpy as np
DESCRIPTION="Residual=x_token+x_pos (5 positions, width 4). Two softmax attention layers, no mask or bias: score=xAxT/2 with A=WQ WK^T (A0 of layer 0, A1 of layer 1), softmax over the 5 keys, then x+=(attn x)V with V=WV WO. Layer 0 is nearly one-hot, layer 1 soft; only position 4 is read: out=(x4+a1.(x V1))WU (class 4 never wins, its score is a constant). Every matrix is the network's own, as integers over 1000; an attention row stores 4 weights, the 5th=1-sum."
t=np.array("NUMBERS".split(),float)/1000
E=t[0:20].reshape(5, 4)
P=t[20:40].reshape(5, 4)
M0=t[40:56].reshape(4, 4)
V0=t[56:72].reshape(4, 4)
M1=t[72:88].reshape(4, 4)
V1=t[88:104].reshape(4, 4)
U=t[104:120].reshape(4, 4)
def sm(s):s=np.exp(s-s.max(-1,keepdims=True));return s/s.sum(-1,keepdims=True)
def rw(a):return np.concatenate([a,1-a.sum(-1,keepdims=True)],-1)
def model(x,iv):
    b=len(x)
    g=lambda n,v:iv(n,v.reshape(b,20)).reshape(b,5,4)
    e=g('e',E[x]+P)
    A=g('A',sm(e@M0@e.transpose(0,2,1)/2)[...,:4])
    o=g('o',rw(A)@e@V0)
    Z=g('Z',e+o)
    a=iv('a',sm((Z@M1)@Z.transpose(0,2,1)/2)[:,4,:4])
    q=iv('q',(rw(a)[:,None]@Z)[:,0]@V1)
    return np.c_[iv('r',Z[:,4]+q)@U, np.full((b,2),-99.)]
ALIGN={'e':['embed'],'A':['attn0.pattern'],'o':['attn0.out'],'Z':['resid0.mid'],'a':['attn1.pattern'],'q':['attn1.out'],'r':['resid1.mid']}
