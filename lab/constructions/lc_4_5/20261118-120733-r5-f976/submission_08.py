import numpy as np
DESCRIPTION="Residual=x_token+x_pos. Two softmax attention layers: score=xAxT, A=WQ WK^T (already halved), x+=(attn x)V; layer 0 nearly one-hot, layer 1 soft; only position 4 is read, out=(x4+a1.(xV1))WU. Matrices are the network's own integers over the stated scale; an attention row stores 4 weights, the 5th=1-sum."
t=np.array("-166 -608 427 90 307 -326 204 554 300 -60 343 419 531 569 328 -156 -192 -5 864 -22 280 -158 676 8 572 -346 178 -502 -389 587 764 -142 -478 -136 -20 618 -471 -345 51 782 -1339 274 -1714 -1229 -1646 866 -1644 686 68 459 1057 860 930 -622 462 -331 846 469 -247 226 262 292 -70 380 -300 531 -74 -419 230 663 -129 522 -29 -55 47 123 -17 -34 29 51 -156 -528 638 931 -26 -77 106 102 -13 288 -75 -4 53 -41 -12 35 -4 401 -49 -98 -13 -188 80 -68 21 -15 -4 -4 -24 -7 17 17 -21 -59 103 -63 30 -79 -220 -47 -54 91 54 1".split(),float)
E=t[0:20].reshape(5, 4)/300
P=t[20:40].reshape(5, 4)/300
M0=t[40:56].reshape(4, 4)/160
V0=t[56:72].reshape(4, 4)/500
M1=t[72:88].reshape(4, 4)/200
V1=t[88:104].reshape(4, 4)/25
U=t[104:124].reshape(4, 5)/150
def sm(s):s=np.exp(s-s.max(-1,keepdims=True));return s/s.sum(-1,keepdims=True)
def rw(a):return np.concatenate([a,1-a.sum(-1,keepdims=True)],-1)
def model(x,iv):
    b=len(x)
    g=lambda n,v:iv(n,v.reshape(b,20)).reshape(b,5,4)
    e=g('e',E[x]+P)
    A=g('A',sm(e@M0@e.transpose(0,2,1))[...,:4])
    o=g('o',rw(A)@e@V0)
    Z=g('Z',e+o)
    a=iv('a',sm(Z@M1@Z.transpose(0,2,1))[:,4,:4])
    q=iv('q',(rw(a)[:,None]@Z)[:,0]@V1)
    return iv('r',Z[:,4]+q)@U
ALIGN={'e':['embed'],'A':['attn0.pattern'],'o':['attn0.out'],'Z':['resid0.mid'],'a':['attn1.pattern'],'q':['attn1.out'],'r':['resid1.mid']}
