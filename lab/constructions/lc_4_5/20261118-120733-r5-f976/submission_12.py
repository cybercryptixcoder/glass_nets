import numpy as np
DESCRIPTION="Residual=x_token+x_pos (5 positions, width 4). Two softmax attention layers, no mask or bias: score=xAxT/2 with A=WQ WK^T (A0 layer 0, A1 layer 1), softmax over the 5 keys, then x+=(attn x)V with V=WV WO. Layer 0 is nearly one-hot, layer 1 soft; only position 4 is read: out=(x4+a1.(x V1))WU, and class 4 never wins so its score is a constant. Every matrix is the network's own, as integers over 1000; an attention row stores 4 weights, the 5th=1-sum."
t=np.array("-555 -2026 1424 301 1022 -1087 678 1846 999 -200 1144 1396 1769 1896 1094 -522 -640 -17 2882 -74 933 -525 2255 26 1908 -1154 594 -1673 -1298 1957 2546 -474 -1592 -453 -67 2061 -1571 -1151 170 2608 -16742 3424 -21421 -15363 -20575 10823 -20548 8575 851 5740 13216 10749 11627 -7776 5781 -4136 1691 938 -494 453 523 583 -140 760 -601 1063 -147 -838 459 1326 -258 1043 -291 -549 465 1231 -166 -341 288 505 -1556 -5278 6376 9313 -259 -770 1061 1025 -533 11540 -3010 -161 2118 -1621 -470 1404 -168 16038 -1941 -3937 -535 -7531 3190 -2708 140 -102 -27 -30 -44 114 112 -139 689 -418 198 -527 -315 -363 607 361".split(),float)/1000
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
    return np.c_[iv('r',Z[:,4]+q)@U, np.full((b,1),-6.)]
ALIGN={'e':['embed'],'A':['attn0.pattern'],'o':['attn0.out'],'Z':['resid0.mid'],'a':['attn1.pattern'],'q':['attn1.out'],'r':['resid1.mid']}
