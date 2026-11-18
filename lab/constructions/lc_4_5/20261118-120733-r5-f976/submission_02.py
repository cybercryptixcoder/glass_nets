import numpy as np
DESCRIPTION=("Residual = token + position embedding. Layer 0: scores R[p]M0.R[q]/2 (M0=W_Q0W_K0^T), softmax over keys "
"(nearly one-hot); each position adds the attended residual back through V0=W_V0W_O0, giving Z. Layer 1: scores "
"(Z[p]M1).Z[q]/2 (M1=W_Q1W_K1^T), softmax over keys; only position 4 is read: out=(Z[4]+a1.(Z V1))W_U, V1=W_V1W_O1. "
"Matrices are the network's as integers/1000; an attention row is given by its first 4 weights, 5th=1-sum.")
def T(s,r,c): return np.array([int(v) for v in s.split()]).reshape(r,c)/1000.
E=T("-555 -2026 1424 301 1022 -1087 678 1846 999 -200 1144 1396 1769 1896 1094 -522 -640 -17 2882 -74",5,4)
P=T("933 -525 2255 26 1908 -1154 594 -1673 -1298 1957 2546 -474 -1592 -453 -67 2061 -1571 -1151 170 2608",5,4)
M0=T("-16742 3424 -21421 -15363 -20575 10823 -20548 8575 851 5740 13216 10749 11627 -7776 5781 -4136",4,4)
V0=T("1691 938 -494 453 523 583 -140 760 -601 1063 -147 -838 459 1326 -258 1043",4,4)
M1=T("-291 -549 465 1231 -166 -341 288 505 -1556 -5278 6376 9313 -259 -770 1061 1025",4,4)
V1=T("-533 11540 -3010 -161 2118 -1621 -470 1404 -168 16038 -1941 -3937 -535 -7531 3190 -2708",4,4)
WU=T("140 -102 -27 -30 -158 -44 114 112 -139 -395 689 -418 198 -527 -1465 -315 -363 607 361 6",4,5)
def sm(s):
    s=np.exp(s-s.max(-1,keepdims=True));return s/s.sum(-1,keepdims=True)
def rw(a): return np.concatenate([a,1-a.sum(-1,keepdims=True)],-1)
def model(x,iv):
    B=len(x)
    e=iv('e',(E[x]+P).reshape(B,20)).reshape(B,5,4)
    A=iv('A0',sm(e@M0@e.transpose(0,2,1)/2)[...,:4].reshape(B,20)).reshape(B,5,4)
    o=iv('o0',(rw(A)@e@V0).reshape(B,20)).reshape(B,5,4)
    Z=iv('Z',(e+o).reshape(B,20)).reshape(B,5,4)
    a=iv('a1',sm((Z@M1)@Z.transpose(0,2,1)/2)[:,4,:4])
    o=iv('o1',np.einsum('bk,bkd->bd',rw(a),Z)@V1)
    return iv('r1',Z[:,4]+o)@WU
ALIGN={'e':['embed'],'A0':['attn0.pattern'],'o0':['attn0.out'],'Z':['resid0.mid'],'a1':['attn1.pattern'],'o1':['attn1.out'],'r1':['resid1.mid']}
