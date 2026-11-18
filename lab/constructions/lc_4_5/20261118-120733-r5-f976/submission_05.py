import numpy as np
DESCRIPTION=("Residual = token + position embedding (5x4). Two softmax attention layers, no mask or bias: "
"score=(x A_l x^T)/2, A_l=W_Ql W_Kl^T, softmax over keys, then x+=attn(x) V_l, V_l=W_Vl W_Ol. Layer 0 is nearly one-hot, "
"layer 1 soft. Only position 4 is read: out=(x4+a1.(x V1)) W_U. Matrices are the network's own integers/1000; an attention row "
"keeps 4 weights, the 5th is 1 minus their sum.")
t=np.array("-555 -2026 1424 301 1022 -1087 678 1846 999 -200 1144 1396 1769 1896 1094 -522 -640 -17 2882 -74 933 -525 2255 26 1908 -1154 594 -1673 -1298 1957 2546 -474 -1592 -453 -67 2061 -1571 -1151 170 2608 -16742 3424 -21421 -15363 -20575 10823 -20548 8575 851 5740 13216 10749 11627 -7776 5781 -4136 1691 938 -494 453 523 583 -140 760 -601 1063 -147 -838 459 1326 -258 1043 -291 -549 465 1231 -166 -341 288 505 -1556 -5278 6376 9313 -259 -770 1061 1025 -533 11540 -3010 -161 2118 -1621 -470 1404 -168 16038 -1941 -3937 -535 -7531 3190 -2708 140 -102 -27 -30 -158 -44 114 112 -139 -395 689 -418 198 -527 -1465 -315 -363 607 361 6".split(),float)/1000
E=t[:20].reshape(5,4);P=t[20:40].reshape(5,4);M0=t[40:56].reshape(4,4);V0=t[56:72].reshape(4,4)
M1=t[72:88].reshape(4,4);V1=t[88:104].reshape(4,4);U=t[104:].reshape(4,5)
def sm(s):
    s=np.exp(s-s.max(-1,keepdims=True));return s/s.sum(-1,keepdims=True)
def rw(a):return np.concatenate([a,1-a.sum(-1,keepdims=True)],-1)
def model(x,iv):
    b=len(x)
    e=iv('e',(E[x]+P).ravel()).reshape(b,5,4)
    A=iv('A',sm((e@M0)@e.transpose(0,2,1)/2)[...,:4].ravel()).reshape(b,5,4)
    o=iv('o',(rw(A)@e@V0).ravel()).reshape(b,5,4)
    Z=iv('Z',(e+o).ravel()).reshape(b,5,4)
    a=iv('a',sm((Z@M1)@Z.transpose(0,2,1)/2)[:,4,:4])
    q=iv('q',np.einsum('bk,bkd->bd',rw(a),Z)@V1)
    return iv('r',Z[:,4]+q)@U
ALIGN={'e':['embed'],'A':['attn0.pattern'],'o':['attn0.out'],'Z':['resid0.mid'],'a':['attn1.pattern'],'q':['attn1.out'],'r':['resid1.mid']}
