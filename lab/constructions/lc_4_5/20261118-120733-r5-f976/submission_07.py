import numpy as np
DESCRIPTION=("Residual = token + position embedding. Two softmax attention layers: score=(x A x^T)/2 with A=W_Q W_K^T, softmax over keys, "
"then x+=attn(x) V. Layer 0 nearly one-hot, layer 1 soft. Only position 4 is read: out=(x4+a1.(x V1)) W_U. Matrices are the network's own, "
"as integers over the given scale; an attention row keeps 4 weights, the 5th is 1 minus their sum.")
E=np.array("-166 -608 427 90 307 -326 204 554 300 -60 343 419 531 569 328 -156 -192 -5 864 -22".split(),float).reshape(5, 4)/300
P=np.array("280 -158 676 8 572 -346 178 -502 -389 587 764 -142 -478 -136 -20 618 -471 -345 51 782".split(),float).reshape(5, 4)/300
M0=np.array("-1674 342 -2142 -1536 -2057 1082 -2055 858 85 574 1322 1075 1163 -778 578 -414".split(),float).reshape(4, 4)/100
V0=np.array("1691 938 -494 453 523 583 -140 760 -601 1063 -147 -838 459 1326 -258 1043".split(),float).reshape(4, 4)/1000
M1=np.array("-58 -110 93 246 -33 -68 58 101 -311 -1056 1275 1863 -52 -154 212 205".split(),float).reshape(4, 4)/200
V1=np.array("-53 1154 -301 -16 212 -162 -47 140 -17 1604 -194 -394 -54 -753 319 -271".split(),float).reshape(4, 4)/100
U=np.array("7 -5 -1 -1 -8 -2 6 6 -7 -20 34 -21 10 -26 -73 -16 -18 30 18 0".split(),float).reshape(4, 5)/50
def sm(s):
    s=np.exp(s-s.max(-1,keepdims=True));return s/s.sum(-1,keepdims=True)
def rw(a):return np.concatenate([a,1-a.sum(-1,keepdims=True)],-1)
def model(x,iv):
    b=len(x)
    e=iv('e',(E[x]+P).reshape(b,20)).reshape(b,5,4)
    A=iv('A',sm((e@M0)@e.transpose(0,2,1)/2)[...,:4].reshape(b,20)).reshape(b,5,4)
    o=iv('o',(rw(A)@e@V0).reshape(b,20)).reshape(b,5,4)
    Z=iv('Z',(e+o).reshape(b,20)).reshape(b,5,4)
    a=iv('a',sm((Z@M1)@Z.transpose(0,2,1)/2)[:,4,:4])
    q=iv('q',np.einsum('bk,bkd->bd',rw(a),Z)@V1)
    return iv('r',Z[:,4]+q)@U
ALIGN={'e':['embed'],'A':['attn0.pattern'],'o':['attn0.out'],'Z':['resid0.mid'],'a':['attn1.pattern'],'q':['attn1.out'],'r':['resid1.mid']}
