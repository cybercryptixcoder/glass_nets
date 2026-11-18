import numpy as np
DESCRIPTION="Two exact-softmax attention layers on R=W_E[tok]+W_pos: scores bilinear (R M R^T/2), one head each, values added to the residual; logits = residual at the last position times W_U. All seven matrices are tabulated as integers in one base-81 string."
s="JkH0NPLIMdIlM1O>Ma"
T=np.array([ord(c)-35 for c in s])
K=(T[0::2]*81+T[1::2]-3314).reshape(30,4)
E,P,Q,V,W,G,U=np.split(K,[5,10,14,18,22,26])
def f(S):
    e=np.exp(S-S.max(-1,keepdims=True));return e/e.sum(-1,keepdims=True)
def model(x,iv):
    B=len(x)
    R=iv('R',(E[x]+P).reshape(B,20)/150).reshape(B,5,4)
    Z=iv('Z',(R+f(R@Q@R.transpose(0,2,1)/300)@R@V/150).reshape(B,20)).reshape(B,5,4)
    a=iv('a',f(Z@W@Z.transpose(0,2,1)/300).reshape(B,25)).reshape(B,5,5)
    o=iv('o',(a@Z@G/150).reshape(B,20)).reshape(B,5,4)
    r=iv('r',(Z+o)[:,4])
    u=r@U/150
    return np.c_[u,0*u[:,0]]
ALIGN={'R':['embed'],'Z':['resid0.mid'],'a':['attn1.pattern'],'o':['attn1.out'],'r':['resid1.mid']}
