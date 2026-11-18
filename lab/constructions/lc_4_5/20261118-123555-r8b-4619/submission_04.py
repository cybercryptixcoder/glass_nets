import numpy as np
DESCRIPTION="Two attention layers on R=WE[tok]+WP. Layer0: score R.M0.R^T/2 (M0=WQ0.WK0^T), softmax over keys, add attended values via V0. Layer1: score (R'.M1).R'/2, softmax, position4 adds its attended values via V1. Readout: residual at position4 times WU. Integers, scales in code."
K=np.array([-111,-405,285,60,204,-217,136,369,200,-40,229,279,354,379,219,-104,-128,-3,576,-15,187,-105,451,5,382,-231,119,-335,-260,391,509,-95,-318,-91,-13,412,-314,-230,34,522,-1674,342,-2142,-1536,-2058,1082,-2055,858,85,574,1322,1075,1163,-778,578,-414,338,188,-99,91,105,117,-28,152,-120,213,-29,-168,92,265,-52,209,-58,-110,93,246,-33,-68,58,101,-311,-1056,1275,1863,-52,-154,212,205,-11,231,-60,-16,321,-79,-107,-151,638,-54,-22872])
E=K[:20].reshape(5,4)
P=K[20:40].reshape(5,4)
M0=K[40:56].reshape(4,4)
V0=K[56:72].reshape(4,4)
M1=K[72:88].reshape(4,4)
V1=K[88:104].reshape(4,4)
U=K[104:124].reshape(4,5)
def f(S):
    e=np.exp(S-S.max(-1,keepdims=True))
    return e/e.sum(-1,keepdims=True)
def model(x,iv):
    B=len(x)
    R=iv('R',(E[x]+P).reshape(B,20)/200).reshape(B,5,4)
    Z=iv('Z',(R+f(R@M0@R.transpose(0,2,1)/200)@R@V0/200).reshape(B,20)).reshape(B,5,4)
    a=iv('a',f(Z@M1@Z.transpose(0,2,1)/400).reshape(B,25)).reshape(B,5,5)
    o=iv('o',(a@Z@V1/50).reshape(B,20)).reshape(B,5,4)
    return iv('r',(Z+o)[:,4])@U/200
ALIGN={'R':['embed'],'Z':['resid0.mid'],'a':['attn1.pattern'],'o':['attn1.out'],'r':['resid1.mid']}
