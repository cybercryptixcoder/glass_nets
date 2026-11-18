import numpy as np
a=np.fromstring("""DATA""",sep=" ")
D="Reduced exact form: R=WE[tok]+WP; A0=softmax(R M0 R^T); Z=R+A0 R MO0; A1=softmax((Z M1).Z^T); out=(Z+A1 Z MO1)[4]WU. Tables x1000."
def model(x,iv):
 i=0
 def g(k,sh):
  nonlocal i
  r=a[i:i+k].reshape(sh)/1000;i+=k;return r
 E=g(20,(5,4));P=g(20,(5,4));m=g(16,(4,4));o=g(16,(4,4));n=g(16,(4,4));w=g(16,(4,4));u=g(20,(4,5))
 R=iv("e",(E[x]+P).reshape(len(x),-1)).reshape(-1,5,4)
 S=R@m@R.transpose(0,2,1);S=S-S.max(-1,keepdims=True);A=np.exp(S);A/=A.sum(-1,keepdims=True)
 Z=iv("z",(R+A@R@o).reshape(len(x),-1)).reshape(-1,5,4)
 T=np.einsum("bqd,bkd->bqk",Z@n,Z);T=T-T.max(-1,keepdims=True);B=np.exp(T);B/=B.sum(-1,keepdims=True)
 r=Z+np.einsum("bqk,bkd->bqd",B,Z@w)
 return r[:,4]@u
ALIGN={"z":["resid0.mid"]}
