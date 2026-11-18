import numpy as np
a=np.fromstring("10 -42 27 6 9 -40 4 77 16 98 -18 15 9 -21 35 18 37 -36 40 -14 -1 8 -65 -11 5 -26 23 3 2 -17 -4 50 -56 8 -102 47 -142 91 -74 -66 -119 -18 -81 -24 -4 -45 -13 -74 -16 -135 46 11 -112 8 35 -40 -2 44 -46 42 -67 12 34 -33 25 -50 6 26 39 -23 3 -26 10 -19 -54 -10 -28 14 -5 -46 43 42 34 17 11 6 -46 -53 9 -27 25 -11 13 -40 5 17 23 13 4 6 15 13 11 43 -9 13 2 -13 -63 -52 -91 40 -43 -68 8 17 -38 23 -29 -84 -8 -75 -37 82",sep=" ")/200
def model(x,iv):
 E=a[:20].reshape(5,4);P=a[20:40].reshape(5,4);m=a[40:56].reshape(4,4);o=a[56:72].reshape(4,4);n=a[72:88].reshape(4,4);w=a[88:104].reshape(4,4);u=a[104:].reshape(4,5)
 R=E[x]+P;b=len(x)
 S=R@m@R.transpose(0,2,1);S-=S.max(-1,keepdims=True);A=np.exp(S);A/=A.sum(-1,keepdims=True)
 Z=iv("z",(R+A@R@o).reshape(b,-1)).reshape(b,5,4)
 T=np.einsum("bqd,bkd->bqk",Z@n,Z);T-=T.max(-1,keepdims=True);B=np.exp(T);B/=B.sum(-1,keepdims=True)
 r=Z+np.einsum("bqk,bkd->bqd",B,Z@w)
 return r[:,4]@u
ALIGN={"z":["resid0.mid"]}
D="Two softmax attention layers. L0 scores q.k=R M0 R^T; adds A0(R MO0). L1 scores (Z M1).Z^T; adds A1(Z MO1). Read pos4 through WU. Integer tables /200."
