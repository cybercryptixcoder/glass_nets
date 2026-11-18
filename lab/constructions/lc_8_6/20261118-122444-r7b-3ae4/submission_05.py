
import numpy as np
DESCRIPTION=("tokens are one-hot; the embedding is W_E[token]+W_pos. Layer 0 scores every pair (query p, key j) by "
 "e[p].A0.e[j], softmaxes over keys and adds the values e[j].P0, so each position holds a learned soft summary of the "
 "map f near it. Layer 1 reads only position 5: it scores keys j by r0[5].A1.r0[j], mixes values into v1, and the "
 "output is r0[5].W_U + v1.(P1.W_U). The six numbers named 'logits' ARE the output.")
'''+mats+'''
def model(x,iv):
    tok=[]
    for p in range(6):
        oh=(x[:,p][:,None]==np.arange(6)[None,:]).astype(float)
        tok.append(iv("t%d"%p,oh))
    E=np.stack([tok[p]@WE for p in range(6)],1)+WP[None]
    s0=np.einsum('bpd,bjd->bpj',E@A0,E)
    a0=np.exp(s0-s0.max(2,keepdims=True)); a0/=a0.sum(2,keepdims=True)
    r0=E+np.einsum('bpj,bjd->bpd',a0,E@P0)
    s1=np.einsum('bd,bjd->bj',r0[:,5]@A1,r0)
    a1=np.exp(s1-s1.max(1,keepdims=True)); a1/=a1.sum(1,keepdims=True)
    v1=np.einsum('bj,bjd->bd',a1,r0)
    out=r0[:,5]@WU+v1@MW
    return iv("logits",out)
ALIGN={"t0":["embed"],"t1":["embed"],"t2":["embed"],"t3":["embed"],"t4":["embed"],"t5":["embed"],"logits":["resid1.mid"]}
'''
