import numpy as np
DESCRIPTION="One-hot tokens plus a fixed 5x6 position table U. Two single-head attention layers in a canonical basis: the QK product is folded into one bilinear form per layer (B,C) and the OV product into one map per layer (V,W); the readout is K at position 4. All coefficients are measured, four significant digits."
D="E03F5RF4PFHCFDVGEKGJ8G97G8IHB1GNLFOJG2IFTGFPVG5REX0D43FK0EY9F0KCZRFVRFKBFNLG44FI9FVKDKGGN9DWHF9MF1IFTHE8GGMTFEGH2EF55GTSGAKF4PIZJESXJOOEEPJ7L9G5E56FEAEWOF47EBPGWSE90FRDFKJFZPF1DG75ES0EYSFRTF66FAJFSIF8UFINFKTFR2FO4FQ3GL7FQYFHQGPIEC8FTCGJ5FRIFJ7GOSEE7FS5EUZF57FB5EUFFYMF9QF5XEU1FEZE67G55ESFFHSFLOFOGG2EF2EFR9EJ6GVFKFVEKAED9F3UEKVG3HPAAFI7FH7E9QDGNFG2NYSDS0D50G27FMAFCFBSTFGSFCVF6WDZMHNZGKNEMPEDKEPGE87IAU3KNFBDF8VF8EFGFFPJFM1GJYFMQFV9LPZIAMGPK8D2CCVGRVE51HJWH21S1NI9PG44E2CF60F2EFMMEQQF01H7WGVZFX7DIFDJSG2WG5NF3MFGIENMG6GFFNJ5466DEZAHYBFCF56KGWPM8ZK4YBWOIG1JPTDN0GMEHG5EH3FV7LT7CM2H6OBUJG5CDHADCOD8CGVBCU4G4HE3ADD6"
a=np.array([int(D[i:i+3],36) for i in range(0,612,3)])-20000
b=a.reshape(34,6)/1e3
U=b[:5];B=b[5:11];V=b[11:17];C=b[17:23];W=b[23:29];K=b[29:].T
def sm(s):
    e=np.exp(s-s.max(-1,keepdims=True));return e/e.sum(-1,keepdims=True)
def at(S,E,V):
    p=sm(np.einsum('bid,de,bje->bij',E,S,E));return p,np.einsum('bij,bje->bie',p,E@V)
def model(x,iv):
    n,P=x.shape
    E=iv('embed',(np.eye(6)[x]+U).reshape(n,-1)).reshape(n,P,6)
    p,a=at(B,E,V)
    p=iv('attn0.pattern',p.reshape(n,-1)).reshape(n,P,P)
    a=iv('attn0.out',a.reshape(n,-1)).reshape(n,P,6)
    r=iv('resid0.mid',(E+a).reshape(n,-1)).reshape(n,P,6)
    p,a=at(C,r,W)
    return iv('resid1.mid',(r+a).reshape(n,-1)).reshape(n,P,6)[:,4]@K
ALIGN={k:[k] for k in 'embed attn0.pattern attn0.out resid0.mid resid1.mid'.split()}
