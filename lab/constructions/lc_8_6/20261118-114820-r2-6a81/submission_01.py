
import numpy as np
DESCRIPTION = (
 "Read the input as a function f on positions 0..5, f(p)=x_p (6 tokens). The network runs two attention layers "
 "and reads the last position. LAYER 0 (per query position p, over key positions j): the score is a bilinear form "
 "in the (token,position) embeddings, so it splits EXACTLY into four 6x6 tables - G[x_p,x_j] (token-token), "
 "B[x_p,j] (token of the query, key position), C[p,x_j] (query position, token of the key), D[p,j] (two positions); "
 "all four are needed. Attention weights are softmax_j of that score; the value added at each position is the "
 "a0-weighted average of a per-position vector (token table WE[x_j] + position table WP[j]) projected by one "
 "8x8 value-output map, i.e. TE[x_j]+TP[j]. LAYER 1: only position 5 matters; its query is r0[5]@A1 and it "
 "attends over j with weight softmax_j(r0[5]@A1 . r0[j]); the output adds the a1-weighted average of r0[j]@P1 "
 "to r0[5]. READOUT: the 6 logits are r1[5]@WU and the decision is the largest. There is no thresholding, "
 "counting or case analysis anywhere: every logit is a smooth linear readout of two attention-weighted averages, "
 "and the structure the network 'classifies' (fixed points, 2-cycles, preimage counts) only arises implicitly "
 "through these sums. Rules: softmax over keys (layer 0 over the 6 positions for every query, layer 1 over the 6 "
 "positions from position 5 only), value = weighted average, output = linear map of the last residual; the "
 "remaining numbers are the four score tables and the four 8-dim token/position tables.")
def _a(s_,shape): return np.array([float(t) for t in s_.split()]).reshape(shape)
G=_a("3.5605 0.4585 2.7842 -5.8291 -3.7477 -1.5272 0.0430 -0.9802 -1.0006 -4.7115 -2.1904 -0.6891 0.8776 0.0723 -1.3363 -0.4498 1.4949 0.3179 1.2265 5.7633 2.4649 16.3132 -0.0050 1.3644 1.0487 0.0857 1.1420 -3.4677 2.8045 0.6962 9.7382 6.3745 7.3647 7.9014 7.9058 -0.2813",(6,6))
Bt=_a("-4.8018 5.1857 -9.0693 -1.8042 0.9851 1.6907 -1.4456 5.8249 -3.4690 -4.0135 -3.0054 -4.6533 22.5884 -8.1664 22.8355 -8.9926 -9.0992 -3.8780 -21.3179 -2.4527 6.5066 31.8183 0.5718 1.0224 -7.1647 -3.4234 -17.7234 5.2408 13.1756 7.9260 17.6834 -33.2988 6.7081 14.8341 16.8722 27.1612",(6,6))
Ct=_a("-0.9270 -0.4043 -1.3767 0.3522 -1.7046 -0.8274 -0.6455 -1.7008 -0.9502 3.6029 -1.8053 -0.1151 -0.9605 -2.2039 -9.2826 1.1972 -2.2588 -2.4190 0.3262 -0.5926 -1.9275 -11.4558 -0.2541 -1.9218 13.4470 14.7135 13.9077 21.6960 21.4234 1.9432 2.4630 2.0377 5.0318 -1.3960 -1.7618 2.6403",(6,6))
Dt=_a("-19.8303 8.4674 -20.9220 8.1498 4.8775 0.1444 14.4214 -0.7086 12.3193 -14.2151 -5.4957 -1.0271 6.3476 3.1105 -0.6963 -5.0507 -9.7655 -11.3839 6.0940 2.4037 2.4820 -4.3217 -8.1340 -8.8074 55.4759 -74.6425 65.2721 32.0335 10.6325 40.0596 -21.3461 6.2157 -14.6117 13.2707 10.9000 3.8090",(6,6))
WE=_a("-0.6097 -0.1057 -1.6105 0.3764 1.1128 -0.1818 -0.1577 0.0198 -0.0687 -0.2665 -0.2466 0.2747 0.4371 0.1993 -0.2368 0.2485 0.9552 -1.0912 -0.3150 -0.8274 0.4039 0.4189 0.1872 0.9523 -0.1798 2.1800 0.4638 -0.3522 0.4492 1.1441 -0.0833 0.2163 -1.0620 0.0922 1.3427 0.2846 0.4413 -0.1973 0.4021 -0.0918 -2.0123 -0.0398 -0.7205 -1.3029 0.3314 0.0150 0.9065 0.2904",(6,8))
WP=_a("-0.4576 1.1520 -0.0339 0.7371 0.3243 -1.2223 -0.6192 1.2438 0.5413 -0.7447 -1.0702 -0.1334 -1.2177 0.5085 1.1144 -0.2639 -0.6646 -0.2131 0.0556 0.2147 -0.5444 0.8677 -0.2504 3.5778 -0.5267 -0.8446 0.3237 -0.0802 0.3578 0.7395 -1.6647 -0.4734 -0.3608 -0.2123 -0.5091 -4.3772 0.2719 -0.5513 -0.3959 -0.7580 0.0101 0.7159 0.6607 0.4590 2.6205 0.3174 1.1648 -0.0576",(6,8))
TE=_a("-0.6120 -0.4297 0.6420 0.1755 0.4858 -3.7228 0.9627 0.0950 -0.2719 -0.5783 0.3838 0.0320 0.1056 -1.6345 0.5521 0.0602 -0.3874 -0.5821 0.6115 -0.2729 -0.0736 -2.3154 1.3306 0.1214 -0.9741 -0.3059 -0.2408 0.2647 -0.1254 -1.7687 1.0507 0.1519 -0.7668 -0.1056 -0.2720 0.4455 0.0099 -1.8449 0.8482 -0.0097 -0.9508 -0.9549 0.2884 -0.0415 -0.7764 -0.5507 0.3702 -0.4062",(6,8))
TP=_a("0.2149 2.5077 -0.1494 0.8082 1.2379 -0.7078 -0.9163 0.0020 0.4933 -0.5587 0.9400 -1.0199 -0.5615 2.1029 -0.3910 0.0563 -0.1878 -0.6665 2.0183 -0.3074 -0.1828 0.0425 -0.4807 -0.1997 0.2794 -2.7192 -0.3305 -0.0114 -0.9522 2.1649 -0.7738 -0.4135 0.2294 -0.2288 -2.2992 0.1498 -1.5292 5.1448 -1.3011 -0.9121 -2.8100 0.1071 0.2963 0.8634 1.2230 -12.3542 5.3870 0.8545",(6,8))
A1=_a("0.1441 0.1399 -0.0621 -0.2295 0.4043 0.3238 0.4067 0.3473 -0.0894 -0.2812 -0.1419 0.0491 -0.3333 -0.2650 -0.3715 0.0961 0.1053 0.0010 0.1096 0.1143 0.2375 0.1454 0.2435 -0.0822 0.1986 -0.6010 0.2071 0.9627 -0.1965 -0.1445 -0.4957 0.0244 0.0421 0.0983 -0.2429 0.1658 -1.1150 -0.5991 -0.6848 -0.1284 0.1772 -0.1631 0.3445 0.6410 0.6392 0.3382 0.4315 -0.1022 -0.1309 -0.1852 0.6939 0.8801 0.4799 0.2859 0.1073 0.5486 0.3077 -1.5339 0.9574 2.9601 0.8976 0.1888 -0.0110 -0.4096",(8,8))
P1=_a("-0.4194 0.7252 0.1641 1.0228 -0.0896 -0.6176 0.9106 -0.0462 -0.0093 0.1436 -0.1348 -0.2283 -0.0856 -0.1082 -0.5774 0.1034 0.4834 0.4037 -1.4166 -0.2137 -1.8975 -1.9386 -4.8199 0.2621 0.0156 1.4936 -1.4852 0.9135 -2.4814 -2.7304 -4.4372 0.4448 -0.0996 0.4804 -0.3273 -0.5755 -0.0644 -0.5485 -1.6184 0.1933 0.0810 -0.0771 -0.1134 -0.2938 0.0133 -0.1031 -0.6061 -0.0028 0.1102 0.0268 -0.2754 -0.2007 -0.2488 -0.4084 -1.1066 0.0165 -0.1269 0.1338 0.1248 0.0590 0.1842 0.0562 0.3938 0.0032",(8,8))
WU=_a("-2.2972 0.0253 0.6741 1.1797 1.8430 2.8577 0.0945 0.4778 -0.2113 -0.1214 -0.0444 0.1064 0.5478 -0.8072 -0.0664 0.1252 0.0945 0.0167 1.9288 0.8088 -0.9401 -0.8709 -0.7018 -0.6608 1.4186 -2.8487 -0.1423 0.0842 -0.2796 -1.0043 1.3771 -0.5809 -0.2026 -0.3427 -0.5432 -0.8179 1.9681 -0.8370 -0.1592 0.3885 0.7378 1.1850 4.9214 3.2705 -1.2725 -2.5730 -3.3011 -4.2380",(8,6))

def model(x, iv):
    tok=[iv('tok%d'%v,(x==v)*1.0) for v in range(6)]          # [B,6] one-hot per position
    TK=np.stack(tok,2)                                        # [B,6(pos),6(tok)]
    # ---- layer 0: four-part bilinear score, softmax over key positions
    s0=(np.einsum('bpv,vw,bjw->bpj',TK,G,TK)+np.einsum('bpv,vj->bpj',TK,Bt)
        +np.einsum('pv,bjv->bpj',Ct,TK)+Dt[None])
    e=np.exp(s0-s0.max(2,keepdims=True)); a0=e/e.sum(2,keepdims=True)
    a0=np.stack([iv('a0_%d'%p,a0[:,p,:]) for p in range(6)],1)  # attention rows, query p
    E=sum(tok[v][:,:,None]*WE[v] for v in range(6))+WP[None]     # [B,6,8] embeddings
    V0=sum(tok[v][:,:,None]*TE[v] for v in range(6))+TP[None]    # [B,6,8] per-position value
    o0=np.einsum('bji,bid->bjd',a0,V0)                           # attention-weighted average
    o0=np.stack([iv('o0_%d'%p,o0[:,p,:]) for p in range(6)],1)
    r0=np.stack([iv('r0_%d'%p,(E+o0)[:,p,:]) for p in range(6)],1)
    # ---- layer 1: from position 5
    a1=np.einsum('bd,bjd->bj',r0[:,5]@A1,r0)
    a1=np.exp(a1-a1.max(1,keepdims=True)); a1=a1/a1.sum(1,keepdims=True)
    a1=iv('a1',a1)
    o1=iv('o1',(a1[:,:,None]*r0).sum(1)@P1)
    r1=iv('r1_last',r0[:,5]+o1)
    return r1@WU

ALIGN={}
for v in range(6): ALIGN['tok%d'%v]=['embed']
for p in range(6):
    ALIGN['a0_%d'%p]=['attn0.pattern']; ALIGN['o0_%d'%p]=['attn0.out']; ALIGN['r0_%d'%p]=['resid0.mid']
ALIGN['a1']=['attn1.pattern']; ALIGN['o1']=['attn1.out']; ALIGN['r1_last']=['resid1.mid']
