
import numpy as np, math
DESCRIPTION = ("The network runs two bilinear attention rounds on a 4-dim residual stream and reads out the last position. "
 "Residual at position p starts as token_embedding[t_p] + position_embedding[p]. Each round: a single bilinear "
 "score G = W_Q W_K^T (only the product is used) / 2, a softmax over all 5 keys (no mask), and each position adds the "
 "attention-weighted average of the other positions' residuals through the value map M = W_V W_O. Only the last "
 "position's row of round-1 attention matters downstream, and resid0 at all five positions feeds round 1's keys. "
 "The residual basis is chosen so that the readout is the identity: the four residual coordinates ARE the score "
 "gaps of classes (0,1,2,4) against class 3 (which is scored 0), so the decision is a plain argmax over the "
 "last position's residual coordinates with class 3 as reference. 104 numbers describe the whole computation: "
 "5 token codes, 5 position codes, and four 4x4 effective maps (two score forms, two value maps).")
WE=np.array([0.105297,-1.02932,0.398553,0.0385505,-0.362507,-1.67432,0.906683,-0.331482,-0.706685,-1.05864,-0.534537,-0.705072,0.415564,-2.62736,-0.906987,-0.306798,-1.4993,-0.570802,0.116246,-1.20663]).reshape(5,4)
WP=np.array([1.06548,0.0865205,0.440709,0.788627,-1.4219,-1.71548,-5.23146,-2.29489,1.62682,-0.438129,1.29011,1.43798,-1.47356,-0.515539,-4.06046,-1.88897,0.0436534,1.61441,-0.903858,0.204713]).reshape(5,4)
G0=np.array([-3762.86,-961.836,-811.18,5002.18,-1203.53,-309.866,-268.335,1601.24,-730.221,-187.92,-154.079,964.491,5101.46,1313.3,1099.85,-6787.2]).reshape(4,4)
G1=np.array([-222.01,-51.0644,-53.6606,302.26,-28.6987,-9.27133,-6.40162,39.089,-16.1258,-4.36641,-3.19821,21.932,193.802,48.4821,43.9841,-265.409]).reshape(4,4)
M0=np.array([8.1149,11.6095,37.297,14.1945,2.90839,4.09152,9.78256,4.5239,1.29451,2.11826,7.0542,2.49675,-11.5318,-16.1398,-50.1776,-19.7189]).reshape(4,4)
M1=np.array([49.6481,-38.4852,21.1353,30.981,14.7388,-2.99518,23.8971,13.9598,16.3616,-5.48647,27.3502,15.2058,-73.8567,41.9647,-73.0591,-56.3774]).reshape(4,4)

def sm(s):
    s=s-s.max(-1,keepdims=True); e=np.exp(s); return e/e.sum(-1,keepdims=True)

def model(x, iv):
    em=WE[x]+WP[None,:,:]
    em=np.stack([iv("e%d"%p, em[:,p,:].copy()) for p in range(5)],1)
    A0=sm((em@G0)@np.swapaxes(em,1,2)*0.5)
    r0=em+(A0@(em@M0))
    r0=np.stack([iv("r0_%d"%p, r0[:,p,:].copy()) for p in range(5)],1)
    A1=sm((r0@G1)@np.swapaxes(r0,1,2)*0.5)
    A1[:,4,:]=iv("a1p4", A1[:,4,:].copy())
    a1=(A1@(r0@M1))
    a1[:,4,:]=iv("a1o4", a1[:,4,:].copy())
    r1=r0+a1
    r1=np.stack([iv("r1_%d"%p, r1[:,p,:].copy()) for p in range(5)],1)
    y=r1[:,4,:]
    return np.stack([y[:,0],y[:,1],y[:,2],np.zeros_like(y[:,0]),y[:,3]],1)

ALIGN={"e0":["embed"],"e1":["embed"],"e2":["embed"],"e3":["embed"],"e4":["embed"],
 "r0_0":["resid0.mid"],"r0_1":["resid0.mid"],"r0_2":["resid0.mid"],"r0_3":["resid0.mid"],"r0_4":["resid0.mid"],
 "a1p4":["attn1.pattern"],"a1o4":["attn1.out"],
 "r1_0":["resid1.mid"],"r1_1":["resid1.mid"],"r1_2":["resid1.mid"],"r1_3":["resid1.mid"],"r1_4":["resid1.mid"]}
