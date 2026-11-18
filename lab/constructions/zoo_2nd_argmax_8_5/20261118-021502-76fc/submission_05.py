
import numpy as np
DESCRIPTION = ("Growth-dominated gated amplifier chain (5 steps, no biases). Step 1 holds only two numbers: p=x1^+ drives units 7,5,0,1 (gains 7.00,0.64,0.42,0.045) and "
 "n=x1^- drives units 2,4,6 (6.57,4.91,6.42). Each later step is a gate: unit = relu(x_t*gain_u + weighted sum of the previous state), with unit 7 a positive-feedback "
 "accumulator (1.09 of itself + 0.96 of unit3 + 0.79 of unit5) so the state grows like 1.09^t; at step 2 only 6 combinations of the state matter (units 0,5 are just "
 "copies of the rest) and at step 4 unit 2 is redundant. Units 1 and 6 are nearly always shut. The final 5 class scores are a fixed linear map of the last state; "
 "class 0 wins unless the amplified positive branch becomes large.")
WX=np.array([0.4235,0.0451,-6.5737,-0.0541,-4.9132,0.6426,-6.4213,7.0017])
WH=np.array([[-0.8475,-0.8279,-0.7468,-0.8905,0.3478,0.7712,-0.4647,0.2159],
[0.0816,0.0236,0.5697,1.4186,-0.6185,0.1272,-0.1902,0.02],
[0.2723,0.6004,0.4761,0.2793,0.1986,0.5457,0.4929,-0.129],
[-0.0865,-0.7503,-0.1069,-0.2517,0.0606,0.1549,1.102,-0.3243],
[0.5281,-0.4238,0.1254,0.944,0.2399,-0.3628,-0.1915,0.1257],
[0.6795,-0.027,-0.1774,-0.3,0.8149,-0.2364,-1.2074,0.7867],
[0.1558,0.9728,0.881,0.1268,-0.7322,0.1691,-0.1633,0.0696],
[0.3804,0.0612,0.0359,0.9617,-0.0172,0.4242,-1.6559,1.0877]])
WO=np.array([[-0.3767,0.6789,0.3173,-0.8669,0.3597],[0.5665,-1.1366,4.8036,-0.0691,0.3778],
[-2.4261,-0.8445,0.3365,-0.4678,-1.9484],[2.1603,-3.9047,-3.3868,0.493,-2.253],
[0.7274,-0.9378,0.6334,-0.4028,0.2398],[0.6797,0.8486,-1.3507,-2.8565,0.716],
[0.5178,-1.3102,-0.8712,4.1999,3.8852],[-0.2393,0.3559,-0.4673,-0.6731,-0.7161]])
K2=[1,2,3,4,6,7]; D2=[0,5]           # at step2 units 0,5 are linear copies of the other six
R2=[[0.34,-0.0,0.36,-0.02,-0.04,0.06,0.01],[-0.02,-0.01,0.39,-0.02,-0.05,0.09,0.01]]
K4=[0,1,3,4,5,6,7]; D4=[2]           # at step4 unit 2 is a linear copy of the other seven
R4=[0.14,0.84,0.03,0.2,0.23,0.78,-0.23,0.75]
def reg(h, iv, tag, keep, drop, R, mu):
    cols=[]
    for i in keep:
        cols.append(iv("%s_%d"%(tag,i), h[:,i]))
    full=np.zeros((h.shape[0],8))
    for jj,i in enumerate(keep): full[:,i]=cols[jj]
    F=np.column_stack(cols+[np.ones(h.shape[0])])
    if drop:
        rec=F@np.array(R).T
        for jj,i in enumerate(drop): full[:,i]=rec[:,jj]
    else:
        for i in drop: full[:,i]=mu[i]
    return full
def model(x, iv):
    p=iv("p",np.maximum(x[:,0],0.0)); n=iv("n",np.maximum(-x[:,0],0.0))
    h=p[:,None]*np.maximum(WX,0)+n[:,None]*np.maximum(-WX,0)      # step 1: x1^+ and x1^- branches
    h=np.maximum(x[:,[1]]*WX+h@WH,0.0); h=reg(h,iv,"a",K2,D2,R2,np.array([1.9,3.3,5.4,5.4,1.7,2.4,1.9,5.3]))
    h=np.maximum(x[:,[2]]*WX+h@WH,0.0); h=reg(h,iv,"b",list(range(8)),[],None,None)
    h=np.maximum(x[:,[3]]*WX+h@WH,0.0); h=reg(h,iv,"c",K4,D4,[R4],np.zeros(8))
    h=np.maximum(x[:,[4]]*WX+h@WH,0.0); h=reg(h,iv,"d",list(range(8)),[],None,None)
    return h@WO
ALIGN={"p":["h1"],"n":["h1"]}
for i in K2: ALIGN["a_%d"%i]=["h2"]
for i in range(8): ALIGN["b_%d"%i]=["h3"]
for i in K4: ALIGN["c_%d"%i]=["h4"]
for i in range(8): ALIGN["d_%d"%i]=["h5"]
