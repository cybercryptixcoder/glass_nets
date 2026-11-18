import numpy as np
DESCRIPTION="Account of the network in a canonical token basis (one-hot tokens + 5 position vectors). Layer 0 and layer 1 are single-head attentions with the QK product folded into one bilinear form (B0, B1) and the OV product folded into one map (M0, M1); the readout is WU applied to position 4. Coefficients are measured; the token embedding is exactly the one-hot basis, so no token table is needed."
SC=0.001
WP = np.array([[-1853,-353,-391,64,-61,1260],[1428,1067,1042,2429,1585,323],[826,500,371,943,-668,-3005],[160,-623,-540,-3161,583,171],[289,884,97,576,-2416,1573]])*SC
B0 = np.array([[-4857,-524,-1240,1227,-3802,3813],[-97,5188,-918,4428,2733,-957],[11285,-1996,13501,-3251,11994,-18995],[-4090,-113,-1665,-1002,-3514,4694],[-3752,1041,438,1777,-1252,2432],[-2078,-1480,1081,-828,-444,1141]])*SC
M0 = np.array([[-242,111,189,414,308,379],[1499,410,78,1654,-1416,496],[1425,430,131,1628,-1345,453],[-741,-373,-159,-761,686,-210],[-347,-775,-21,-1633,921,-833],[80,220,320,822,-474,421]])*SC
B1 = np.array([[-2857,4573,15900,-2758,-3378,-1034],[-2706,2109,31281,233,145,-3689],[-6253,44,27092,-5253,-7281,1995],[592,-277,-11530,108,-237,-765],[-4580,7090,3623,-2544,-3350,-2304],[-3823,9108,-37647,-370,-589,-633]])*SC
M1 = np.array([[31,359,233,1454,258,565],[8151,3710,1656,-9162,-3985,1739],[-1675,2748,2105,16347,3677,884],[-1772,-344,-474,254,-894,-559],[2316,1887,635,-2489,-2440,840],[939,-430,34,-1006,968,3]])*SC
WU = np.array([[4808,1913,-2324,-3654,-2852],[-11995,8835,1542,2272,1863],[-586,6098,2613,-4645,-3364],[3267,-4568,-1241,928,897],[-113,3905,563,-2530,-1738],[-13284,5553,8267,-2696,-2678]])*SC

def sm(s):
    s=s-s.max(-1,keepdims=True); e=np.exp(s); return e/e.sum(-1,keepdims=True)
def model(x, iv):
    B=x.shape[0]; P=x.shape[1]
    oh=(x[...,None]==np.arange(5)).astype(float)
    E=np.concatenate([oh+WP[None,:P,:5], np.broadcast_to(WP[None,:P,5:],(B,P,1))],-1)
    E=iv('embed',E.reshape(B,-1)).reshape(B,P,6)
    p0=iv('pattern0',sm(np.einsum('bid,de,bje->bij',E,B0,E)/np.sqrt(6)).reshape(B,-1)).reshape(B,P,P)
    a0=iv('attn_out0',np.einsum('bij,bje->bie',p0,E@M0).reshape(B,-1)).reshape(B,P,6)
    r0=iv('resid_mid0',(E+a0).reshape(B,-1)).reshape(B,P,6)
    p1=sm(np.einsum('bid,de,bje->bij',r0,B1,r0)/np.sqrt(6))
    r1=iv('resid_mid1',(r0+np.einsum('bij,bje->bie',p1,r0@M1)).reshape(B,-1)).reshape(B,P,6)
    return r1[:,4]@WU
ALIGN={'embed':['embed'],'pattern0':['attn0.pattern'],'attn_out0':['attn0.out'],'resid_mid0':['resid0.mid'],'resid_mid1':['resid1.mid']}
