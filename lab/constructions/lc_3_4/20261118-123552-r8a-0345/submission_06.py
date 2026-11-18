import numpy as np
DESCRIPTION = "Two attention layers, width 3. Embed token+position (position 3 is gauge zero). Each layer computes softmax(h M h^T) and adds (pattern @ h) @ VO to the residual. The last position is read out linearly by WU."
E = np.array([0.2049,0.4534,0.1529,2.8284,2.2434,-0.0289,2.1140,2.3576,0.3227,3.1022,1.5858,1.2642]).reshape(4,3)
P = np.array([-0.9315,-2.1154,-0.5496,-1.8651,-1.5470,3.0974,-5.9734,-0.1231,1.3084,0.0,0.0,0.0]).reshape(4,3)
M0= np.array([0.1867,-0.3944,0.0983,-0.4482,-1.2630,1.4457,-2.8277,-1.6535,2.6497]).reshape(3,3)
V0= np.array([0.0304,-0.7735,0.1300,1.0644,-0.1022,0.2457,0.7457,1.1135,-0.1066]).reshape(3,3)
M1= np.array([0.2582,2.9383,0.2658,-0.1954,-2.2351,-0.2117,-0.2779,1.2730,3.8097]).reshape(3,3)
V1= np.array([-1.8793,1.5715,-1.8118,0.0451,0.4417,-0.4049,-0.1692,-9.8468,5.8460]).reshape(3,3)
U = np.array([37.4455,0.0,35.6038,36.8855,6.3027,0.0,3.6344,4.5312,1.0640,0.0,-4.6660,-2.5860]).reshape(3,4)
def model(x,iv):
    B=x.shape[0]
    h=iv('embed',(E[x]+P).reshape(B,-1)).reshape(B,4,3)
    for l,(M,VO) in enumerate([(M0,V0),(M1,V1)]):
        s=h@M@h.transpose(0,2,1); s=s-s.max(-1,keepdims=True)
        a=np.exp(s); a/=a.sum(-1,keepdims=True)
        a=iv('pattern%d'%l,a.reshape(B,-1)).reshape(B,4,4)
        o=iv('attn_out%d'%l,np.einsum('bij,bjk,kl->bil',a,h,VO).reshape(B,-1)).reshape(B,4,3)
        h=iv('resid_mid%d'%l,(h+o).reshape(B,-1)).reshape(B,4,3)
    return h[:,3]@U
ALIGN={'embed':['embed'],'pattern0':['attn0.pattern'],'attn_out0':['attn0.out'],'resid_mid0':['resid0.mid'],'pattern1':['attn1.pattern'],'attn_out1':['attn1.out'],'resid_mid1':['resid1.mid']}
