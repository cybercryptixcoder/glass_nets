import numpy as np
DESCRIPTION = "Reduced-gauge account: token+position embeddings (position 3 = gauge zero); two attention layers, each softmax(h M h^T) applied to h @ VO added to the residual; final position read out by WU."
E2 = np.array([[0.204888,0.453383,0.152880],[2.828385,2.243450,-0.028929],[2.113987,2.357577,0.322715],[3.102210,1.585758,1.264188]])
P2 = np.array([[-0.931494,-2.115436,-0.549555],[-1.865087,-1.547001,3.097395],[-5.973404,-0.123080,1.308425],[0.,0.,0.]])
M0 = np.array([[0.186683,-0.394446,0.098268],[-0.448225,-1.262993,1.445701],[-2.827664,-1.653529,2.649654]])
VO0= np.array([[0.030355,-0.773538,0.130014],[1.064353,-0.102242,0.245684],[0.745715,1.113531,-0.106624]])
M1 = np.array([[0.258160,2.938304,0.265750],[-0.195368,-2.235139,-0.211740],[-0.277916,1.272985,3.809736]])
VO1= np.array([[-1.879262,1.571496,-1.811751],[0.045086,0.441714,-0.404851],[-0.169248,-9.846808,5.845982]])
WU = np.array([[37.445528,0.0,35.603833,36.885549],[6.302658,0.0,3.634388,4.531247],[1.063955,0.0,-4.665999,-2.586020]])
def model(x,iv):
    B=x.shape[0]
    h=iv('embed',(E2[x]+P2[None]).reshape(B,-1)).reshape(B,4,3)
    for l,(M,VO) in enumerate([(M0,VO0),(M1,VO1)]):
        sc=h@M@h.transpose(0,2,1); sc=sc-sc.max(-1,keepdims=True)
        a=np.exp(sc); a/=a.sum(-1,keepdims=True)
        a=iv(f'pattern{l}',a.reshape(B,-1)).reshape(B,4,4)
        o=np.einsum('bij,bjk,kl->bil',a,h,VO)
        o=iv(f'attn_out{l}',o.reshape(B,-1)).reshape(B,4,3)
        h=iv(f'resid_mid{l}',(h+o).reshape(B,-1)).reshape(B,4,3)
    return h[:,3]@WU
ALIGN={'embed':['embed'],'pattern0':['attn0.pattern'],'attn_out0':['attn0.out'],'resid_mid0':['resid0.mid'],'pattern1':['attn1.pattern'],'attn_out1':['attn1.out'],'resid_mid1':['resid1.mid']}
