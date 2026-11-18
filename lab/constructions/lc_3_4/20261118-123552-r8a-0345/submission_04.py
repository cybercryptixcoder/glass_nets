import numpy as np
DESCRIPTION="Two attention layers, width 3. Embeds token+position (pos3 zero). Each layer: softmax(h M h^T), add (pattern@h)@VO. Read last position by WU."
E=np.array([0.205,0.453,0.153,2.828,2.243,-0.029,2.114,2.358,0.323,3.102,1.586,1.264]).reshape(4,3)
P=np.array([-0.931,-2.115,-0.550,-1.865,-1.547,3.097,-5.973,-0.123,1.308,0,0,0]).reshape(4,3)
M0=np.array([0.187,-0.394,0.098,-0.448,-1.263,1.446,-2.828,-1.654,2.650]).reshape(3,3)
V0=np.array([0.030,-0.774,0.130,1.064,-0.102,0.246,0.746,1.114,-0.107]).reshape(3,3)
M1=np.array([0.258,2.938,0.266,-0.195,-2.235,-0.212,-0.278,1.273,3.810]).reshape(3,3)
V1=np.array([-1.879,1.571,-1.812,0.045,0.442,-0.405,-0.169,-9.847,5.846]).reshape(3,3)
U=np.array([37.446,0,35.604,36.886,6.303,0,3.634,4.531,1.064,0,-4.666,-2.586]).reshape(3,4)
def model(x,iv):
 B=x.shape[0];h=iv('embed',(E[x]+P[None]).reshape(B,-1)).reshape(B,4,3)
 for l,(M,VO) in enumerate([(M0,V0),(M1,V1)]):
  s=h@M@h.transpose(0,2,1);s=s-s.max(-1,keepdims=True)
  a=np.exp(s);a/=a.sum(-1,keepdims=True)
  a=iv('pattern%d'%l,a.reshape(B,-1)).reshape(B,4,4)
  o=iv('attn_out%d'%l,np.einsum('bij,bjk,kl->bil',a,h,VO).reshape(B,-1)).reshape(B,4,3)
  h=iv('resid_mid%d'%l,(h+o).reshape(B,-1)).reshape(B,4,3)
 return h[:,3]@U
ALIGN={'embed':['embed'],'pattern0':['attn0.pattern'],'attn_out0':['attn0.out'],'resid_mid0':['resid0.mid'],'pattern1':['attn1.pattern'],'attn_out1':['attn1.out'],'resid_mid1':['resid1.mid']}
