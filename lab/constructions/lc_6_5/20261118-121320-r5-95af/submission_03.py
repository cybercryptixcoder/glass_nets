import numpy as np
DESCRIPTION="Account of the network in a canonical token basis (one-hot tokens + 5 position vectors). Layer 0 and layer 1 are single-head attentions with the QK product folded into one bilinear form (B0, B1) and the OV product folded into one map (M0, M1); the readout is WU applied to position 4. Coefficients are measured to 2 decimals."
WP=np.array([[-185,-35,-39,6,-6,126],[143,107,104,243,158,32],[83,50,37,94,-67,-300],[16,-62,-54,-316,58,17],[29,88,10,58,-242,157]])/100
B0=np.array([[-486,-52,-124,123,-380,381],[-10,519,-92,443,273,-96],[1128,-200,1350,-325,1199,-1900],[-409,-11,-166,-100,-351,469],[-375,104,44,178,-125,243],[-208,-148,108,-83,-44,114]])/100
M0=np.array([[-24,11,19,41,31,38],[150,41,8,165,-142,50],[143,43,13,163,-135,45],[-74,-37,-16,-76,69,-21],[-35,-77,-2,-163,92,-83],[8,22,32,82,-47,42]])/100
B1=np.array([[-286,457,1590,-276,-338,-103],[-271,211,3128,23,14,-369],[-625,4,2709,-525,-728,200],[59,-28,-1153,11,-24,-76],[-458,709,362,-254,-335,-230],[-382,911,-3765,-37,-59,-63]])/100
M1=np.array([[3,36,23,145,26,56],[815,371,166,-916,-399,174],[-167,275,211,1635,368,88],[-177,-34,-47,25,-89,-56],[232,189,64,-249,-244,84],[94,-43,3,-101,97,0]])/100
WU=np.array([[481,191,-232,-365,-285],[-1200,884,154,227,186],[-59,610,261,-465,-336],[327,-457,-124,93,90],[-11,390,56,-253,-174],[-1328,555,827,-270,-268]])/100
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
