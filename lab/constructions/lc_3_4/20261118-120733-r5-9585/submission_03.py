import numpy as np
DESCRIPTION = 'Attention-only: 2 layers, 1 head, width 3, no MLP, no biases. h = token embedding + position embedding (position 3 has no position vector). Each layer adds to every position the value VO weighted by its row-softmaxed bilinear scores s_ij = h_i M h_j^T. The last position is read out linearly, class 1 as the reference class. All numbers are measured constants.'
E  = np.array([[0.2049,0.4534,0.1529],[2.828,2.243,-0.02893],[2.114,2.358,0.3227],[3.102,1.586,1.264]])
P  = np.array([[-0.9315,-2.115,-0.5496],[-1.865,-1.547,3.097],[-5.973,-0.1231,1.308],[0,0,0]])
M0 = np.array([[0.1867,-0.3944,0.09827],[-0.4482,-1.263,1.446],[-2.828,-1.654,2.65]])
V0 = np.array([[0.03035,-0.7735,0.13],[1.064,-0.1022,0.2457],[0.7457,1.114,-0.1066]])
M1 = np.array([[0.2582,2.938,0.2658],[-0.1954,-2.235,-0.2117],[-0.2779,1.273,3.81]])
V1 = np.array([[-1.879,1.571,-1.812],[0.04509,0.4417,-0.4049],[-0.1692,-9.847,5.846]])
WU = np.array([[37.45,0,35.6,36.89],[6.303,0,3.634,4.531],[1.064,0,-4.666,-2.586]])

def model(x, iv):
    B = len(x)
    h = iv('embed', E[x]+P).reshape(B,4,3)
    for l,(m,vo) in enumerate(((M0,V0),(M1,V1))):
        a = h@m@h.transpose(0,2,1); a = np.exp(a-a.max(-1,keepdims=True)); a = a/a.sum(-1,keepdims=True)
        a = iv('pat%d'%l, a.reshape(B,-1)).reshape(B,4,4)
        o = iv('out%d'%l, (a@(h@vo)).reshape(B,-1)).reshape(B,4,3)
        h = iv('res%d'%l, (h+o).reshape(B,-1)).reshape(B,4,3)
    return h[:,3]@WU

ALIGN = {'embed':['embed'],'pat0':['attn0.pattern'],'out0':['attn0.out'],'res0':['resid0.mid'],
         'out1':['attn1.out'],'res1':['resid1.mid']}
