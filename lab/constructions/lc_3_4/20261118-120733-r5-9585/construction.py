import numpy as np
DESCRIPTION="Attention-only transformer: 3-wide residual = token+position embedding; each layer adds the softmax_j(h_i M h_j)-weighted values h_j V; the last position is read out by U (class-1 column dropped)."
E=np.array([[0.20489,0.45338,0.15288],[2.8284,2.2434,-0.028928],[2.114,2.3576,0.32271],[3.1022,1.5858,1.2642]])
P=np.array([[-0.93149,-2.1154,-0.54955],[-1.8651,-1.547,3.0974],[-5.9734,-0.12308,1.3084],[0,0,0]])
M0=np.array([[0.18668,-0.39445,0.09827],[-0.44822,-1.263,1.4457],[-2.8277,-1.6535,2.6497]])
V0=np.array([[0.030347,-0.77354,0.13001],[1.0643,-0.10224,0.24568],[0.74572,1.1135,-0.10662]])
M1=np.array([[0.25816,2.9383,0.26575],[-0.19537,-2.2351,-0.21174],[-0.27792,1.273,3.8097]])
V1=np.array([[-1.8793,1.5715,-1.8117],[0.045087,0.44171,-0.40485],[-0.16925,-9.8468,5.846]])
U=np.array([[37.446,0,35.604,36.886],[6.3027,0,3.6344,4.5313],[1.064,0,-4.666,-2.586]])
def model(x, iv):
    B = x.shape[0]
    h = iv('e', (E[x] + P).reshape(B, -1)).reshape(B, 4, 3)
    for l, (m, v) in enumerate(((M0, V0), (M1, V1))):
        s = h @ m @ h.transpose(0, 2, 1)
        a = np.exp(s - s.max(-1, keepdims=True)); a = a / a.sum(-1, keepdims=True)
        a = iv('p%d' % l, a.reshape(B, -1)).reshape(B, 4, 4)
        o = iv('o%d' % l, (a @ (h @ v)).reshape(B, -1)).reshape(B, 4, 3)
        h = iv('r%d' % l, (h + o).reshape(B, -1)).reshape(B, 4, 3)
    return h[:, 3] @ U
ALIGN = {'e': ['embed'], 'p0': ['attn0.pattern'], 'o0': ['attn0.out'], 'r0': ['resid0.mid'], 'p1': ['attn1.pattern'], 'o1': ['attn1.out'], 'r1': ['resid1.mid']}
