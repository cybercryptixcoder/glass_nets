
import numpy as np
DESCRIPTION = ("Two-layer attention-only transformer, residual width 3, 4 positions. Attention scores are "
 "q.k/sqrt(3) which depends on the input only through the composite QK (product of the query and key maps), "
 "and the attention output depends on values only through the composite VO (product of value and output maps). "
 "Both layers are written with these composites. The measured weight values are stored as integers in units of "
 "one ten-thousandth; the matrices below are therefore the network's weights at the measured precision.")

u = 1 / 10000
E = np.array([[-15940,-9618,2582],[10295,8282,764],[3151,9424,4281],[13033,1706,13696]]) * u
P = np.array([[8674,-7002,-6549],[-662,-1318,29920],[-41745,12921,12031],[17989,14152,-1054]]) * u
U = np.array([[129028,-245428,110611,123428],[29150,-33877,2467,11436],[23887,13247,-33413,-12613]]) * u
QK0 = np.array([[3233,-6832,1702],[-7763,-21876,25040],[-48977,-28640,45893]]) * u
VO0 = np.array([[304,-7735,1300],[10644,-1022,2457],[7457,11135,-1066]]) * u
QK1 = np.array([[4471,50893,4603],[-3384,-38714,-3667],[-4814,22049,65987]]) * u
VO1 = np.array([[-18793,15715,-18118],[451,4417,-4049],[-1692,-98468,58460]]) * u

def _sm(s):
    s = s - s.max(-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(-1, keepdims=True)

def model(x, iv):
    B = x.shape[0]
    emb = E[x] + P[None, :, :]
    e = iv("embed", emb.reshape(B, 12))
    h = e.reshape(B, 4, 3)
    p0 = _sm((h @ QK0) @ h.transpose(0, 2, 1) / np.sqrt(3.0))
    p0 = iv("p0", p0.reshape(B, 16)).reshape(B, 4, 4)
    ao = iv("ao", ((p0 @ h) @ VO0).reshape(B, 12)).reshape(B, 4, 3)
    h = h + ao
    r0 = iv("r0", h.reshape(B, 12)).reshape(B, 4, 3)
    h = r0
    p1 = _sm((h @ QK1) @ h.transpose(0, 2, 1) / np.sqrt(3.0))
    p1 = iv("p1", p1.reshape(B, 16)).reshape(B, 4, 4)
    a1 = iv("a1", ((p1 @ h) @ VO1).reshape(B, 12)).reshape(B, 4, 3)
    h = h + a1
    r1 = iv("r1", h.reshape(B, 12)).reshape(B, 4, 3)
    return r1[:, 3, :] @ U

ALIGN = {"embed": ["embed"], "p0": ["attn0.pattern"], "ao": ["attn0.out"],
         "r0": ["resid0.mid"], "p1": ["attn1.pattern"], "a1": ["attn1.out"],
         "r1": ["resid1.mid"]}
