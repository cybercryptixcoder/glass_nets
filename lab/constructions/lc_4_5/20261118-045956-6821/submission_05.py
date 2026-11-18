
import numpy as np

DESCRIPTION = ("Two attention steps over the residual stream, described with the QK and OV products only (the residual "
  "basis is a gauge freedom, so Q,K,V,O need not be written separately). embed = token+position embedding. Layer-0 "
  "attention scores query/key embeddings with the 4x4 QK0 (logits of order 100, so its softmax is essentially a hard "
  "pointer) and adds OV0 applied to the average of the attended embeddings. Layer-1 attention scores residuals with QK1 "
  "(nearly rank 1: key score = one projection, query score = one projection) and adds OV1 applied to the average of the "
  "attended residuals. The last position's residual is unembedded into 5 class scores.")

WE = np.array([[-0.55467927,-2.0260413,1.4237577,0.30095226],
  [1.0221858,-1.0873666,0.67846268,1.845897],
  [0.99914759,-0.20007868,1.1440504,1.3960565],
  [1.7692684,1.8962755,1.0939301,-0.52155262],
  [-0.64047682,-0.016611824,2.8815513,-0.073833235]])
WP = np.array([[0.93302113,-0.52507764,2.2546914,0.025921376],
  [1.9076753,-1.1535764,0.59385604,-1.6734813],
  [-1.2980766,1.9568782,2.5464535,-0.47379798],
  [-1.592253,-0.45253354,-0.06707824,2.0607474],
  [-1.5712401,-1.1512868,0.16956818,2.6083066]])
WU = np.array([[0.13969728,-0.1018092,-0.02717563,-0.029559035,-0.1578685],
  [-0.043621596,0.11429895,0.11216771,-0.13866065,-0.39538255],
  [0.68908471,-0.41800332,0.19833878,-0.5265218,-1.4645016],
  [-0.31521925,-0.36295864,0.60684919,0.36108619,0.0060048779]])
QK0 = np.array([[-16.742004,3.4242093,-21.421343,-15.362744],
  [-20.57456,10.82288,-20.548366,8.5751949],
  [0.85057018,5.7400581,13.216331,10.749499],
  [11.626506,-7.7762623,5.7809384,-4.1355934]])
QK1 = np.array([[-0.29133108,-0.54899215,0.46523686,1.2308603],
  [-0.16625653,-0.34124126,0.28763994,0.50536607],
  [-1.5563547,-5.2784543,6.3756691,9.3131612],
  [-0.2589757,-0.76969629,1.0612567,1.0248802]])
OV0 = np.array([[1.6914169,0.93760566,-0.49436762,0.45276995],
  [0.52307011,0.58326265,-0.14003223,0.76033288],
  [-0.60052119,1.0629264,-0.14703065,-0.83825804],
  [0.45949412,1.3260114,-0.25803519,1.0431164]])
OV1 = np.array([[-0.53349402,11.539671,-3.0097208,-0.1606452],
  [2.1180059,-1.6214242,-0.46972691,1.4038387],
  [-0.16761479,16.037908,-1.9412859,-3.9370693],
  [-0.53546904,-7.5310877,3.1900926,-2.7078737]])

def _softmax(L):
    L = L - L.max(axis=2, keepdims=True)
    A = np.exp(L)
    return A / A.sum(axis=2, keepdims=True)

def model(x, iv):
    B = x.shape[0]
    e = iv("emb", (WP[None] + WE[x]).reshape(B, 20))
    emb = e.reshape(B, 5, 4)
    A0 = _softmax((emb @ QK0) @ emb.transpose(0, 2, 1) / 2.0)
    p0 = iv("pat0", A0.reshape(B, 25))
    a0 = iv("aout0", (p0.reshape(B, 5, 5) @ emb @ OV0).reshape(B, 20))
    h = iv("res0", e + a0)
    hm = h.reshape(B, 5, 4)
    A1 = _softmax((hm @ QK1) @ hm.transpose(0, 2, 1) / 2.0)
    p1 = iv("pat1", A1.reshape(B, 25))
    a1 = iv("aout1", (p1.reshape(B, 5, 5) @ hm @ OV1).reshape(B, 20))
    r1 = iv("res1", h + a1)
    return r1.reshape(B, 5, 4)[:, -1, :] @ WU

ALIGN = {"emb": ["embed"], "pat0": ["attn0.pattern"], "aout0": ["attn0.out"],
         "res0": ["resid0.mid"], "pat1": ["attn1.pattern"], "aout1": ["attn1.out"],
         "res1": ["resid1.mid"]}
