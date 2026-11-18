
import numpy as np

DESCRIPTION = ("EXPERIMENT: layer-0 attention is treated as a pure hard pointer (one-hot argmax) instead of a softmax. "
  "Same products QK0/QK1/OV0/OV1 as the verified construction; only the layer-0 softmax is replaced by its argmax, to "
  "measure how much of the behaviour depends on the soft part of the pointer. embed = token+position embedding "
  "(position 0's vector shifted into the token table).")

WE = np.array([[0.37834186,-2.55111894,3.6784491,0.32687364],
  [1.95520693,-1.61244424,2.93315408,1.87181838],
  [1.93216872,-0.72515632,3.3987418,1.42197788],
  [2.70228953,1.37119786,3.3486215,-0.49563124],
  [0.29254431,-0.54168946,5.1362427,-0.04791186]])
WP = np.array([[0.,0.,0.,0.],
  [0.97465417,-0.62849876,-1.66083536,-1.69940268],
  [-2.23109773,2.48195584,0.2917621,-0.49971936],
  [-2.52527413,0.0725441,-2.32176964,2.03482602],
  [-2.50426123,-0.62620916,-2.08512322,2.58238522]])
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

def _hard(L):
    M = (L == L.max(axis=2, keepdims=True)).astype(float)
    return M / M.sum(axis=2, keepdims=True)

def model(x, iv):
    B = x.shape[0]
    e = iv("emb", (WP[None] + WE[x]).reshape(B, 20))
    emb = e.reshape(B, 5, 4)
    S0 = (emb @ QK0) @ emb.transpose(0, 2, 1) / 2.0
    A0 = 0.5 * _hard(S0) + 0.5 * _softmax(S0)
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

