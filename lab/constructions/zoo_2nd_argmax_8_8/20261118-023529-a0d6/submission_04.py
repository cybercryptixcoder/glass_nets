
import numpy as np

DESCRIPTION = ("Hypothesis test: at each step the state may just record simple prefix statistics of the inputs (running max, running min, running sum, "
               "and the current number). The output here is still produced by the exact recursion so that behaviour is not the issue - the site report "
               "tells me which of these semantics the real hidden state actually carries.")

Wx = np.array([-0.2073,0.0302,-1.2027,-0.1506,-0.4672,0.0032,0.6188,0.0040])
Wh = np.array([
[ 0.0817, 0.3865, 0.7401,-1.3310,-0.6463, 0.1432,-0.3229, 0.6877],
[-0.1382, 0.4797,-0.8372, 0.0943, 0.7727,-1.1995,-0.4489, 1.9091],
[ 0.6685,-0.1553, 0.2351, 0.0359, 0.9014, 1.7540, 0.9659,-0.5198],
[ 1.2273, 1.9529,-0.5836,-0.6715,-0.1939, 0.9701, 0.1679, 0.1140],
[-0.3183,-0.8647,-0.4506, 1.2788, 0.2721,-0.0427, 0.5373,-0.4094],
[-0.2141,-0.0229,-0.3180, 1.0977,-0.6634, 0.7855, 0.8663, 0.6138],
[ 1.3999,-0.1146, 0.5128, 0.3298, 1.2084, 0.5880, 0.4884,-1.1035],
[-0.2462, 0.7908, 0.0335, 0.1042,-0.2450, 0.2823, 1.8160,-0.6004]])
Wo = np.array([
[-0.8521,-0.3524,-0.2432,-0.1793,-0.5582,-0.1289, 2.1222,-0.3717],
[ 0.6794, 0.1149,-0.0501,-0.2234,-1.2559,-0.2163, 0.3626, 0.1263],
[-0.1604, 0.0717,-0.1359, 0.1546, 0.5337, 0.1673, 0.6589,-4.4499],
[ 0.4240, 0.0231, 0.1565,-0.1993,-0.2877,-2.5932, 1.3143, 0.0448],
[ 0.5565,-0.0034, 0.2122, 0.1732, 0.4811, 0.8646,-2.2472, 0.2676],
[ 0.5340, 0.3342, 0.5564, 0.2751,-0.1111, 0.2445,-2.4800, 0.3442],
[-0.4327,-0.0988,-0.4478, 0.1051, 0.7073, 0.1084, 0.0275,-0.0958],
[-0.0038, 0.0113, 0.0672,-0.6583, 0.3722, 0.0293,-0.3641, 0.0130]])

def model(x, iv):
    N = x.shape[0]
    mx = np.full(N, -1e9); mn = np.full(N, 1e9); s = np.zeros(N)
    h = np.zeros((N, 8))
    for t in range(8):
        mx = np.maximum(mx, x[:, t]); mn = np.minimum(mn, x[:, t]); s = s + x[:, t]
        iv("runmax%d" % (t+1), mx)
        iv("runmin%d" % (t+1), mn)
        iv("runsum%d" % (t+1), s)
        iv("last%d" % (t+1), x[:, t])
        h = np.maximum(x[:, t:t+1]*Wx + h @ Wh, 0.0)
    return h @ Wo

ALIGN = {}
for t in range(1, 9):
    ALIGN["runmax%d" % t] = ["h%d" % t]
    ALIGN["runmin%d" % t] = ["h%d" % t]
    ALIGN["runsum%d" % t] = ["h%d" % t]
    ALIGN["last%d" % t] = ["h%d" % t]
