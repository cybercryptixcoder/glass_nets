
import numpy as np
DESCRIPTION = ("Diagnostic exact re-implementation of the 2-layer attention-only transformer; "
               "only the embedding stage is aligned, to token+position embeddings per position. "
               "Layer0 and layer1 are plain 1-head softmax attention (no mask, no biases) added to the residual.")
WE = np.array([[1.303,0.258,0.942],[0.828,0.076,1.370],[-1.594,1.030,0.315],[0.171,0.428,-0.962]])
WP = np.array([[1.415,1.799,2.992],[-0.132,-0.655,1.292],[-4.175,-0.700,-0.066],[-0.105,1.203,0.867]])
W0Q = np.array([[0.162,0.506,-0.849],[-2.851,0.995,0.635],[-0.235,0.174,1.358]])
W0K = np.array([[-0.476,-0.203,0.881],[-1.359,1.592,-0.567],[0.536,-1.485,0.326]])
W0V = np.array([[0.283,-0.967,0.040],[-0.320,0.564,1.175],[0.111,0.353,-0.081]])
W0O = np.array([[-0.510,0.885,0.174],[0.154,-1.218,-0.130],[-0.166,0.053,-0.206]])
W1Q = np.array([[-0.001,0.474,-3.683],[-0.317,-0.983,-0.000],[0.745,0.001,0.250]])
W1K = np.array([[-1.778,-0.477,-0.002],[-1.215,0.069,0.104],[0.001,-0.000,-4.788]])
W1V = np.array([[2.046,0.050,-0.276],[2.948,-0.677,0.196],[-1.375,-1.169,-1.116]])
W1O = np.array([[-2.226,0.900,1.762],[0.226,-1.115,1.079],[0.123,0.194,0.385]])
WU = np.array([[2.915,-3.388,1.144,0.247],[11.061,12.343,-3.341,1.325],[-1.261,2.389,-24.543,12.903]])

def _attn(h, WQ, WK, WV, WO):
    q = h @ WQ; k = h @ WK; v = h @ WV
    S = q @ k.transpose(0, 2, 1) / np.sqrt(3.0)
    A = np.exp(S - S.max(-1, keepdims=True)); A = A / A.sum(-1, keepdims=True)
    return h + (A @ v) @ WO, A

def model(x, iv):
    x = np.asarray(x).astype(int)
    B = x.shape[0]
    h = iv("emb", (WE[x] + WP).reshape(B, 12)).reshape(B, 4, 3)
    h0, _ = _attn(h, W0Q, W0K, W0V, W0O)
    h1, _ = _attn(h0, W1Q, W1K, W1V, W1O)
    return h1[:, -1] @ WU

ALIGN = {"emb": ["embed"]}
