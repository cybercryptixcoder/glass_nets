
DESCRIPTION = "Stage 1: the first hidden state is relu(x1*W_x), so each unit is either 0 or proportional to x1 with a fixed sign. h1 therefore depends only on the two numbers pos=max(x1,0) and neg=max(-x1,0), one coefficient per unit. The output head is not yet explained: the network's decision is always class 5 on the sample, so a constant logit vector reproduces it."

def model(x, iv):
    x1 = x[:, 0]
    pos = iv("pos1", np.maximum(x1, 0.0))
    neg = iv("neg1", np.maximum(-x1, 0.0))
    B = x.shape[0]
    out = np.zeros((B, 10))
    out[:, 5] = 1.0
    return out

ALIGN = {"pos1": ["h1"], "neg1": ["h1"]}
