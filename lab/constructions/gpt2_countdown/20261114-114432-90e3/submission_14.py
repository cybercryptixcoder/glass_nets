
import numpy as np

DESCRIPTION = (
    "The four numbers form an arithmetic sequence and the model predicts the next term: "
    "answer = n4 + step, step = sign of the majority of the adjacent differences (prefix irrelevant). "
    "The network writes the *identity of the predicted answer* into the residual stream: at the n4 "
    "position from layer 6 on, and at the final comma from layer 9 on; the logits are a linear read-out "
    "of that one-hot code (a positive weight on the answer token). Nothing else is needed at those depths: "
    "the earlier number positions are no longer read once the answer identity is formed."
)

def model(x, iv):
    x = np.asarray(x, dtype=float)
    B = x.shape[0]
    n1 = x[:, 0]; n2 = x[:, 1]; n3 = x[:, 2]; n4 = x[:, 3]
    d1 = np.sign(n2 - n1); d2 = np.sign(n3 - n2); d3 = np.sign(n4 - n3)
    step = iv("step", np.sign(d1 + d2 + d3))
    nxt = np.clip((n4 + step).astype(int), 0, 361)
    h = np.eye(362)[nxt]
    a6 = iv("ans6", h)
    a7 = iv("ans7", a6)
    a12 = iv("ans12", a7)
    out = np.zeros((B, 362))
    out[np.arange(B), a12.argmax(1)] = 10.0
    return out

ALIGN = {
    "ans6": ["resid6@n4"],
    "ans7": ["resid7@n4"],
    "ans12": ["resid12@end"],
}
