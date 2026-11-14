
import numpy as np

DESCRIPTION = (
    "The four numbers form an arithmetic sequence and the model predicts the next term "
    "(answer = n4 + step, step = sign of the majority of the adjacent differences; the prefix is "
    "irrelevant). The embedding at the four number positions is exactly the token+position embedding, "
    "i.e. the concatenated identities of the four numbers, which is what the rest of the network uses. "
    "At the very end of the network the last comma position holds the identity of the predicted answer "
    "(one-hot), and the logits are a linear read-out of it (large positive weight on that token)."
)

def model(x, iv):
    x = np.asarray(x, dtype=float)
    B = x.shape[0]
    blocks = [np.eye(128)[x[:, j].astype(int)] for j in range(4)]
    vals = iv("vals4", np.concatenate(blocks, axis=1))
    n = [vals[:, 128 * j:128 * (j + 1)].argmax(1).astype(float) for j in range(4)]
    n1, n2, n3, n4 = n
    step = np.sign(np.sign(n2 - n1) + np.sign(n3 - n2) + np.sign(n4 - n3))
    nxt = np.clip((n4 + step).astype(int), 0, 361)
    a12 = iv("ans12", np.eye(362)[nxt])
    out = np.zeros((B, 362))
    out[np.arange(B), a12.argmax(1)] = 10.0
    return out

ALIGN = {
    "vals4": ["resid0@n1", "resid0@n2", "resid0@n3", "resid0@n4"],
    "ans12": ["resid12@end"],
}
