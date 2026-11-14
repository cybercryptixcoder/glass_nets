
import numpy as np

DESCRIPTION = (
    "The four numbers are an arithmetic progression and the model predicts the next term: "
    "answer = n4 + s, s = +1 for an ascending progression, -1 for a descending one (s is the sign of the "
    "majority of the three adjacent differences; the leading prefix is irrelevant). "
    "Working backwards from the answer: by layer 6 the n4-position of the residual stream already holds, "
    "in a linearly readable form, exactly the two quantities the rest of the network needs - the identity "
    "of the number n4 (a one-hot code) and the direction of the progression (a one-bit code). From there on "
    "nothing else at that depth is used; the answer identity is formed at the last comma by layer 8-9 and "
    "carried to resid12@end as a one-hot code of the answer, whose logits are a linear read-out "
    "(a single large weight on the answer token)."
)

def model(x, iv):
    x = np.asarray(x, dtype=float)
    B = x.shape[0]
    n1, n2, n3, n4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    asc = (np.sign(n2 - n1) + np.sign(n3 - n2) + np.sign(n4 - n3) > 0).astype(int)
    # layer 4-6: the n4 position holds n4's identity and the progression direction
    a6 = iv("n4v6", np.eye(128)[n4.astype(int)])
    s6 = iv("st6", np.eye(2)[asc])
    n4b = a6.argmax(1).astype(float)
    ascb = s6.argmax(1)
    # layer 7-8: same two codes, now also at the last-comma position
    a8 = iv("n4v8", np.eye(128)[n4b.astype(int)])
    s8 = iv("st8", np.eye(2)[ascb])
    ae = iv("n4ve", np.eye(128)[a8.argmax(1)])
    se = iv("ste", np.eye(2)[s8.argmax(1)])
    nxt = np.clip((ae.argmax(1) + 2 * se.argmax(1) - 1), 0, 361)
    ans = iv("ans12", np.eye(362)[nxt])
    out = np.zeros((B, 362))
    out[np.arange(B), ans.argmax(1)] = 10.0
    return out

ALIGN = {
    "n4v6": ["resid6@n4"],
    "st6": ["resid6@n4"],
    "n4v8": ["resid8@n4"],
    "st8": ["resid8@n4"],
    "n4ve": ["resid8@end"],
    "ste": ["resid8@end"],
    "ans12": ["resid12@end"],
}
