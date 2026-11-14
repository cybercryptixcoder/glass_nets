import numpy as np

DESCRIPTION = (
 "GPT-2 small continues a run of consecutive integers: after an increasing run it emits n4+1, "
 "after a decreasing run n4-1. The only two things the model needs are WHICH number comes last "
 "(n4) and WHICH WAY the run goes (sign of n4-n1, i.e. of the sum of the three consecutive "
 "differences). Both are number-token identities, so every variable below is a one-hot identity "
 "code (one-hot over the 362 number tokens ' 0'...' 361', or over the 2 directions, or over the "
 "4 prefixes) -- the form the residual stream holds a token in at its own position. "
 "STAGE embedding (resid0): each of the four number positions carries its own token; the two "
 "prefix positions carry the prefix identity. "
 "STAGE layers 0-2 (cut resid3): the number identities stay on their own positions; the commas "
 "c1,c2,c3 carry the previous number (previous-token copying); comparing the numbers gives the "
 "run direction, which by layer 3 sits on the last position (end, the final comma). "
 "STAGE layers 3-5 (cut resid6): the last number is copied to the end position (attention from "
 "the final comma back to n4), so resid6@end holds (n4, direction). "
 "STAGE layers 6-8 (cut resid9): the successor/predecessor map is applied at the end position, so "
 "resid9@end holds n4+direction = the answer's number identity, alongside n4 and the direction. "
 "STAGE layers 9-11 (cut resid12): the end position holds only the answer identity; ln_f + W_U "
 "read out ' n4+direction'. "
 "The model below mirrors that chain: every later stage's variables are computed from the values "
 "iv returns for the previous stage, so an intervention on any variable (a patched n4, a patched "
 "direction, a patched answer code) propagates exactly the way the network propagates it, and the "
 "output logits are those of the answer identity the final stage holds."
)

N = 362


def _oh(k, K):
    k = np.clip(np.asarray(k, dtype=np.int64).reshape(-1), 0, K - 1)
    return np.eye(K)[k]


def _id(v):
    v = np.asarray(v, dtype=np.float64)
    if v.ndim == 1:
        v = v[None, :]
    return np.argmax(v, axis=1)


def model(x, iv):
    x = np.asarray(x, dtype=np.int64)
    n1, n2, n3, n4, pf = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]

    # ---------------- stage 0 : embeddings (resid0) ----------------
    a1 = iv("emb_n1", _oh(n1, N))
    a2 = iv("emb_n2", _oh(n2, N))
    a3 = iv("emb_n3", _oh(n3, N))
    a4 = iv("emb_n4", _oh(n4, N))
    iv("emb_pfx1", _oh(pf, 4))
    iv("emb_pfx2", _oh(pf, 4))
    w1, w2, w3, w4 = _id(a1), _id(a2), _id(a3), _id(a4)

    # ---------------- stage 1 : layers 0-2 (resid3) ----------------
    b1 = iv("L3_n1", _oh(w1, N))
    b2 = iv("L3_n2", _oh(w2, N))
    b3 = iv("L3_n3", _oh(w3, N))
    b4 = iv("L3_n4", _oh(w4, N))
    iv("L3_c1_n1", _oh(w1, N))       # comma c1 has copied n1
    iv("L3_c2_n2", _oh(w2, N))       # comma c2 has copied n2
    iv("L3_c3_n3", _oh(w3, N))       # comma c3 has copied n3
    w1, w2, w3, w4 = _id(b1), _id(b2), _id(b3), _id(b4)
    d = (w2 - w1) + (w3 - w2) + (w4 - w3)
    inc = (d >= 0).astype(np.int64)            # 1 = increasing run, 0 = decreasing
    bd = iv("L3_end_dir", _oh(inc, 2))
    inc = _id(bd)

    # ---------------- stage 2 : layers 3-5 (resid6) ----------------
    c1 = iv("L6_n1", _oh(w1, N))
    c2 = iv("L6_n2", _oh(w2, N))
    c3 = iv("L6_n3", _oh(w3, N))
    c4 = iv("L6_n4", _oh(w4, N))
    iv("L6_c1_n1", _oh(w1, N))
    iv("L6_c2_n2", _oh(w2, N))
    iv("L6_c3_n3", _oh(w3, N))
    w1, w2, w3, w4 = _id(c1), _id(c2), _id(c3), _id(c4)
    cn = iv("L6_end_n4", _oh(w4, N))            # end position now also holds n4
    cd = iv("L6_end_dir", _oh(inc, 2))
    w4 = _id(cn)
    inc = _id(cd)

    # ---------------- stage 3 : layers 6-8 (resid9) ----------------
    e1 = iv("L9_n1", _oh(w1, N))
    e2 = iv("L9_n2", _oh(w2, N))
    e3 = iv("L9_n3", _oh(w3, N))
    e4 = iv("L9_n4", _oh(w4, N))
    w1, w2, w3, w4 = _id(e1), _id(e2), _id(e3), _id(e4)
    iv("L9_c1_n1", _oh(w1, N))
    iv("L9_c2_n2", _oh(w2, N))
    iv("L9_c3_n3", _oh(w3, N))
    en = iv("L9_end_n4", _oh(w4, N))
    ed = iv("L9_end_dir", _oh(inc, 2))
    w4 = _id(en)
    inc = _id(ed)
    s = np.where(inc > 0, 1, -1)
    ans = np.clip(w4 + s, 0, N - 1)             # successor / predecessor applied
    ea = iv("L9_end_ans", _oh(ans, N))
    ans = _id(ea)

    # ---------------- stage 4 : layers 9-11 (resid12) ----------------
    fa = iv("L12_end_ans", _oh(ans, N))
    ans = _id(fa)

    # logits over the 362 number tokens ' 0' ... ' 361'
    K = np.arange(N)[None, :]
    logits = -0.5 * np.abs(K - ans[:, None])
    logits[np.arange(ans.shape[0]), ans] = 12.0
    return logits


ALIGN = {
    # embedding stage (resid0)
    "emb_n1": ["resid0@n1"], "emb_n2": ["resid0@n2"],
    "emb_n3": ["resid0@n3"], "emb_n4": ["resid0@n4"],
    "emb_pfx1": ["resid0@1"], "emb_pfx2": ["resid0@2"],
    # layers 0-2 (resid3)
    "L3_n1": ["resid3@n1"], "L3_n2": ["resid3@n2"],
    "L3_n3": ["resid3@n3"], "L3_n4": ["resid3@n4"],
    "L3_c1_n1": ["resid3@c1"], "L3_c2_n2": ["resid3@c2"],
    "L3_c3_n3": ["resid3@c3"], "L3_end_dir": ["resid3@end"],
    # layers 3-5 (resid6)
    "L6_n1": ["resid6@n1"], "L6_n2": ["resid6@n2"],
    "L6_n3": ["resid6@n3"], "L6_n4": ["resid6@n4"],
    "L6_c1_n1": ["resid6@c1"], "L6_c2_n2": ["resid6@c2"],
    "L6_c3_n3": ["resid6@c3"],
    "L6_end_n4": ["resid6@end"], "L6_end_dir": ["resid6@end"],
    # layers 6-8 (resid9)
    "L9_n1": ["resid9@n1"], "L9_n2": ["resid9@n2"],
    "L9_n3": ["resid9@n3"], "L9_n4": ["resid9@n4"],
    "L9_c1_n1": ["resid9@c1"], "L9_c2_n2": ["resid9@c2"],
    "L9_c3_n3": ["resid9@c3"],
    "L9_end_n4": ["resid9@end"], "L9_end_dir": ["resid9@end"],
    "L9_end_ans": ["resid9@end"],
    # layers 9-11 (resid12)
    "L12_end_ans": ["resid12@end"],
}
