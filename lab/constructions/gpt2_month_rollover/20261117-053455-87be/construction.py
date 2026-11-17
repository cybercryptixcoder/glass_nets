import numpy as np

DESCRIPTION = (
    "GPT-2 reads the date-range prompt as a calendar comparison.  The month identities "
    "m1 and m2 and the input year are carried in the residual stream at their token "
    "positions through the early blocks (resid0, resid3).  By block 6 the end position "
    "also carries the relation label: same year when m2 >= m1, next year when m2 < m1.  "
    "Layers 6-8 combine that relation with the input year into an output-year identity at "
    "the final position (resid9); layers 9-11 pass that output year to the unembedding "
    "(resid12).  The answer logit for the same-year token is highest when the output year "
    "equals the prompt year, for the next-year token when it is prompt year + 1, and for "
    "the previous-year token when it is prompt year - 1.  Variables are categorical "
    "one-hot identities of the month, year, relation, and resulting output year; each "
    "value occurs many times in the prompt set."
)

def _oh(v, K):
    v = np.asarray(v, dtype=int)
    return np.eye(K, dtype=float)[v]

def _rel(m1, m2):
    a = np.argmax(m1, axis=1)
    b = np.argmax(m2, axis=1)
    # 0 = same year (m2 on or after m1), 1 = next year (m2 before m1)
    idx = (b < a).astype(int)
    return np.eye(3, dtype=float)[idx]

def _ans(y, rel):
    yy = np.argmax(y, axis=1) + 1980
    rr = np.argmax(rel, axis=1)
    off = np.where(rr == 1, 1, np.where(rr == 2, -1, 0))
    ay = np.clip(yy + off, 1980, 2000)
    return np.eye(21, dtype=float)[ay - 1980]

def model(x, iv):
    x = np.asarray(x, dtype=int)

    # Embedding stage: the three informative token positions.
    m1_0 = iv("m1_0", _oh(x[:, 0], 12))
    y_0  = iv("y_0",  _oh(x[:, 2] - 1980, 20))
    m2_0 = iv("m2_0", _oh(x[:, 1], 12))

    # Layers 0-2: token identities are still carried at their positions.
    m1_3 = iv("m1_3", m1_0)
    y_3  = iv("y_3",  y_0)
    m2_3 = iv("m2_3", m2_0)

    # Layers 3-5: token identities plus the calendar relation at the end position.
    m1_6 = iv("m1_6", m1_3)
    y_6  = iv("y_6",  y_3)
    m2_6 = iv("m2_6", m2_3)
    rel_6 = iv("rel_6", _rel(m1_6, m2_6))

    # Layers 6-8: combine relation and year into the output year.
    ans_9 = iv("ans_9", _ans(y_6, rel_6))

    # The month/year identities are also present at this depth; they are kept in
    # the cut so substitution at resid9 is faithful, but the answer is read from ans_9.
    m1_9 = iv("m1_9", m1_6)
    y_9  = iv("y_9",  y_6)
    m2_9 = iv("m2_9", m2_6)

    # Layers 9-11: pass the output year to the final residual stream.
    y_12  = iv("y_12",  y_9)
    ans_12 = iv("ans_12", ans_9)

    ay = np.argmax(ans_12, axis=1) + 1980
    yb = x[:, 2]
    cand = np.stack([yb, yb + 1, yb - 1], axis=1)  # same, next, previous
    logits = -np.abs(cand - ay[:, None]).astype(float)
    logits[:, 0] += 0.01  # deterministic tie-break toward "same year"
    return logits


ALIGN = {
    # embedding
    "m1_0": ["resid0@m1", "resid0@year", "resid0@m2"],
    "y_0":  ["resid0@m1", "resid0@year", "resid0@m2"],
    "m2_0": ["resid0@m1", "resid0@year", "resid0@m2"],

    # layers 0-2
    "m1_3": ["resid3@m1", "resid3@year", "resid3@m2"],
    "y_3":  ["resid3@m1", "resid3@year", "resid3@m2"],
    "m2_3": ["resid3@m1", "resid3@year", "resid3@m2"],

    # layers 3-5
    "m1_6": ["resid6@m1", "resid6@year", "resid6@m2"],
    "y_6":  ["resid6@m1", "resid6@year", "resid6@m2"],
    "m2_6": ["resid6@m1", "resid6@year", "resid6@m2"],
    "rel_6": ["resid6@m1", "resid6@year", "resid6@m2"],

    # layers 6-8
    "m1_9": ["resid9@m1", "resid9@year", "resid9@m2"],
    "y_9":  ["resid9@m1", "resid9@year", "resid9@m2"],
    "m2_9": ["resid9@m1", "resid9@year", "resid9@m2"],
    "ans_9": ["resid9@m1", "resid9@year", "resid9@m2"],

    # layers 9-11 (only the final position reaches the unembedding)
    "y_12":  ["resid12@m2"],
    "ans_12": ["resid12@m2"],
}
