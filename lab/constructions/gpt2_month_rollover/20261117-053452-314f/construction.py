import numpy as np

DESCRIPTION = """Rough first pass. The task is to predict the year token that follows "m1 YYYY - m2":
the answer is one of three year tokens (same year, next year, previous year). What I have
established so far: mean-ablating the end position (position 4) at any depth destroys the
answer (0.35-0.74 survival), the year token at position 2 matters until late (still 0.89 at
resid10@2, 0.97 at resid11@2), and position 1 (m1) stops mattering after layer ~8
(resid8@1 ablated: answers survive 1.00). So the end-position residual at depth 12 is the
place where the answer-year decision lives, and the 'which year number' part is copied from
position 2 by late-layer attention. The behaviour is mostly 'next year' with a switch to
'same year' as the second month gets late in the year, so the first draft encodes a single
month-order scalar at the final residual and reads the three answer logits off it."""

def model(x, iv):
    m1 = x[:, 0].astype(float)
    m2 = x[:, 1].astype(float)
    # single scalar: how much earlier m2 is than m1 (drives 'next year')
    s = iv("month_order", (m1 - m2) / 11.0)
    z = 2.0 * s
    out = np.stack([np.zeros_like(z), z, -z], axis=1)
    return out

ALIGN = {"month_order": ["resid12@end"]}
