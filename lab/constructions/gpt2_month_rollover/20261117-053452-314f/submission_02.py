import numpy as np

DESCRIPTION = """The answer logits of the three year tokens are (verified to 2e-4) exactly the last
LayerNorm of resid12 at the end position (position 4 = the second month) dot the unembedding rows
of year-1, year, year+1. So the whole decision lives in resid12@end and is an affine functional of
it: only the two differences next-same and prev-same matter. Decomposing those differences over the
writes into position 4 shows the decision is built there: mlp9@4 contributes +1.05 (of the mean),
mlp10@4 +0.45, attn8@4 -0.25, and nothing else contributes much; the year itself does not move
during layers 9-11 (mean-ablating position 2 at resid10/resid11 keeps 89%/97% of the answers).
The construction therefore models the last stage as two numbers held at resid12@end: T = logit(next)
- logit(same) and U = logit(prev) - logit(same), computed here with an additive table in m1, m2 and
year (a first approximation: it explains about half of T's variance; the rest is a month-by-year
interaction I have not yet traced). U is negative everywhere, so 'previous year' never wins."""

# additive tables measured on the prompts (main effects of m1, m2 and year on T = next - same)
A_M1 = [-0.30, -0.11, -0.06, -0.31, 0.17, -0.05, -0.07, 0.42, 0.37, -0.11, 0.12, -0.08]
A_M2 = [1.55, 1.09, 0.62, 0.55, -0.06, 0.22, 0.03, -0.30, -0.61, -0.87, -1.08, -1.12]
A_YR = [-0.25, -0.09, 0.12, 0.26, -0.13, 0.05, 0.24, 0.45, -0.32, -0.14,
        0.16, 0.24, -0.33, -0.13, 0.12, 0.28, -0.34, -0.24, -0.05, 0.08]
MEAN_T = 0.85

def model(x, iv):
    m1 = x[:, 0].astype(int)
    m2 = x[:, 1].astype(int)
    yr = x[:, 2].astype(int) - 1980
    t = MEAN_T + np.array(A_M1)[m1] + np.array(A_M2)[m2] + np.array(A_YR)[yr]
    T = iv("T_next_minus_same", t)
    U = iv("U_prev_minus_same", np.full_like(T, -1.5))
    out = np.stack([np.zeros_like(T), T, U], axis=1)
    return out

ALIGN = {"T_next_minus_same": ["resid12@end"],
         "U_prev_minus_same": ["resid12@end"]}
