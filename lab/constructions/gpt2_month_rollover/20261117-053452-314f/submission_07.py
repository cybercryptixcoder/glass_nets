import numpy as np

DESCRIPTION = """Cuts for the embedding, layers 0-2 and layers 3-5 stages, plus the behaviour.

BEHAVIOUR. The answer is the sign of T = logit(next)-logit(same) ('previous' never wins). T is a
function of the two month tokens and (weakly) the year token: T = 0.83 + A[m1,m2] + B[year], R2 0.96,
residual sd 0.24. A is a clean month-order table: positive when the end month is at or before the
start month (so the range rolls into the next year), negative once m2 has passed m1, with a ~1.3 wide
diagonal transition band. This decides 100% of the clearly-decided prompts.

CUTS. The state at every early depth is the input tokens themselves, so at each named position I
name the three token identities as one-hots (which month is at position 1, which at position 4,
which year is at position 2; each value occurs in at least 12 prompts). Measured with a held-out
split (fit the affine map on 2400 prompts, substitute on 480): resid0 -> answers 100% survive,
resid1 100%, resid2 97.5%, resid3 97.7%, resid4 97.3%. The end-position identity code starts to be
entangled after layer 4 (resid5 87%), so the cut for the later bands is still missing."""

A = np.array([
 [ 1.21,-0.78,-0.54,-0.57,-0.51,-0.50,-0.53,-0.94,-1.04,-1.27,-1.44,-1.49],
 [ 0.59, 0.89,-1.37,-1.04,-1.16,-0.62,-0.65,-1.02,-1.42,-1.40,-1.53,-1.57],
 [ 0.61, 0.24, 0.18,-1.30,-1.14,-0.98,-0.65,-0.99,-1.33,-1.42,-1.50,-1.49],
 [ 0.82, 0.56, 0.06, 0.73,-1.32,-0.82,-0.79,-0.88,-1.18,-1.30,-1.68,-1.56],
 [ 1.02, 0.54, 0.25, 0.02,-0.04,-1.10,-0.80,-1.02,-1.11,-1.35,-1.47,-1.52],
 [ 1.41, 1.12, 0.68, 0.61,-0.00, 0.70,-0.95,-1.15,-1.39,-1.06,-1.36,-1.36],
 [ 1.71, 1.34, 1.04, 0.86, 0.34, 0.55, 0.73,-1.12,-1.19,-1.25,-1.08,-1.27],
 [ 1.84, 1.54, 1.11, 1.08, 0.18, 0.66, 0.36, 0.57,-1.30,-1.03,-1.27,-1.04],
 [ 2.03, 1.53, 1.22, 1.22, 0.50, 0.77, 0.55, 0.30, 0.78,-1.29,-1.23,-1.05],
 [ 2.35, 1.89, 1.35, 1.50, 0.54, 1.13, 0.80, 0.74, 0.44, 0.72,-1.41,-0.79],
 [ 2.67, 2.09, 1.76, 1.81, 1.02, 1.40, 1.18, 0.98, 0.74, 0.10, 1.03,-1.20],
 [ 2.36, 2.16, 1.72, 1.74, 0.90, 1.49, 1.15, 1.00, 0.73, 0.21, 0.03, 0.93]])
B = np.array([0.11,-0.54,0.15,0.13,-0.43,-0.15,0.05,0.22,0.18,-0.39,0.35,-0.23,
              0.69,0.65,-0.05,-0.03,0.39,0.07,-0.12,-0.23])
T0 = 0.8318

def model(x, iv):
    m1 = x[:, 0].astype(int); m2 = x[:, 1].astype(int); yi = x[:, 2].astype(int) - 1980
    v1 = iv("m1_id", np.eye(12)[m1])
    v2 = iv("m2_id", np.eye(12)[m2])
    v3 = iv("year_id", np.eye(20)[yi])
    a = np.argmax(v1, 1); b = np.argmax(v2, 1); c = np.argmax(v3, 1)
    T = T0 + A[a, b] + B[c]
    cs = np.full_like(T, -149.03)
    return np.stack([cs, cs + T, cs + T - 1.5], axis=1)

ALIGN = {"m1_id":  ["resid0@1", "resid2@1", "resid2@2", "resid2@3", "resid2@4",
                    "resid4@1", "resid4@2", "resid4@3", "resid4@4"],
         "m2_id":  ["resid0@1", "resid0@4", "resid2@1", "resid2@2", "resid2@3", "resid2@4",
                    "resid4@1", "resid4@2", "resid4@3", "resid4@4"],
         "year_id":["resid0@2", "resid2@1", "resid2@2", "resid2@3", "resid2@4",
                    "resid4@1", "resid4@2", "resid4@3", "resid4@4"]}
