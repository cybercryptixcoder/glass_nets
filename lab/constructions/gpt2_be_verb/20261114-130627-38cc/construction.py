import numpy as np

DESCRIPTION = """
The model uses the cue word to determine tense and the noun to determine number.
At resid0 the cue identity is at position 1 and the noun identity at position 4.
By layer 3 the cue information is available at the last position (4); the noun identity is also there.
Layers 0-5 combine them into a 4-way answer code (present/past x singular/plural).
The answer code is carried at the last position through layers 6-11 and read by the unembedding.
We represent each stage's state with one-hot identities: cue (12), noun (60), and answer (4).
The answer is recomputed from cue and noun at early stages, and the answer code is carried forward
once formed. Later stages keep the answer code as the primary variable.
"""

def _ans_idx(cue_idx, noun_idx):
    return 2 * (cue_idx >= 6).astype(int) + (noun_idx >= 30).astype(int)

def model(x, iv):
    B = x.shape[0]
    cue0 = x[:, 0].astype(int)
    noun0 = x[:, 1].astype(int)

    # Stage 0: resid0
    cue_e_1 = iv("cue_e_1", np.eye(12)[cue0])
    noun_e_4 = iv("noun_e_4", np.eye(60)[noun0])
    cue_idx0 = np.argmax(cue_e_1, axis=1)
    noun_idx0 = np.argmax(noun_e_4, axis=1)

    # Stage 1: resid3 (layers 0-2)
    cue_3_1 = iv("cue_3_1", np.eye(12)[cue_idx0])
    cue_3_4 = iv("cue_3_4", np.eye(12)[np.argmax(cue_3_1, axis=1)])
    noun_3_4 = iv("noun_3_4", np.eye(60)[noun_idx0])
    ans_3_4 = iv("ans_3_4", np.eye(4)[_ans_idx(
        np.argmax(cue_3_4, axis=1), np.argmax(noun_3_4, axis=1))])

    # Stage 2: resid6 (layers 3-5)
    cue_6_1 = iv("cue_6_1", np.eye(12)[np.argmax(cue_3_4, axis=1)])
    cue_6_4 = iv("cue_6_4", np.eye(12)[np.argmax(cue_6_1, axis=1)])
    noun_6_4 = iv("noun_6_4", np.eye(60)[np.argmax(noun_3_4, axis=1)])
    ans_6_4 = iv("ans_6_4", np.eye(4)[_ans_idx(
        np.argmax(cue_6_4, axis=1), np.argmax(noun_6_4, axis=1))])

    # Stage 3: resid9 (layers 6-8)
    cue_9_1 = iv("cue_9_1", np.eye(12)[np.argmax(cue_6_4, axis=1)])
    cue_9_4 = iv("cue_9_4", np.eye(12)[np.argmax(cue_9_1, axis=1)])
    noun_9_4 = iv("noun_9_4", np.eye(60)[np.argmax(noun_6_4, axis=1)])
    ans_9_4 = iv("ans_9_4", ans_6_4)

    # Stage 4: resid12 (layers 9-11)
    ans_12_4 = iv("ans_12_4", ans_9_4)

    logits = 10.0 * ans_12_4 - 5.0
    return logits

ALIGN = {
    "cue_e_1": ["resid0@cue"],
    "noun_e_4": ["resid0@noun"],
    "cue_3_1": ["resid3@cue"],
    "cue_3_4": ["resid3@noun"],
    "noun_3_4": ["resid3@noun"],
    "ans_3_4": ["resid3@noun"],
    "cue_6_1": ["resid6@cue"],
    "cue_6_4": ["resid6@noun"],
    "noun_6_4": ["resid6@noun"],
    "ans_6_4": ["resid6@noun"],
    "cue_9_1": ["resid9@cue"],
    "cue_9_4": ["resid9@noun"],
    "noun_9_4": ["resid9@noun"],
    "ans_9_4": ["resid9@noun"],
    "ans_12_4": ["resid12@noun"],
}

# ---- added by the constructor at the end of the run ----
# These sites failed the judge's tests in the reader's best submission, so the delivered construction does not claim
# them. The variables stay in model(); they are just not aligned there. See run.json for the attempt and its report.
_UNVERIFIED_SITES = ["cut0", "cut3", "cut6", "cut9"]
ALIGN = {v: [s for s in where if s not in _UNVERIFIED_SITES] for v, where in ALIGN.items()}
ALIGN = {v: where for v, where in ALIGN.items() if where}
