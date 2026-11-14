
import numpy as np

DESCRIPTION = (
 "Prompt: '<|endoftext|><cue>, the <noun>' (5 tokens; cue at position 1, noun at position 4 = last). "
 "Answer in [' is',' are',' was',' were'] = 2*TENSE + NUMBER, TENSE=1 iff cue index>=6 "
 "(Yesterday..Originally), NUMBER=1 iff noun is plural (noun index>=30). "
 "Two independent facts: WHICH CUE WORD is at position 1 (only its tense half is used) and WHETHER the noun "
 "at position 4 is plural. The residual at each named position depends only on the tokens that produced it, "
 "so the cue identity at a cue-carrying position and the number bit at the last position explain it. "
 "Stage by stage: at the embedding, the cue identity is at position 1 and the number bit at position 4. "
 "Layers 0-2: the cue identity is copied from position 1 into position 2 (the comma), so by depth 3 both "
 "position 1 and position 2 carry it; the number bit stays at position 4. Layers 3-5: same, the cue is at "
 "positions 1 and 2 at depth 6. Layers 6-8: by depth 9 the cue has also been copied into position 3 (the "
 "' the'), so positions 1,2,3 all carry it; position 4 still carries the number bit. Layers 9-11: the last "
 "position assembles the 4-way answer class (tense,number) and the answer logits come straight off it. "
 "model() carries the cue identity and the number bit from stage to stage and combines them into the class."
)

def oh(v, K):
    return np.eye(K)[np.asarray(v).astype(int).reshape(-1)]

def model(x, iv):
    x = np.asarray(x)
    cue = x[:, 0].astype(int)
    noun = x[:, 1].astype(int)

    t0 = iv("t0", oh(cue, 12))                 # which cue word is at position 1
    n0 = iv("n0", oh((noun >= 30).astype(int), 2))   # is the noun plural

    t3 = iv("t3", np.asarray(t0)); n3 = iv("n3", np.asarray(n0))
    t6 = iv("t6", np.asarray(t3)); n6 = iv("n6", np.asarray(n3))
    t9 = iv("t9", np.asarray(t6)); n9 = iv("n9", np.asarray(n6))

    c = 2 * (np.asarray(t9).argmax(1) >= 6).astype(int) + np.asarray(n9).argmax(1)
    cls = iv("cls12", oh(c, 4))
    return 8.0 * np.asarray(cls) - 4.0

ALIGN = {
    "t0": ["resid0@1"],
    "n0": ["resid0@4"],
    "t3": ["resid3@1", "resid3@2"],
    "n3": ["resid3@4"],
    "t6": ["resid6@1", "resid6@2"],
    "n6": ["resid6@4"],
    "t9": ["resid9@1", "resid9@2", "resid9@3"],
    "n9": ["resid9@4"],
    "cls12": ["resid12@4"],
}
