
import numpy as np

DESCRIPTION = (
 "Prompt: '<|endoftext|><cue>, the <noun>' (5 tokens; cue at position 1, noun at position 4 = last). "
 "Answer in [' is',' are',' was',' were'] = 2*TENSE + NUMBER, TENSE=1 iff cue index>=6 "
 "(Yesterday..Originally), NUMBER=1 iff noun is plural (noun index>=30). Two independent binary facts: "
 "the tense bit sits at the cue position (2), the number bit at the last position (4). Position 0 "
 "(<|endoftext|>), 2 (',') and 3 (' the') are constants of the prompt, and the residual at each named "
 "position depends only on the token that produced it. "
 "Carried quantities, stage by stage: at the embedding (resid0) position 1 holds the tense bit and position "
 "4 the number bit; these are carried forward and only the bits matter to the later layers. At depth 3 and 6 "
 "the tense bit is still at position 1. Between depth 6 and 9 it moves to position 2 (the comma): by depth 9 "
 "the tense signal at position 1 is no longer read, the one at position 2 is. The number bit stays at "
 "position 4 throughout. At depth 12 the last position holds the 4-way answer class (tense,number) and the "
 "answer logits are read straight off it. model() carries each bit from stage to stage and combines them "
 "into that class."
)

def oh(v, K):
    return np.eye(K)[np.asarray(v).astype(int).reshape(-1)]

def model(x, iv):
    x = np.asarray(x)
    cue = x[:, 0].astype(int)
    noun = x[:, 1].astype(int)
    t = oh((cue >= 6).astype(int), 2)      # tense bit
    n = oh((noun >= 30).astype(int), 2)    # number bit

    t0 = iv("t0", t); n0 = iv("n0", n)
    t3 = iv("t3", np.asarray(t0)); n3 = iv("n3", np.asarray(n0))
    t6 = iv("t6", np.asarray(t3)); n6 = iv("n6", np.asarray(n3))
    t9 = iv("t9", np.asarray(t6)); n9 = iv("n9", np.asarray(n6))
    c = 2 * np.asarray(t9).argmax(1) + np.asarray(n9).argmax(1)
    cls = iv("cls12", oh(c, 4))
    return 8.0 * np.asarray(cls) - 4.0

ALIGN = {
    "t0": ["resid0@1"], "n0": ["resid0@4"],
    "t3": ["resid3@1"], "n3": ["resid3@4"],
    "t6": ["resid6@1"], "n6": ["resid6@4"],
    "t9": ["resid9@2"], "n9": ["resid9@4"],
    "cls12": ["resid12@4"],
}
