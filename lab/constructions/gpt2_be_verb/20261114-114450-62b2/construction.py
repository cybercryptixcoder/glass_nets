
import numpy as np

DESCRIPTION = (
 "Prompt: '<|endoftext|><cue>, the <noun>' (5 tokens; cue at position 1, noun at position 4 = last). "
 "Answer in [' is',' are',' was',' were'] = 2*TENSE + NUMBER, TENSE=1 iff cue index>=6 "
 "(Yesterday..Originally), NUMBER=1 iff noun is plural (noun index>=30). "
 "The two facts live at two positions: WHICH CUE WORD is at position 1 (only its tense half is used) and "
 "WHETHER the noun at position 4 is plural. Each named residual depends only on the tokens that produced it, "
 "so the cue identity at a cue-carrying position and the number at the last position explain it. "
 "Stage by stage: at the embedding the cue identity is at position 1 and the number at position 4. Layers 0-2 "
 "copy the cue into position 2 (the comma), so at depth 3 and 6 positions 1 and 2 both carry it; the number "
 "stays at position 4. Layers 6-8 keep the cue at positions 1-2 and start to write the answer class into the "
 "last position: at depth 9 position 4 already carries the 4-way class code, but only its NUMBER half is read "
 "by the later layers - the tense is still taken from the cue at positions 1-2 (swapping just the code at "
 "position 4 changes only the verb number). Layers 9-11 finish the job: at depth 12 the last position holds "
 "the answer class and the four answer logits are read straight off it. model() carries cue and number from "
 "stage to stage, takes the number from the class code at depth 9, and rebuilds the class for the last stage."
)

def oh(v, K):
    return np.eye(K)[np.asarray(v).astype(int).reshape(-1)]

def model(x, iv):
    x = np.asarray(x)
    cue = x[:, 0].astype(int)
    noun = x[:, 1].astype(int)

    t0 = iv("t0", oh(cue, 12))                        # which cue word is at position 1
    n0 = iv("n0", oh((noun >= 30).astype(int), 2))    # is the noun plural

    t3 = iv("t3", np.asarray(t0)); n3 = iv("n3", np.asarray(n0))
    t6 = iv("t6", np.asarray(t3)); n6 = iv("n6", np.asarray(n3))

    t9 = iv("t9", np.asarray(t6))                     # cue identity at positions 1 and 2
    ten = (np.asarray(t9).argmax(1) >= 6).astype(int)
    c9 = iv("c9", oh(2 * ten + np.asarray(n6).argmax(1), 4))   # class code at position 4

    num = np.asarray(c9).argmax(1) % 2                # only its number half is read downstream
    cls = iv("cls12", oh(2 * ten + num, 4))
    return 8.0 * np.asarray(cls) - 4.0

ALIGN = {
    "t0": ["resid0@1"],
    "n0": ["resid0@4"],
    "t3": ["resid3@1", "resid3@2"],
    "n3": ["resid3@4"],
    "t6": ["resid6@1", "resid6@2"],
    "n6": ["resid6@4"],
    "t9": ["resid9@1", "resid9@2"],
    "c9": ["resid9@4"],
    "cls12": ["resid12@4"],
}
