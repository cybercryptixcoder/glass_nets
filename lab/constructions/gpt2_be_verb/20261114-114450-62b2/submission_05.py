
import numpy as np

DESCRIPTION = (
 "Prompt: '<|endoftext|><cue>, the <noun>' (5 tokens; cue at position 1, noun at position 4 = last). "
 "Answer in [' is',' are',' was',' were'] = 2*TENSE + NUMBER, TENSE=1 iff cue index>=6 "
 "(Yesterday..Originally), NUMBER=1 iff noun is plural (noun index>=30). "
 "Two independent binary facts sit at two positions and never mix until late: which cue word is at "
 "position 1 (only the tense half of it is used) and which noun is at position 4 (only the number half is "
 "used). Both are present already at the embedding (resid0 there is exactly token+position embedding, so a "
 "one-hot of the token at that position predicts that vector exactly) and are carried unchanged up the "
 "residual stream at their own positions (the residual at position 1 depends only on the cue token, the one "
 "at position 4 only on the noun token, so same one-hots still fit exactly at every depth). Positions 0 "
 "(<|endoftext|>), 2 (',') and 3 (' the') are constant across prompts and carry nothing. "
 "Tense leaves position 1 around layers 4-9 and by depth 9 the last position carries what is needed "
 "together with the cue at position 1; in the last layer the answer class (tense,number) is assembled at "
 "the last position, whose residual then contains the 4-way class. model() turns the cue identity into a "
 "tense bit, the noun identity into a number bit and reads the four answer scores off the class."
)

def onehot(vals, K):
    return np.eye(K)[np.asarray(vals).astype(int).reshape(-1)]

def model(x, iv):
    x = np.asarray(x)
    cue = x[:, 0].astype(int)
    noun = x[:, 1].astype(int)

    cue0 = iv("cue0", onehot(cue, 12))      # which cue word is at position 1
    noun0 = iv("noun0", onehot(noun, 60))   # which noun is at position 4

    cue3 = iv("cue3", np.asarray(cue0))
    noun3 = iv("noun3", np.asarray(noun0))
    cue6 = iv("cue6", np.asarray(cue3))
    noun6 = iv("noun6", np.asarray(noun3))
    cue9 = iv("cue9", np.asarray(cue6))
    noun9 = iv("noun9", np.asarray(noun6))

    c = np.asarray(cue9).argmax(1)
    n = np.asarray(noun9).argmax(1)
    cls = iv("anscls12", onehot(2 * (c >= 6).astype(int) + (n >= 30).astype(int), 4))
    return 8.0 * np.asarray(cls) - 4.0

ALIGN = {
    "cue0":  ["resid0@1"],
    "noun0": ["resid0@4"],
    "cue3":  ["resid3@1"],
    "noun3": ["resid3@4"],
    "cue6":  ["resid6@1"],
    "noun6": ["resid6@4"],
    "cue9":  ["resid9@1"],
    "noun9": ["resid9@4"],
    "anscls12": ["resid12@4"],
}
