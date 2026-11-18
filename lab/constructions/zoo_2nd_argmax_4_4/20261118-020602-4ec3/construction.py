import numpy as np

DESCRIPTION = (
    "Shift register of the four numbers plus a rank readout. The four hidden units are one "
    "position slot each: at step t the state holds the t numbers read so far, the new number "
    "driven in by W_x and moved into its own slot by the recurrent shift (h_1 = x1 in slot 0, "
    "h_2 = (x1, x2), h_3 = (x1, x2, x3), h_4 = all four numbers; slots not yet filled are 0). "
    "No comparison happens inside the recurrence. The task, 'position of the second-largest "
    "number', is done by the readout: for each position i count how many of the four numbers "
    "are larger than it (its rank r_i), and give position i the class logit "
    "relu(2 - 1.5*|r_i - 1|). That is 2 for the position with exactly one larger number (the "
    "second largest), 0.5 for a position with no or two larger numbers, and 0 for a position "
    "with three larger numbers, so the largest logit sits on the position of the second "
    "largest number."
)


def model(x, iv):
    b = x.shape[0]

    def col(t):
        return x[:, t:t + 1]

    # state = running slots for the positions read so far (a shift register)
    v1 = iv("val1", col(0))
    v2 = iv("val2", np.concatenate([v1, col(1)], axis=1))
    v3 = iv("val3", np.concatenate([v2, col(2)], axis=1))
    v4 = iv("val4", np.concatenate([v3, col(3)], axis=1))

    # readout: one logit per position, peaked at rank 1 (exactly one larger number)
    out = np.zeros((b, 4))
    for i in range(4):
        rank = np.zeros(b)
        for j in range(4):
            if j != i:
                rank = rank + (v4[:, j] > v4[:, i]).astype(float)
        out[:, i] = np.maximum(2.0 - 1.5 * np.abs(rank - 1.0), 0.0)
    return out


ALIGN = {"val1": ["h1"], "val2": ["h2"], "val3": ["h3"], "val4": ["h4"]}
