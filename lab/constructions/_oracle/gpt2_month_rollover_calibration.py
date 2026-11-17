DESCRIPTION = "Hand-made calibration (not the constructor's): identities placed where flow() shows each input lives; decision from a measured (month1, month2) table; the final stage holds the identity of the year token to output."
TAB = np.array([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
YEARS = np.arange(1979, 2001)
def oh(v, k): return np.eye(k)[np.asarray(v).astype(int)]
def model(x, iv):
    m1 = iv("m1", oh(x[:, 0], 12)); y = iv("year", oh(x[:, 2] - 1980, 20)); m2 = iv("m2", oh(x[:, 1], 12))
    m1b = iv("m1_mid", oh(m1.argmax(1), 12)); yb = iv("year_mid", oh(y.argmax(1), 20)); m2b = iv("m2_mid", oh(m2.argmax(1), 12))
    m1c = iv("m1_late", oh(m1b.argmax(1), 12)); yc = iv("year_late", oh(yb.argmax(1), 20)); m2c = iv("m2_late", oh(m2b.argmax(1), 12))
    nxt = TAB[m1c.argmax(1), m2c.argmax(1)]
    ans = iv("answer", oh(yc.argmax(1) + 1980 + nxt - 1979, 22))
    yr = yc.argmax(1) + 1980; ay = ans.argmax(1) + 1979
    cand = np.stack([yr, yr + 1, yr - 1], 1)          # the prompt's three answer tokens
    return -2.0 * np.abs(ay[:, None] - cand)          # year tokens are read out by numeric proximity to the year held
ALIGN = {"m1": ["resid0@m1"], "year": ["resid0@year"], "m2": ["resid0@m2"],
         "m1_mid": ["resid4@m1", "resid4@year", "resid4@m2"], "year_mid": ["resid4@year"], "m2_mid": ["resid4@m2"],
         "m1_late": ["resid8@year", "resid8@m2"], "year_late": ["resid8@year", "resid8@m2"], "m2_late": ["resid8@m2"],
         "answer": ["resid12@m2"]}
