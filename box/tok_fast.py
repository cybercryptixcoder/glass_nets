"""A fast numeric evaluator of the token-sequence vocabulary (RASP), for the definer's search. Values are arrays over a
batch of sequences padded to a common length: s-ops are float arrays [N, T] (NaN for padding or an undefined value),
selectors boolean arrays [N, T, T] (query x key). Semantics follow bench/accept/vocab (domain `tok`) exactly; every
definition the search picks is re-evaluated with the reference evaluator before it is used.
Tokens are coded as floats that keep their order (numbers as themselves, strings by sorted rank), so comparisons
agree with Python's on the raw values."""
import numpy as np

PREDS = {"EQ": lambda k, q: k == q, "NEQ": lambda k, q: k != q, "LT": lambda k, q: k < q, "LEQ": lambda k, q: k <= q,
         "GT": lambda k, q: k > q, "GEQ": lambda k, q: k >= q, "TRUE": lambda k, q: np.ones_like(k, bool) & np.ones_like(q, bool),
         "FALSE": lambda k, q: np.zeros_like(k, bool) & np.zeros_like(q, bool)}
FN1 = {"id": lambda x: x, "inc": lambda x: x + 1, "dec": lambda x: x - 1, "neg": lambda x: -x,
       "not1": lambda x: (x == 0).astype(float), "abs1": np.abs, "relu1": lambda x: np.maximum(x, 0)}
FN1C = {"eqc": lambda c: (lambda x: (x == c).astype(float)), "ltc": lambda c: (lambda x: (x < c).astype(float)),
        "gtc": lambda c: (lambda x: (x > c).astype(float)), "addc": lambda c: (lambda x: x + c),
        "mulc": lambda c: (lambda x: x * c), "modc": lambda c: (lambda x: np.mod(x, c) if c else x * np.nan),
        "subfrom": lambda c: (lambda x: c - x)}
FN2 = {"plus": lambda a, b: a + b, "minus": lambda a, b: a - b, "times": lambda a, b: a * b,
       "equal": lambda a, b: (a == b).astype(float), "less": lambda a, b: (a < b).astype(float),
       "greater": lambda a, b: (a > b).astype(float), "larger": np.maximum, "smaller": np.minimum,
       "both": lambda a, b: ((a != 0) & (b != 0)).astype(float), "either": lambda a, b: ((a != 0) | (b != 0)).astype(float)}


class Codec:
    """Raw token values <-> order-preserving float codes."""
    def __init__(self, vocab):
        self.vocab = sorted(vocab, key=lambda v: (isinstance(v, str), v))
        self.numeric = all(not isinstance(v, str) for v in self.vocab)
        self.code = {v: (float(v) if self.numeric else float(i)) for i, v in enumerate(self.vocab)}
        self.decode = {c: v for v, c in self.code.items()}

    def encode_batch(self, seqs):
        T = max(len(s) for s in seqs)
        tok = np.full((len(seqs), T), np.nan)
        idx = np.full((len(seqs), T), np.nan)
        for i, s in enumerate(seqs):
            tok[i, :len(s)] = [self.code[v] for v in s]
            idx[i, :len(s)] = np.arange(len(s))
        return tok, idx

    def literal(self, c):
        """A constant as it is written in a definition: a token constant is quoted."""
        if self.numeric:
            return repr(int(c)) if float(c).is_integer() else repr(float(c))
        return '"' + str(self.decode[c]) + '"'


def select(keys, queries, pred):
    k = keys[:, None, :]; q = queries[:, :, None]
    with np.errstate(invalid="ignore"):
        m = PREDS[pred](k, q)
    return m & ~np.isnan(k) & ~np.isnan(q)


def aggregate(sel, vals, valid):
    cnt = sel.sum(-1)
    s = (sel * np.nan_to_num(vals)[:, None, :]).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(cnt > 0, s / np.maximum(cnt, 1), 0.0)
    return np.where(valid, out, np.nan)


def aggregate_cat(sel, vals, valid):
    v = np.where(sel, vals[:, None, :], np.nan)
    has_nan = (sel & np.isnan(vals)[:, None, :]).any(-1)
    with np.errstate(all="ignore"):
        mx, mn = np.nanmax(np.where(sel, vals[:, None, :], -np.inf), -1), np.nanmin(np.where(sel, vals[:, None, :], np.inf), -1)
    ok = (sel.sum(-1) > 0) & (mx == mn) & ~has_nan
    return np.where(valid & ok, mx, np.nan)


def selector_width(sel, valid):
    return np.where(valid, sel.sum(-1).astype(float), np.nan)
