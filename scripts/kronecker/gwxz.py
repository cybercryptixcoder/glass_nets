"""Known theorem used as a vanishing test: the Kronecker square of a two-row rectangle
(Garsia, Wallach, Xin, Zabrocki 2012): s_(k,k) * s_(k,k) = sum of s_nu over nu with at most 4 parts whose
4 parts (padded with zeros) are all even or all odd. Conjugating one argument conjugates nu.
gwxz_zero(A, B, t) is True when the theorem forces g = 0 (in any slot)."""
from skew import conj

def _all_same_parity4(p):
    if len(p) > 4: return False
    q = list(p) + [0] * (4 - len(p))
    return all(x % 2 == 0 for x in q) or all(x % 2 == 1 for x in q)

def _two_row_rect(p):
    return len(p) == 2 and p[0] == p[1]

def gwxz_zero(A, B, t):
    for (x, y, w) in ((A, B, t), (B, t, A), (t, A, B)):
        for (u, v, s) in ((x, y, w), (conj(x), conj(y), w), (conj(x), y, conj(w)), (x, conj(y), conj(w))):
            if _two_row_rect(u) and u == v:
                return not _all_same_parity4(s)
    return False
