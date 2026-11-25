"""Berele-Regev hook theorem (Adv. Math. 64, 1987; also Dvir, J. Algebra 154, 1993) as a zero test.
H(k,l) = partitions p with p_{k+1} <= l. If A is in H(k1,l1), B in H(k2,l2) and g(A,B,t) > 0, then t is in
H(k1k2 + l1l2, k1l2 + k2l1). Taking for each k the smallest l (l = A_{k+1}) gives every instance."""

def _ok_target(A, B, t):
    for k1 in range(len(A) + 1):
        l1 = A[k1] if k1 < len(A) else 0
        for k2 in range(len(B) + 1):
            l2 = B[k2] if k2 < len(B) else 0
            K, L = k1 * k2 + l1 * l2, k1 * l2 + k2 * l1
            if len(t) > K and t[K] > L:
                return False
    return True

def fathook_zero(A, B, t):
    """True if the hook theorem forces g(A, B, t) = 0 (some target slot fails)."""
    return not (_ok_target(A, B, t) and _ok_target(B, t, A) and _ok_target(t, A, B))
