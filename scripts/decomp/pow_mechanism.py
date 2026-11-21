import numpy as np, sys
sys.path.insert(0,'lab'); from decomp import program as pg
nm=sys.argv[1] if len(sys.argv)>1 else "dc_pow_s1"
W=dict(np.load(f"lab/models/{nm}/weights.npz")); p=113; t=np.arange(p); rng=np.random.default_rng(0)
dl=np.full(p,-1); x=1
for e in range(p-1): dl[x]=e; x=x*3%p
ex=np.array([pow(3,int(e),p) for e in range(112)])
I=pg.internals(W,pg.all_pairs(p)); x1=I["x1"][:,-1].reshape(p,p,-1); net=I["logits"].argmax(-1).reshape(p,p)
F=x1.mean(1); G=x1.mean(0)
def fit_parts(X,g,ok):
    # X[ok] ~ c + f7[g%7] + f16[g%16]; return f7 (7xd), f16 (16xd)
    Xs=X[ok]; gs=g[ok]; n=len(Xs)
    A=np.zeros((n,23)); A[np.arange(n),gs%7]=1; A[np.arange(n),7+gs%16]=1
    B,*_=np.linalg.lstsq(np.column_stack([np.ones(n),A]),Xs,rcond=None)
    return B[1:8],B[8:]
f7a,f16a=fit_parts(F,dl,dl>=0); f7b,f16b=fit_parts(G,t,np.ones(p,bool))
def crt(r7,r16):
    for z in range(112):
        if z%7==r7 and z%16==r16: return z
N=3000
A=rng.integers(1,p,N); A2=rng.integers(1,p,N); B=rng.integers(0,p,N); B2=rng.integers(0,p,N)
def run(delta_fn, want_fn, label, ctrl_dim=None):
    toks=np.stack([A,B,np.full(N,p)],1)
    D=np.array([delta_fn(i) for i in range(N)])
    if ctrl_dim:   # control: same-norm delta in a random subspace of that dimension
        Q=np.linalg.qr(rng.standard_normal((D.shape[1],ctrl_dim)))[0]; R=rng.standard_normal((N,ctrl_dim))@Q.T
        D=R/np.linalg.norm(R,axis=1,keepdims=True)*np.linalg.norm(D,axis=1,keepdims=True)
    out=pg.internals(W,toks,patch1=lambda x1: x1+D)["logits"].argmax(-1)
    want=np.array([want_fn(i) for i in range(N)])
    print(f"{label:55s} success {np.mean(out==want):.3f}   (unpatched answer kept: {np.mean(out==net[A,B]):.3f})")
xa=dl[A]; xa2=dl[A2]
# swap only the mod-7 features of a: predicted dlog(c) = CRT( b*x' mod 7 , b*x mod 16 )
run(lambda i: f7a[xa2[i]%7]-f7a[xa[i]%7], lambda i: ex[crt((B[i]*xa2[i])%7,(B[i]*xa[i])%16)], "a: swap mod-7 features (a -> a')")
run(lambda i: f7a[xa2[i]%7]-f7a[xa[i]%7], lambda i: ex[crt((B[i]*xa2[i])%7,(B[i]*xa[i])%16)], "   control: random subspace, same size", ctrl_dim=6)
run(lambda i: f16a[xa2[i]%16]-f16a[xa[i]%16], lambda i: ex[crt((B[i]*xa[i])%7,(B[i]*xa2[i])%16)], "a: swap mod-16 features (a -> a')")
run(lambda i: f7b[B2[i]%7]-f7b[B[i]%7], lambda i: ex[crt((B2[i]*xa[i])%7,(B[i]*xa[i])%16)], "b: swap mod-7 features (b -> b')")
run(lambda i: f16b[B2[i]%16]-f16b[B[i]%16], lambda i: ex[crt((B[i]*xa[i])%7,(B2[i]*xa[i])%16)], "b: swap mod-16 features (b -> b')")
run(lambda i: f16b[B2[i]%16]-f16b[B[i]%16], lambda i: ex[crt((B[i]*xa[i])%7,(B2[i]*xa[i])%16)], "   control: random subspace, same size", ctrl_dim=15)

# ---- second level: inside the mod-7 features, the multiplicative characters of Z_7* (order 2 = Legendre, order 3 = cubic)
d7=np.full(7,-1); y=1
for e in range(6): d7[y]=e; y=y*3%7
g7=np.array([pow(3,e,7) for e in range(6)])
def char_parts(f7):
    # f7[r] for units r: f7[r] = mu + sum_{j=1..5} chi_j(r) beta_j  (chi_j(r)=exp(2pi i j d7[r]/6)); return function of (e2, e3)
    U=[r for r in range(1,7)]; e=d7[U]
    Bm=np.array([np.exp(2j*np.pi*j*e/6) for j in range(6)]).T   # 6x6
    beta=np.linalg.solve(Bm,f7[U].astype(complex))              # 6 x d
    def feat(e_ord2, e_ord3):
        # value with order-2 part from exponent e_ord2 (mod 2) and order-3 part from e_ord3 (mod 3); chi_j has order 6/gcd(j,6)
        v=beta[0].copy()
        for j in range(1,6):
            o=6//np.gcd(j,6)
            if o==2: ee=e_ord2
            elif o==3: ee=e_ord3
            else: ee=None
            if ee is None:   # order-6 characters: need both parts; combine e from CRT(e2 mod 2, e3 mod 3)
                ee=[z for z in range(6) if z%2==e_ord2%2 and z%3==e_ord3%3][0]
            v=v+beta[j]*np.exp(2j*np.pi*j*ee/6)
        return v.real
    return feat
fb=char_parts(f7b)
okm=(B%7!=0)&(B2%7!=0)&(xa%7!=0)
def want_legendre(i):
    e=d7[B[i]%7]; e2=d7[B2[i]%7]
    enew=[z for z in range(6) if z%2==e2%2 and z%3==e%3][0]      # b's second-level log with its parity taken from b'
    y7=(g7[(d7[xa[i]%7]+enew)%6])                                  # x7 * b7_new
    return ex[crt(y7%7,(B[i]*xa[i])%16)]
def delta_legendre(i):
    if not okm[i]: return np.zeros(F.shape[1])
    e=d7[B[i]%7]; e2=d7[B2[i]%7]
    return fb(e2%2,e%3)-fb(e%2,e%3)
toks=np.stack([A,B,np.full(N,p)],1); D=np.array([delta_legendre(i) for i in range(N)])
out=pg.internals(W,toks,patch1=lambda x1: x1+D)["logits"].argmax(-1); want=np.array([want_legendre(i) for i in range(N)])
sel=okm&(d7[B%7]%2!=d7[B2%7]%2)
print(f"b: swap ONLY the order-2 (Legendre) character of b mod 7: success {np.mean(out[sel]==want[sel]):.3f} on {sel.sum()} cases (unpatched kept {np.mean(out[sel]==net[A,B][sel]):.3f})")
def want_cubic(i):
    e=d7[B[i]%7]; e2=d7[B2[i]%7]; enew=[z for z in range(6) if z%2==e%2 and z%3==e2%3][0]
    return ex[crt(g7[(d7[xa[i]%7]+enew)%6]%7,(B[i]*xa[i])%16)]
D=np.array([ (fb(d7[B[i]%7]%2,d7[B2[i]%7]%3)-fb(d7[B[i]%7]%2,d7[B[i]%7]%3)) if okm[i] else np.zeros(F.shape[1]) for i in range(N)])
out=pg.internals(W,toks,patch1=lambda x1: x1+D)["logits"].argmax(-1); want=np.array([want_cubic(i) for i in range(N)])
sel=okm&(d7[B%7]%3!=d7[B2%7]%3)
print(f"b: swap ONLY the order-3 (cubic) character of b mod 7:    success {np.mean(out[sel]==want[sel]):.3f} on {sel.sum()} cases (unpatched kept {np.mean(out[sel]==net[A,B][sel]):.3f})")
