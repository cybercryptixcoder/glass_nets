"""Fresh N-body verification set (never seen by the law, the network, or any fit).
make: draw source systems from the held-out random population (excluding every
system used in this study), keep masses, semimajor axes, eccentricities and
inclinations, and draw NEW random angles (pericentre, node, mean longitude), so each
is a new dynamical system. Saves a Simulationarchive and an element table.
run W w LO HI TMAX: integrate items LO..HI-1 with i % W == w using SPOCK's own
N-body setup (WHFast, Hill-sphere crossing = instability), up to TMAX inner orbits."""
import os, sys, numpy as np, rebound, warnings
warnings.filterwarnings("ignore")
P = os.path.expanduser(os.environ.get("PDATA", "~/pdata"))
def elements(sim):
    ps = sim.particles; M = ps[0].m; out = []
    for p in ps[1:4]:
        o = p.orbit(primary=ps[0])
        out.append([p.m / M, o.a, o.e, o.inc, o.pomega, o.l, o.P, o.Omega])
    return np.array(out)
def make(n=800, seed=11):
    z = np.load(f"{P}/random_ic.npz"); lt = np.log10(z["t"])
    used = set(np.loadtxt(f"{P}/nn_random_batch.csv", delimiter=",", ndmin=2)[:, 0].astype(int))
    pool = np.array([i for i in np.where((lt >= 4) | z["stable"].astype(bool))[0] if i not in used])
    rng = np.random.default_rng(seed); src = np.sort(rng.choice(pool, n, replace=False))
    sa = rebound.Simulationarchive(f"{P}/spock2/dataset/random/clean_initial_conditions.bin")
    fn = f"{P}/fresh.bin"
    if os.path.exists(fn): os.remove(fn)
    E = []
    for i in src:
        s = sa[int(i)]; st = s.particles[0]
        new = rebound.Simulation(); new.G = s.G; new.add(m=st.m)
        for p in s.particles[1:4]:
            o = p.orbit(primary=st)
            new.add(m=p.m, a=o.a, e=o.e, inc=o.inc, pomega=rng.uniform(0, 2 * np.pi),
                    Omega=rng.uniform(0, 2 * np.pi), l=rng.uniform(0, 2 * np.pi), primary=new.particles[0])
        new.move_to_com(); new.save_to_file(fn); E.append(elements(new))
    np.savez(f"{P}/fresh_ic.npz", E=np.array(E), src=src)
    print("made", n, "fresh systems from", len(pool), "unused population members")
def run(W, w, lo, hi, tmax):
    from spock import NbodyRegressor
    sa = rebound.Simulationarchive(f"{P}/fresh.bin"); nb = NbodyRegressor()
    out = f"{P}/fresh_nbody_w{w}_{lo}.csv"
    done = {int(l.split(",")[0]) for l in open(out)} if os.path.exists(out) else set()
    for i in range(lo, hi):
        if i % W != w or i in done: continue
        s = sa[i]; P1 = min(p.P for p in s.particles[1:4])
        r = nb.predict_instability_time(s, tmax=tmax * P1, n_jobs=1)
        t = float(r[0]); stable = t >= 0.999 * tmax * P1          # returns (time, ...); tmax means survived
        with open(out, "a") as f: f.write(f"{i},{t / P1:.6g},{int(stable)}\n")
if __name__ == "__main__":
    if sys.argv[1] == "make": make()
    else: run(*map(int, sys.argv[2:6]), float(sys.argv[6]))
