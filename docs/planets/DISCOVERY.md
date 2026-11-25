# A simple physical formula for when packed planetary systems fall apart

## The problem

Many stars host several planets packed close together. Gravity between the planets slowly
disturbs their orbits, and eventually many such systems go unstable: orbits cross, planets
collide or get thrown out. Knowing *when* this happens tells astronomers which observed
systems can survive for billions of years, which configurations are allowed, and how
planetary systems evolve.

The only exact way to get the answer is to simulate the system orbit by orbit, which can take
billions of orbits and days of computing per system. So researchers built shortcuts:

- **Hand-derived formulas** (for example Petit et al. 2020; Lammers, Hadden and Murray 2024).
  Physically meaningful, but built for idealised systems. On realistic systems they are
  far off: errors of 2 to 3 orders of magnitude in the predicted time.
- **A neural network** (Cranmer et al. 2021, the standard tool in the SPOCK package). It runs
  a short simulation of 10,000 orbits and predicts the instability time from it. It is
  accurate, but it is a black box: it does not say *why* a system is long- or short-lived.
- **Equations distilled from that network** (Alford et al. 2026). Readable, but less accurate
  than the network and not physically meaningful (their terms have no consistent units).

The open question: **can a short formula, built only from quantities a physicist would
recognise, predict instability times as well as the neural network?**

## What we found

Yes. For systems of three closely packed planets, this formula does it:

    log10(t / P) = 5.36 + 4.94 S - 0.53 log10(C) - 1.27 log10(F) - 0.54 X - 0.96 log10(D)

- t / P: the instability time, counted in orbits of the innermost planet.
- S: how widely the planets are spaced, measured relative to their masses (the scaling
  that theory predicts for three-planet interactions).
- C: how chaotic the system is, read from the same short 10,000-orbit simulation the network
  uses (the standard chaos indicator MEGNO, minus its value for regular motion).
- F: how strongly the closest pair's orbits are being stretched during that short
  simulation, compared with the stretch needed for the orbits to cross.
- X: how close the starting orbits already are to crossing (0 = far, 1 = touching).
- D: the spacing of the closest pair of planets.

Seven numbers were fitted in total (the six above plus one fixed value, 10^4.34 orbits, for
systems that already break up during the short simulation). Everything else is physics.

## How well it works

We fitted the formula on one published set of simulated systems, then tested it on
systems it had never seen, against the neural network on exactly the same systems.
Errors are in orders of magnitude of the predicted time (lower is better).

| Test | Formula | Neural network |
|---|---|---|
| 731 unseen systems that go unstable | 1.10 | 1.16 |
| All 1,736 unseen systems, including long-lived ones | 0.88 | 0.87 |
| 154 brand-new systems we simulated ourselves, which went unstable | 0.97 | 1.32 |
| All 659 brand-new systems, including the ones that survived | 0.49 | 0.64 |

On the published test set the two are statistically tied. On the brand-new systems,
simulated after the formula was fixed, the formula is clearly ahead: the network more often
calls a strongly chaotic system long-lived when it actually breaks up early.

For scale: no method can do better than about 0.47 orders of magnitude, because these
systems are chaotic and their exact fate is genuinely unpredictable at that level.

## What it tells us physically

- **Chaos is the main clock.** Removing the chaos term hurts the formula most. The time to
  instability grows with the system's chaotic (Lyapunov) timescale to the power of about 0.5,
  and that exponent stays the same whether we fit on one fifth of the data or another.
- **Spacing matters as roughly the fourth power** of the planets' separation measured in
  units set by their masses.
- **Orbits being pushed toward crossing shortens life**, and starting close to crossing
  shortens it further.

In short, most of what the neural network knows comes down to two measurable things: how
chaotic the system is, and how hard its orbits are being pushed toward crossing.

## Checks we ran

- If the chaos and orbit-stretching numbers are shuffled between systems, the formula
  fails badly, so its accuracy comes from those physical measurements, not from fitting.
- Every term in the formula is needed: removing any one makes predictions worse.
- The published 2024 formula scores 2.00 on the same unseen systems; a constant guess 1.46.

## What we do not have yet

- **It is not a formula from the starting orbits alone.** It needs the same short simulation
  the network uses (a fraction of a second per system). Formulas using only the starting
  orbits reached 1.23, clearly short of the network.
- **On the main test it ties the network rather than beating it.** If the network's tendency
  to overestimate is corrected, it comes out slightly ahead (by 0.03).
- **Tested on three-planet systems** drawn from one family of configurations. Systems with
  more planets, or very different setups, are untested.
- **Not yet reviewed by an outside astronomer.**

## Next steps

Test on larger and different kinds of systems (equal masses, even spacing, more than three
planets), where the neural network is known to struggle, and have the result checked by
specialists in planetary dynamics.

---

Data: SPOCK 2.0 datasets (Zenodo 14327471). Code, raw data and full numbers: this
repository, `docs/planets/` and `results/planets/`.
