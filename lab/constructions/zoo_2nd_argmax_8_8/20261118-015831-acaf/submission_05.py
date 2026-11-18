
import numpy as np

DESCRIPTION = """What the network computes: a linear reservoir with a clip, nothing more.
The eight inputs are independent standard normals. At each step the new input is injected with a fixed gain
into all eight accumulators and the previous state is linearly mixed (accumulator v feeds accumulator u with a
fixed coupling); the sum is clipped at zero. That clip is the only nonlinearity, hence the whole network is
positively homogeneous - scaling all inputs by c>0 scales the state and every score by c and never changes the
decision - and there are no comparisons, counts or running maxima anywhere in it.
Each stage's eight accumulators are just fixed linear combinations of the inputs seen so far, each gated by
whether its own pre-activation is positive: an accumulator switches off for good stretches of the input space
(unit 2 is off on 23% of inputs at the last step, unit 4 on 3.5%, unit 7 on 3.7%; all earlier clipping has
already been absorbed by step 5). The eight class scores are a fixed linear read-out of the final state, so the
final state carries all the information the decision uses: dropping it loses the decision completely (5.5% of
the network's decisions survive), while the first two stages only carry the sign and size of x1 and x2.
The accumulators are quoted in a fixed per-unit scale (accumulator u is c_u times hidden unit u), so the gains,
couplings and read-out coefficients below are that account's coefficients."""

# input gain of each accumulator
GAIN = np.array([-0.08291914, 0.07540203, -0.70744346, -1.67309377,
                 -0.84946470, 0.00104553, 3.43799591, 0.00397390])

# coupling: MROWS[v] is the 8-vector that accumulator v contributes to all accumulators
MROWS = [
 np.array([0.08166095, 2.41587944, 1.08836214, -36.973387, -2.93756967, 0.11547852, -4.48434138, 1.71916425]),
 np.array([-0.02211025, 0.47973573, -0.19699070, 0.41916711, 0.56198848, -0.15477960, -0.99747022, 0.76364784]),
 np.array([0.45455986, -0.66010869, 0.23509261, 0.67800135, 2.78625922, 0.96187460, 9.12217332, -0.88362201]),
 np.array([0.04418155, 0.43940080, -0.03089723, -0.67147309, -0.03172868, 0.02816418, 0.08397000, 0.01026242]),
 np.array([-0.07002143, -1.18892999, -0.14579706, 7.81471776, 0.27209100, -0.00757497, 1.64188438, -0.22515926]),
 np.array([-0.26542826, -0.17770318, -0.57984489, 37.81118592, -3.73921904, 0.78547078, 14.92044032, 1.90286348]),
 np.array([0.10078942, -0.05159133, 0.05429484, 0.65958595, 0.39547465, 0.03414239, 0.48844647, -0.19862813]),
 np.array([-0.09846967, 1.97696179, 0.01967648, 1.15809374, -0.44553559, 0.09106698, 10.08907755, -0.60039777])]

# read-out: RROWS[u] is the 8-vector of class scores contributed by accumulator u
RROWS = [
 np.array([-2.13020697, -0.88087834, -0.60796291, -0.44829417, -1.39541954, -0.32219462, 5.30557573, -0.92921093]),
 np.array([0.27175808, 0.04595391, -0.02005682, -0.08935038, -0.50237136, -0.08653215, 0.14505349, 0.05051858]),
 np.array([-0.27273462, 0.12181443, -0.23101035, 0.26283833, 0.90723709, 0.28435908, 1.12017290, -7.56480188]),
 np.array([0.03815583, 0.00207931, 0.01408184, -0.01794074, -0.02589216, -0.23338939, 0.11828873, 0.00403201]),
 np.array([0.30608616, -0.00188744, 0.11668859, 0.09527219, 0.26462601, 0.47554600, -1.23594558, 0.14718184]),
 np.array([1.65525507, 1.03596280, 1.72495327, 0.85277432, -0.34431527, 0.75790093, -7.68788919, 1.06695889]),
 np.array([-0.07788596, -0.01779257, -0.08059602, 0.01891106, 0.12732299, 0.01951240, 0.00494560, -0.01723809]),
 np.array([-0.00377377, 0.01127656, 0.06721882, -0.65831631, 0.37224486, 0.02926491, -0.36405176, 0.01296109])]

def model(x, iv):
    s = np.zeros((x.shape[0], 8))
    for t in range(8):
        acc = x[:, t:t + 1] * GAIN[None, :]
        for v in range(8):
            acc = acc + s[:, v:v + 1] * MROWS[v][None, :]
        s = iv("g%d" % (t + 1), np.maximum(acc, 0.0))
    out = s[:, 0:1] * RROWS[0][None, :]
    for u in range(1, 8):
        out = out + s[:, u:u + 1] * RROWS[u][None, :]
    return out

ALIGN = {"g%d" % t: ["h%d" % t] for t in range(1, 9)}
