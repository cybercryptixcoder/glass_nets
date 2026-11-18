
import numpy as np
DESCRIPTION = ("No biases, so the map is homogeneous of degree one and only the direction of the sequence matters. "
               "The state has four units. In units rescaled by positive factors (a gauge choice that leaves the whole map unchanged) "
               "all four units share a single input direction: reading the value x_t adds x_t to units zero and three and subtracts it from "
               "units one and two. Every step adds that to the previous state passed through one fixed mixing matrix and rectifies. "
               "The mixing is a circulation: unit one mainly drives unit three while unit two cancels unit one with the opposite sign, "
               "unit zero drives unit one, and unit three drives unit two. The same step is repeated for the four values in turn. "
               "The decision comes from four scores that are linear in the final state; since only the order of the scores matters, "
               "class zero's score is subtracted from the other three and the three remaining difference directions are written out, "
               "scaled by one common positive factor.")
IN = np.array([1, -1, -1, 1])
M0 = np.array([-0.213737, 1.273340, -0.069905, 2.318979])
M1 = np.array([5.417269, 1.320339, -0.533874, 14.224694])
M2 = np.array([-4.427177, -1.289346, 0.530658, -14.462238])
M3 = np.array([-0.013221, 0.084671, 0.482565, -0.752541])
MH = np.stack([M0, M1, M2, M3])
D1 = np.array([-0.204678, -0.728751, 1, 0.161464])
D2 = np.array([-0.852806, -0.943350, 0.867646, 0.472285])
D3 = np.array([-0.221551, -0.199152, 0.114800, 0.134029])

def model(x, iv):
    h = iv("h1", np.maximum(x[:, 0:1] * IN, 0))
    h = iv("h2", np.maximum(x[:, 1:2] * IN + h @ MH, 0))
    h = iv("h3", np.maximum(x[:, 2:3] * IN + h @ MH, 0))
    h = iv("h4", np.maximum(x[:, 3:4] * IN + h @ MH, 0))
    return np.stack([np.zeros(len(h)), h @ D1, h @ D2, h @ D3], axis=1)

ALIGN = {"h1": ["h1"], "h2": ["h2"], "h3": ["h3"], "h4": ["h4"]}
