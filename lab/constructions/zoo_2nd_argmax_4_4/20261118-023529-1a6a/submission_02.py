
import numpy as np
DESCRIPTION = ("With no biases the map is homogeneous of degree one, so only the direction of x matters. "
               "In units rescaled by positive factors (a gauge choice that leaves the function unchanged) all four units "
               "share one input direction: reading the value x_t adds x_t to units 0 and 3 and subtracts it from units 1 and 2. "
               "Each step adds that to the running state mixed through a fixed circulation matrix - unit 0 drives unit 1, "
               "unit 1 drives units 0 and 3, unit 2 drives units 0 and 3 with the opposite sign (it undoes unit 1), unit 3 drives unit 2 - "
               "and rectifies. The same step is applied four times. The four scores are linear in the final state; only their order matters, "
               "so each class vector is written scaled by a positive factor and with a common vector subtracted, which leaves the argmax unchanged.")

IN = np.array([1.0, -1.0, -1.0, 1.0])
M0 = np.array([-0.21374, 1.27334, -0.06990, 2.31898])
M1 = np.array([5.41727, 1.32034, -0.53387, 14.22469])
M2 = np.array([-4.42718, -1.28935, 0.53066, -14.46224])
M3 = np.array([-0.01322, 0.08467, 0.48256, -0.75254])
MH = np.stack([M0, M1, M2, M3])
R0 = np.array([0.0, -1.54645, -2.52119, -0.41834])
R1 = np.array([1.0, 0.0, -0.63444, 1.0])
R2 = np.array([-1.19060, 1.0, 0.0, -1.42154])
R3 = np.array([-0.18392, 0.20728, 1.0, 0.0])
RO = np.stack([R0, R1, R2, R3])

def model(x, iv):
    h = iv("h1", np.maximum(x[:, 0:1] * IN, 0))
    h = iv("h2", np.maximum(x[:, 1:2] * IN + h @ MH, 0))
    h = iv("h3", np.maximum(x[:, 2:3] * IN + h @ MH, 0))
    h = iv("h4", np.maximum(x[:, 3:4] * IN + h @ MH, 0))
    return h @ RO

ALIGN = {"h1": ["h1"], "h2": ["h2"], "h3": ["h3"], "h4": ["h4"]}
