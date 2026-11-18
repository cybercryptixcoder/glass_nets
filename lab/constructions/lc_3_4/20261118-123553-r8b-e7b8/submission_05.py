import numpy as np
DESCRIPTION = "constant zero output (calibration probe)."
def model(x, iv):
    return np.zeros((len(x), 4))
ALIGN = {}
