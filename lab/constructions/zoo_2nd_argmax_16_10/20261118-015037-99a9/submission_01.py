
import numpy as np, math
DESCRIPTION = ("First stage only. Each hidden unit of step 1 is relu(Wx_j * x_1): five units have large |Wx| "
               "(sign detectors with magnitude), the rest tiny. h1 is exactly that vector, so it is reproduced "
               "with the 16 input weights; later stages are not yet explained.")
WX = [0.06919536739587784, -10.564790725708008, 0.014607131481170654, -0.12265294790267944,
      10.156975746154785, -0.28130465745925903, -10.995721817016602, -13.16869068145752,
      -12.312334060668945, -0.22756263613700867, 0.05984975025057793, 0.14799705147743225,
      0.29800716042518616, -0.04701320827007294, -1.3206626176834106, -0.1191084161400795]
def model(x, iv):
    x = np.asarray(x, dtype=float)
    h1 = iv("h1", np.maximum(x[:, [0]] * np.array(WX)[None, :], 0.0))
    # placeholders: later stages not explained yet
    s = h1.sum(axis=1, keepdims=True)
    out = np.zeros((x.shape[0], 10)) + s * 0.0
    return out
ALIGN = {"h1": ["h1"]}
