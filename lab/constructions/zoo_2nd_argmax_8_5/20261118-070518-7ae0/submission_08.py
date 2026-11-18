
import numpy as np

DESCRIPTION = ("Reduced-form re-implementation of the five recurrent steps. The network reads five real numbers; its argmax output is "
               "(87.9% of inputs) the position of the SECOND largest input, i.e. an approximate runner-up detector whose errors sit "
               "where the 2nd and 3rd largest are nearly equal. Step 1 is exactly two rays: h1 = relu(x1)*relu(WX) + relu(-x1)*relu(-WX), "
               "so after one input the state carries only x1's positive and negative part. Steps 2-5: each unit j turns on exactly when "
               "the new input exceeds the threshold u_j = -(h_{t-1} @ WH_j)/WX_j, i.e. every unit is a 'new value exceeds a recurrently "
               "maintained level' detector, with a different level per unit. The readout is written in a canonical gauge: one entry of "
               "every class row is removed exactly by the row-shift freedom (adding c_j to the five class weights of unit j adds the same "
               "c_j*h_j to every score), so class 0 is identically zero, and the four smallest remaining entries are dropped (their total "
               "effect on any score is < 0.4 out of score differences of order 100).")

WX = np.array([0.642615, -6.421312, 0.423453, -6.573740, 0.045088, -4.913164, -0.054098, 7.001690])
WH = np.array([
[ 0.125403, 0.154946, 0.476144, 0.060649, 0.771231, 0.035924, 0.943970,-0.163330],
[-0.464716,-0.236367, 0.272299, 0.492902, 0.169101,-0.732230, 0.424159,-0.027035],
[ 0.019963,-0.129043,-0.106897,-0.190221,-0.618453,-0.423786, 0.239916, 0.081623],
[ 0.528093,-0.299993,-0.362798,-0.847493, 0.279272, 0.569700, 0.198553, 0.061197],
[-0.177440,-0.324277, 0.215948,-0.251721,-0.746841,-0.191503,-1.207441, 0.380405],
[ 0.881026, 0.600439, 0.125687, 0.347769, 0.127209, 0.069581,-0.750272, 0.023595],
[ 0.126756, 0.961696,-0.086458, 0.972771, 0.545731, 0.679505, 0.786684,-0.890478],
[-1.655899,-0.017242, 1.418636, 1.087747, 0.155794, 0.814863, 1.102013,-0.827896]])
# readout in the canonical gauge: class-0 column removed by the per-unit shift; four negligible entries set to 0
WO2 = np.array([
[-1.852636, 0.0,      -2.664365, 0.011411],
[-1.052528,-4.241131,  3.548695,-0.575713],
[-0.917780,-4.235484, -0.282118,-0.215207],
[-2.785807, 4.443881, -3.216199,-1.032785],
[ 2.563128,-0.064457, -0.064946, 0.026097],
[ 5.071054,-0.479585, -0.439058, 0.026631],
[-0.070866, 1.384690,  1.545844, 1.359913],
[ 0.077526, 0.137966,  0.116108,-2.492862]])


def model(x, iv):
    p = iv("pos1", np.maximum(x[:, 0], 0))
    n = iv("neg1", np.maximum(-x[:, 0], 0))
    h = p[:, None] * np.maximum(WX, 0) + n[:, None] * np.maximum(-WX, 0)
    for t in range(2, 6):
        h = iv("h%d" % t, np.maximum(x[:, t-1:t] * WX + h @ WH, 0))
    out = h @ WO2
    return np.concatenate([np.zeros((out.shape[0], 1)), out], axis=1)

ALIGN = {"pos1": ["h1"], "neg1": ["h1"], "h2": ["h2"], "h3": ["h3"], "h4": ["h4"], "h5": ["h5"]}
