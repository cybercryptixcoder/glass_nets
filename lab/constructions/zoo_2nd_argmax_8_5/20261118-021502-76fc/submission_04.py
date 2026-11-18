
import numpy as np
DESCRIPTION = ("A growth-dominated gated amplifier chain. h1 holds only two numbers: p=x1^+ fed to units {7,5,0,1} and n=x1^- fed to units {2,4,6} "
 "(7.00, 0.64, 0.42, 0.045) and (6.57, 4.91, 6.42). From step 2 on, each unit is relu(x_t*w_u + a linear combination of the previous state); "
 "unit 7 is a positive-feedback accumulator (self-gain 1.09 plus 0.96 of unit3 and 0.79 of unit5) so the state grows like 1.09^t; units 1 and 6 are "
 "mostly off. The readout is a fixed linear map of the final state to the 5 class scores; class 0 wins unless the amplified positive part becomes large.")
WX=[0.4235,0.0451,-6.5737,-0.0541,-4.9132,0.6426,-6.4213,7.0017]
WH=[[-0.8475,-0.8279,-0.7468,-0.8905,0.3478,0.7712,-0.4647,0.2159],
[0.0816,0.0236,0.5697,1.4186,-0.6185,0.1272,-0.1902,0.02],
[0.2723,0.6004,0.4761,0.2793,0.1986,0.5457,0.4929,-0.129],
[-0.0865,-0.7503,-0.1069,-0.2517,0.0606,0.1549,1.102,-0.3243],
[0.5281,-0.4238,0.1254,0.944,0.2399,-0.3628,-0.1915,0.1257],
[0.6795,-0.027,-0.1774,-0.3,0.8149,-0.2364,-1.2074,0.7867],
[0.1558,0.9728,0.881,0.1268,-0.7322,0.1691,-0.1633,0.0696],
[0.3804,0.0612,0.0359,0.9617,-0.0172,0.4242,-1.6559,1.0877]]
WO=[[-0.3767,0.6789,0.3173,-0.8669,0.3597],[0.5665,-1.1366,4.8036,-0.0691,0.3778],
[-2.4261,-0.8445,0.3365,-0.4678,-1.9484],[2.1603,-3.9047,-3.3868,0.493,-2.253],
[0.7274,-0.9378,0.6334,-0.4028,0.2398],[0.6797,0.8486,-1.3507,-2.8565,0.716],
[0.5178,-1.3102,-0.8712,4.1999,3.8852],[-0.2393,0.3559,-0.4673,-0.6731,-0.7161]]
WH=np.array(WH); WO=np.array(WO); WX=np.array(WX)
DX=np.array([0.0605,0.0064,0.0,0.0,0.0,0.0918,0.0,1.0])*7.0017  # p direction
DN=np.array([0.0,0.0,1.0,0.0,0.7476,0.0,0.9768,0.0])*6.57374  # n direction
def model(x, iv):
    p = iv("p", np.maximum(x[:,0],0.0)); n = iv("n", np.maximum(-x[:,0],0.0))
    h = p[:,None]*DX + n[:,None]*DN
    for t in range(1,5):
        h = np.maximum(x[:,[t]]*WX + h@WH, 0.0)
        h = iv("h%d"% (t+1), h)
    return h@WO
ALIGN={"p":["h1"],"n":["h1"],"h2":["h2"],"h3":["h3"],"h4":["h4"],"h5":["h5"]}
