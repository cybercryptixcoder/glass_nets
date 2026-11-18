
import numpy as np, math
DESCRIPTION = "Diagnostic forward pass (weights as literals) that names only the sites/coordinates the network actually uses downstream: embedding at all positions, the residual after layer 0 at all positions, the last-position row of layer 1's attention pattern, what layer 1 adds at the last position, and the final residual. Layer-0 pattern/out are computed but not named (they only matter through resid0.mid)."
WE=np.array([1.022,-0.074,0.678,1.424, 1.896,-0.640,0.301,1.396, 1.144,0.999,-0.555,1.769, 1.846,-1.087,2.882,-0.017, 1.094,-0.200,-2.026,-0.522]).reshape(5,4)
WP=np.array([-0.474,-1.571,1.908,-1.673, 0.594,2.255,0.026,-1.154, 0.170,-0.525,2.546,1.957, -0.067,2.608,-1.298,-0.453, -1.592,2.061,-1.151,0.933]).reshape(5,4)
WU=np.array([0.112,-0.418,0.006,0.689,-0.044, -0.139,-0.158,-1.465,0.198,-0.395, 0.361,-0.363,-0.527,-0.027,0.114, -0.030,-0.102,0.607,-0.315,0.140]).reshape(4,5)
Q0=np.array([1.143,-0.443,2.807,2.674, 1.021,-0.644,-2.669,3.083, -1.237,-2.000,3.200,-2.109, 5.130,0.987,1.668,-1.828]).reshape(4,4)
K0=np.array([-1.717,1.759,-2.688,-1.776, 0.035,1.285,-2.337,-2.789, 0.334,-0.794,-0.502,0.320, -2.290,-4.662,-3.212,0.550]).reshape(4,4)
V0=np.array([-0.803,0.007,-0.159,0.011, 0.321,-0.358,0.300,0.557, -0.051,-0.326,0.862,-0.003, -0.570,-0.123,-0.912,-0.799]).reshape(4,4)
O0=np.array([-1.025,0.200,-0.101,-0.044, 0.029,-0.058,-0.682,-0.024, 0.343,-0.957,-0.289,-0.709, -0.312,-1.447,0.871,1.083]).reshape(4,4)
Q1=np.array([-0.092,-0.279,-3.578,-0.107, -0.598,-0.172,-0.543,0.284, -0.148,0.063,-0.609,-2.310, 0.608,0.467,0.444,-0.288]).reshape(4,4)
K1=np.array([0.925,-1.147,0.647,0.263, 0.382,0.480,0.799,-0.897, -0.289,0.135,0.164,0.008, -1.958,-0.039,0.134,-0.758]).reshape(4,4)
V1=np.array([-0.342,-1.565,-0.404,-1.463, 1.171,0.514,-2.199,0.058, -1.312,-0.385,-2.518,-0.127, 2.046,0.602,0.028,0.304]).reshape(4,4)
O1=np.array([-0.086,0.673,-2.525,1.424, -0.804,-0.093,-0.203,-1.561, 0.232,-5.500,0.747,0.134, 0.093,-0.007,1.658,1.507]).reshape(4,4)

def sm(s):
    s=s-s.max(-1,keepdims=True); e=np.exp(s); return e/e.sum(-1,keepdims=True)

def model(x, iv):
    B=x.shape[0]
    E=WE[x]+WP[None,:,:]
    em=np.stack([iv("e%d"%p, E[:,p,:].copy()) for p in range(5)],1)
    A0=sm((em@Q0)@np.swapaxes(em@K0,1,2)*0.5)
    a0=(A0@(em@V0))@O0
    r0=em+a0
    r0=np.stack([iv("r0_%d"%p, r0[:,p,:].copy()) for p in range(5)],1)
    A1=sm((r0@Q1)@np.swapaxes(r0@K1,1,2)*0.5)
    A1[:,4,:]=iv("a1p4", A1[:,4,:].copy())
    a1=(A1@(r0@V1))@O1
    a1[:,4,:]=iv("a1o4", a1[:,4,:].copy())
    r1=r0+a1
    r1=np.stack([iv("r1_%d"%p, r1[:,p,:].copy()) for p in range(5)],1)
    return r1[:,4,:]@WU

ALIGN={"e0":["embed"],"e1":["embed"],"e2":["embed"],"e3":["embed"],"e4":["embed"],
 "r0_0":["resid0.mid"],"r0_1":["resid0.mid"],"r0_2":["resid0.mid"],"r0_3":["resid0.mid"],"r0_4":["resid0.mid"],
 "a1p4":["attn1.pattern"],"a1o4":["attn1.out"],
 "r1_0":["resid1.mid"],"r1_1":["resid1.mid"],"r1_2":["resid1.mid"],"r1_3":["resid1.mid"],"r1_4":["resid1.mid"]}
