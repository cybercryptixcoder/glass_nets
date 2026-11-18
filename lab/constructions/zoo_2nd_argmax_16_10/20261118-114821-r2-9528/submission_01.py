import numpy as np
DESCRIPTION = ("ReLU RNN, 16 units, 10 steps, no biases: h_t = relu(x_t*Wx + h_(t-1)@Wh), out = h10@Wo, decision = argmax (class c = position c+1); the whole map is homogeneous of degree 1. "
 "Step 1 is fully explained by a rule: h1 = relu(x1*Wx) = max(x1,0)*A + max(-x1,0)*B with A=relu(Wx), B=relu(-Wx), so each unit is a one-sided detector of the sign of the current input scaled by its input weight; the registered variables are p1=max(x1,0) and n1=max(-x1,0). "
 "After step 1 the pre-activation can only lie in span[Wx, A@Wh, B@Wh] (rank 3), so h2 lives on a union of cones inside a 3-dim subspace; seven coordinates of h2 pin down the whole 16-vector and ten coordinates of h3 (span rank 5) do, so only those coordinates are registered and the other units are recovered by fixed linear maps P2 (7x16) and P3 (10x16) fitted on the reachable set. "
 "From step 4 on the state fills the full 16-dim space, so the whole state is registered at h4..h10. The recurrence is then iterated with the stored weights and the readout is a linear map h10@Wo; the weights are given in units of 1e-3.")
WX=np.array([69,-10565,15,-123,10157,-281,-10996,-13169,-12312,-228,60,148,298,-47,-1321,-119],float)/1e3
WH=np.array([[161,-108,-84,-1229,35,224,-93,-21,-5,-32,23,92,48,151,403,266],
[711,409,-102,1195,36,-1,-135,-100,634,-357,219,243,469,225,171,652],
[-174,504,969,2,-492,-31,535,674,625,4,40,121,-156,384,335,384],
[21,12,39,-359,-3,-200,27,-8,0,-44,-77,-77,-310,427,-358,379],
[595,871,1731,344,-986,1187,-47,1317,-41,146,1755,1715,1277,464,871,179],
[-913,118,-10,-518,2,-421,38,2,-17,43,-14,-73,-136,-86,-424,-663],
[1133,995,-103,485,93,15,361,-182,0,-89,-193,-473,-104,120,765,381],
[-776,-1048,23,-399,-29,-593,-126,78,-1094,-5032,210,382,-52,2,-712,-161],
[-793,-120,155,-1074,-82,-850,-82,167,615,-1092,-209,-163,-279,-289,-65,-724],
[-1424,-686,-229,208,83,297,-640,3,-68,-553,137,677,810,39,-649,-195],
[307,24,15,214,-7,122,28,-4,38,33,-1,4,-120,277,250,-1093],
[150,8,18,-194,-12,328,19,30,55,53,1096,199,7,-88,17,1280],
[225,43,55,499,-59,261,40,70,42,16,-53,663,612,-261,-19,-224],
[77,30,40,-259,-27,-9,16,22,1,-26,77,-375,414,-605,-159,-529],
[420,79,27,588,-33,205,86,0,32,107,-144,-96,341,-220,355,138],
[8,20,-47,810,-11,74,-8,-13,-13,-10,-18,-119,451,654,-193,229]],float)/1e3
WO=np.array([[1962,-2692,-1652,773,683,1963,1243,-2905,-295,686],
[6094,2562,-1661,-5862,-4893,-1211,2168,4410,-1434,-474],
[-919,-2223,-400,287,201,483,1083,152,578,174],
[668,1694,-505,-541,-473,-915,1696,-584,-610,-600],
[-150,424,577,666,750,567,461,414,498,-4992],
[-228,-82,-77,-347,-510,-640,-678,-369,3398,-419],
[2528,4783,3192,-1217,-5554,-5635,-3080,954,3624,-671],
[-2782,483,4694,7328,5869,921,-2849,-5408,-4997,-3789],
[-3574,-6698,-6205,-1495,3101,5485,4394,1436,3645,328],
[-2100,-1034,412,747,1593,1954,436,217,-1659,-604],
[163,-1666,-2808,-3870,1661,1108,2027,1642,704,1674],
[1005,1267,5,295,-4558,-5,454,786,181,785],
[1186,1703,1418,236,321,-2795,-2219,252,-459,266],
[244,373,-2235,247,178,2086,-1389,227,73,260],
[-5144,608,1113,1309,1229,-623,-1006,1216,88,1182],
[1010,2301,2615,-1137,-937,-724,-949,-829,-916,-853]],float)/1e3
AP=np.maximum(WX,0.0); BP=np.maximum(-WX,0.0)
def step(h,xb): return np.maximum(xb*WX[None,:]+h@WH,0.0)
C2=[0,1,2,4,10,13,14]; C3=[0,1,2,3,4,5,6,8,10,13]
_r=np.random.RandomState(0); _x=_r.randn(4000,10)
_h=np.maximum(_x[:,0:1]*WX[None,:],0.0); _h=step(_h,_x[:,1:2])
P2=np.linalg.lstsq(_h[:,C2],_h,rcond=None)[0]
_h=step(_h,_x[:,2:3])
P3=np.linalg.lstsq(_h[:,C3],_h,rcond=None)[0]
def model(x, iv):
    x=np.asarray(x,float)
    p1=iv('p1',np.maximum(x[:,0],0.0)); n1=iv('n1',np.maximum(-x[:,0],0.0))
    h=np.maximum(p1[:,None]*AP+n1[:,None]*BP,0.0)
    z=step(h,x[:,1:2]); v=iv('h2',z[:,C2]); h=z+(v-z[:,C2])@P2
    z=step(h,x[:,2:3]); v=iv('h3',z[:,C3]); h=z+(v-z[:,C3])@P3
    for t in range(3,10):
        h=iv('h%d'%(t+1),step(h,x[:,t:t+1]))
    return h@WO
ALIGN={'p1':['h1'],'n1':['h1'],'h2':['h2'],'h3':['h3'],'h4':['h4'],'h5':['h5'],'h6':['h6'],'h7':['h7'],'h8':['h8'],'h9':['h9'],'h10':['h10']}
