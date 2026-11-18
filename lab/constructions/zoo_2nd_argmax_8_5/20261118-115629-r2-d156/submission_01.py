import numpy as np
DESCRIPTION = ("A one-layer ReLU recurrent net that answers 'which position holds the second-largest input' "
 "approximately. Account given in canonical units: the input weight vector is rescaled to +-1 "
 "(only its SIGN matters -- the 8 magnitudes are a pure gauge, they can be absorbed into a per-unit "
 "scale of the state, so they carry no information), and the readout is written centre-free "
 "(only the differences between the 5 class scores decide, so each hidden unit's five scores are "
 "shifted to sum to zero and the 5th class is minus the sum of the other four). What is left is a "
 "dense 8x8 recurrent matrix A plus a 6x4 effective readout; units 4 and 6 are almost never clipped "
 "(active 99%/98% of the time) i.e. they are near-linear memory units, groups of units in A have "
 "complex-conjugate eigenvalues near the unit circle so the state rotates with time and thus tags "
 "position.")
SIGN = np.array([1,-1,1,-1,1,-1,-1,1],float)
A = np.array([
[0.12540257,0.01550624,0.72257617,0.00592875,10.99196107,0.00469862,11.21315261,-0.01499044],
[-4.64366248,-0.23636714,4.12918173,0.48147249,24.08297926,-0.95699546,50.34678655,-0.02479358],
[0.01315482,-0.00850972,-0.10689737,-0.01225325,-5.80834771,-0.03652504,1.87795082,0.00493648],
[5.40221872,-0.30711431,-5.63212737,-0.84749269,40.71743618,0.76225076,24.12722671,0.05745640],
[-0.01244972,-0.00227694,0.02299346,-0.00172650,-0.74684125,-0.00175741,-1.00633900,0.00244965],
[6.73596089,0.45941631,1.45830164,0.25992024,13.86175685,0.06958129,-68.13949161,0.01655681],
[0.01067084,0.00810205,-0.01104544,0.00800533,0.65478673,0.00748191,0.78668416,-0.00688021],
[-18.04205800,-0.01880030,23.45679150,1.15855870,24.19329871,1.16125149,142.62923088,-0.82789576]])
Wc = np.array([
[0.58374000,-0.60679109,0.56039861,-1.12842020],
[2.98035797,-3.77825519,-24.25326781,25.76763676],
[0.47855180,0.08991509,-1.31497708,0.35908813],
[3.40639409,-14.90677727,32.61931651,-17.73606746],
[-0.02218162,0.09338433,-0.02508784,-0.02510989],
[-4.10646127,20.80845248,-6.46274434,-6.26362570],
[-0.04565419,-0.04948787,0.02925478,0.03797290],
[3.02649677,3.56931224,3.99249089,3.83945144]])
def model(x, iv):
    s = np.zeros((len(x), 8))
    for t in range(5):
        s = iv('h%d' % (t + 1), np.maximum(x[:, t:t + 1] * SIGN + s @ A, 0))
    o = s @ Wc
    return np.c_[o, -o.sum(1)]
ALIGN = {'h1': ['h1'], 'h2': ['h2'], 'h3': ['h3'], 'h4': ['h4'], 'h5': ['h5']}
