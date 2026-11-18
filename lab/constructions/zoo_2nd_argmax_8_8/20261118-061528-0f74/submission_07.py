import numpy as np

DESCRIPTION = ("Mechanism, in the cheapest exact form found (121 numbers; the network's decision has 119 degrees of freedom, so this is "
"the floor for any exact description - see the incompressibility argument at the end). "
"h_t = relu(x_t*W_x + h_(t-1)@W_h), out = h_8@W_o. The recurrence is an AMPLIFYING LEAKY INTEGRATOR: the singular values of W_h are "
"3.9, 3.2, 2.4, 2.2, 1.5, 1.2, 0.7, 0.1 (spectral radius 1.7), so each step multiplies the state by about 1.7 and adds the new input, "
"hence the input read at step t is amplified by about 1.7 to the power (8-t): late inputs dominate and |h_8| is of order 70. "
"The ReLU gates open progressively (fraction of inputs with the unit on, by site: at h1 exactly the inputs with x>0 for the units with "
"positive W_x; at h2 [.93,.10,.55,.82,.85,1,.75,0]; at h4 [1,.76,.60,.95,.75,1,.98,.36]; at h8 [1,1,.77,1,.97,1,1,.96]), so from step 4 "
"on the map x -> h_8 is LINEAR on each ReLU cell, and the whole function is a positively homogeneous piecewise-linear map of the input "
"(degree 1, no biases) with only the late-switching units (mainly unit 3, whose drive is -1.2*x_7 plus h_7.W_h[:,3]) selecting the cell. "
"The readout is a FINELY TUNED DIFFERENCE: every class score is of order 6, assembled from terms of order 16 that almost cancel, and the "
"gap between the best and the second class is only about 2 percent of the score scale (median 0.3). The decision therefore lives in the "
"small residual of the scores and is a sloppy argmax of the inputs: it agrees with argmax(x) on 33 percent of inputs and is biased towards "
"late positions; a linear model reaches 54 percent and a quadratic one 54 percent, so no simple rule is being computed - the scores are "
"tied by construction and only an almost exact copy of the tuned arithmetic (a specific 8 by 8 weight pattern at each of three steps) "
"reproduces them. THIS EXPLAINS THE COMPRESSION FAILURE, and why it is not fixable: measured sensitivity, perturbing all weights by 1e-3 "
"relative keeps only 89 percent of decisions (1e-4 keeps 98.7, 1e-5 keeps 99.9); zeroing a single weight of size 0.5 moves 90 percent of "
"decisions; keeping the 7 largest readout entries of each class leaves 48 percent. W_h and W_o show no structure at all: no low rank "
"(rank-7 truncation of W_h leaves 21 percent), no diagonal plus low rank, no Toeplitz/circulant/symmetric/Hadamard/DFT form, no tied, "
"duplicate or rationally related entries (all pairs checked to 1e-6), and the seven class-score difference directions have nearly equal "
"variance (relative singular values 1, .96, .89, .76, .64, .62, .61). Consequently no compact summary of the state carries the decision: "
"the network is a fine-tuned amplifier whose 119 genuine degrees of freedom (136 weights less 8 unit scales, 1 input scale, 8 readout row "
"shifts) are all needed, which is exactly the number of numbers in this file. The invariances themselves are used here: the unit scale D "
"is absorbed so that row 1 of W_h is plus or minus one, and the last readout column is zeroed (that is a common per-input shift of all "
"class scores, so the argmax is untouched).")

WX = np.array([-0.20729784667491913, 0.07802751416829747, -1.6250185336759608, -0.11312824607819615, -0.7229315326962804, 0.022634683224823156, 1.916667143568936, 0.005778828575963005])
WH = np.array([0.08166094869375229, 1, 1, -1, -1, 1, -1, 1, -0.05341569269183388, 0.4797357320785522, -0.4372678582810838, 0.027388813890733495, 0.46218356557444396, -3.238081539489654, -0.5373738521169534, 1.0731267369797464, 0.49472574374669875, -0.297381272796545, 0.23509261012077332, 0.019957895677720026, 1.032302002058723, 9.06547734137075, 2.213976875497503, -0.5594001552408835, 1.6335416532807585, 6.724729452808489, -1.0496278823664857, -0.6714730858802795, -0.39934948163660366, 9.017478793118514, 0.6923325301774357, 0.2207098312960453, -0.20569281592552713, -1.4456701019389266, -0.39351700495338154, 0.6208865278628212, 0.27209100127220154, -0.19269394113898433, 1.0755536555440646, -0.38473403660027095, -0.030651262211490504, -0.008494174080367945, -0.06152329871048301, 0.11809520638952621, -0.14699208070546155, 0.7854707837104798, 0.3842237271778972, 0.1278178380377932, 0.45197418408489654, -0.09576352959260392, 0.22370917543530083, 0.07999831281307557, 0.6037110730802655, 1.3258408851023358, 0.48844647407531744, -0.5181101002334456, -0.16928553423714376, 1.4068260109343595, 0.031080742635476107, 0.05384828187544393, -0.2607423631582319, 1.3557420894048628, 3.867854833178128, -0.6003977656364441]).reshape(8, 8)
WO = np.array([-0.48039841651916504, 0.01933303475379944, 0.12849920988082886, 0.19236670434474945, -0.1864834427833557, 0.24280652403831482, 2.4939146637916565, 0, 0.21379518523353003, -0.004411070944873696, -0.06820066293175508, -0.13516261499902615, -0.5342861734365405, -0.13243921650817914, 0.09135398379116833, 0, 3.174563982033016, 3.346328883939616, 3.192728428008027, 3.4077229124242034, 3.6882585977077227, 3.4170918616577497, 3.780959108089977, 0, 0.5046693576139725, -0.028879103153858, 0.14863045920627949, -0.32496277279790986, -0.44255911495548395, -3.511309333939207, 1.6897832275623323, 0, 0.18671700450072848, -0.1751605586558999, -0.03583040951006122, -0.06099527680121586, 0.13800017748773463, 0.3858370434428693, -1.6252132626673692, 0, 0.027174229070439537, -0.001431753041053696, 0.030393686893980032, -0.009893486829241738, -0.06518873990187757, -0.014275822050089815, -0.4043987578272308, 0, -0.10878631183057852, -0.0009945928357582012, -0.11364743780961373, 0.06484204705369212, 0.259304493051121, 0.06592069835698311, 0.03979168795512189, 0, -0.011507990608012608, -0.001158396503752046, 0.03731117610291701, -0.4616144489742373, 0.24706712646279616, 0.01121157433349218, -0.25925880752199415, 0]).reshape(8, 8)

def model(x, iv):
    h = np.zeros((x.shape[0], 8))
    for t in range(8):
        h = iv('h%d' % (t + 1), np.maximum(x[:, t:t + 1] * WX + h @ WH, 0))
    return h @ WO

ALIGN = {'h%d' % (t + 1): ['h%d' % (t + 1)] for t in range(8)}
