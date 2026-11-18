import numpy as np

DESCRIPTION = ("A 4-unit ReLU recurrent net with mirror symmetry: unit 0 mirrors unit 3 and unit 1 mirrors unit 2 "
               "(W_h is block-structured under that swap, W_o has equal rows for the mirror pairs). The state is "
               "summarised by pair sums P=(h0+h3)/2, Q=(h1+h2)/2 and pair differences D=(h0-h3)/2, E=(h1-h2)/2; the "
               "recurrent term splits into a weak symmetric part (P,Q coefficients ~0.4-0.55) and a strong "
               "antisymmetric part (D,E coefficients ~3-5). The input injects x_t with opposite signs into the mirror "
               "pairs, so each step rebuilds D,E from x_t. The readout uses only the pair sums of the final state.")

WX = (3.9653, 4.6995, -4.7163, -3.9652)
# block coefficients of W_h in the (P,D,Q,E) basis
S1, A1 = -0.5541, 3.2337      # group1 (units 0/3) symmetric / antisymmetric
S2, A2 = 0.4436, -4.8344      # cross group1<-group2
S3, A3 = -0.4547, 4.3155      # cross group2<-group1
S4, A4 = 0.3623, -5.2301      # group2 (units 1/2)
OA = (-1.6646, -18.736, 11.4537)
OB = (1.6084, 15.979, -15.719)


def step(u, h):
    P = (h[:, 0] + h[:, 3]) / 2
    D = (h[:, 0] - h[:, 3]) / 2
    Q = (h[:, 1] + h[:, 2]) / 2
    E = (h[:, 1] - h[:, 2]) / 2
    r0 = S1 * P + A1 * D + S2 * Q + A2 * E
    r1 = S3 * P + A3 * D + S4 * Q + A4 * E
    r2 = S3 * P - A3 * D + S4 * Q - A4 * E
    r3 = S1 * P - A1 * D + S2 * Q - A2 * E
    v = np.stack([u * WX[0] + r0, u * WX[1] + r1, u * WX[2] + r2, u * WX[3] + r3], 1)
    return np.maximum(v, 0)


def model(x, iv):
    z = np.zeros((x.shape[0], 4))
    h1 = iv("h1", step(x[:, 0], z))
    h2 = iv("h2", step(x[:, 1], h1))
    h3 = iv("h3", step(x[:, 2], h2))
    A = h3[:, 0] + h3[:, 3]
    B = h3[:, 1] + h3[:, 2]
    return A[:, None] * np.array(OA)[None, :] + B[:, None] * np.array(OB)[None, :]


ALIGN = {"h1": ["h1"], "h2": ["h2"], "h3": ["h3"]}
