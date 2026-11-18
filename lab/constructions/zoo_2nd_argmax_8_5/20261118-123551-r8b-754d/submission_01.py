import numpy as np
DESCRIPTION = ("Reads 5 numbers in order. Hidden state of 8 ReLU units, h_t = relu(x_t * W_x + h_{t-1} @ W_h); "
               "the score of class c is a linear read-out of the final state. The read-out is written centred "
               "(each unit's five scores sum to zero), so class 5's score is minus the sum of the others; only the "
               "score differences matter for the decision, so this is exact. Weights written to 3 decimals.")
W_x = np.array([0.643, -6.421, 0.423, -6.574, 0.045, -4.913, -0.054, 7.002])
W_h = np.array([
 [0.125, 0.155, 0.476, 0.061, 0.771, 0.036, 0.944, -0.163],
 [-0.465, -0.236, 0.272, 0.493, 0.169, -0.732, 0.424, -0.027],
 [0.02, -0.129, -0.107, -0.19, -0.618, -0.424, 0.24, 0.082],
 [0.528, -0.3, -0.363, -0.847, 0.279, 0.57, 0.199, 0.061],
 [-0.177, -0.324, 0.216, -0.252, -0.747, -0.192, -1.207, 0.38],
 [0.881, 0.6, 0.126, 0.348, 0.127, 0.07, -0.75, 0.024],
 [0.127, 0.962, -0.086, 0.973, 0.546, 0.68, 0.787, -0.89],
 [-1.656, -0.017, 1.419, 1.088, 0.156, 0.815, 1.102, -0.828]])
W_c = np.array([
 [0.908, -0.944, 0.872, -1.756],
 [0.464, -0.588, -3.777, 4.013],
 [1.13, 0.212, -3.105, 0.848],
 [0.518, -2.268, 4.962, -2.698],
 [-0.492, 2.071, -0.556, -0.557],
 [-0.836, 4.235, -1.315, -1.275],
 [-0.844, -0.915, 0.541, 0.702],
 [0.432, 0.51, 0.57, 0.548]])

def model(x, iv):
    s = np.zeros((len(x), 8))
    for t in range(5):
        s = iv('h%d' % (t + 1), np.maximum(x[:, t:t + 1] * W_x + s @ W_h, 0))
    p = s @ W_c
    return np.column_stack([p, -p.sum(1)])

ALIGN = {'h1': ['h1'], 'h2': ['h2'], 'h3': ['h3'], 'h4': ['h4'], 'h5': ['h5']}
