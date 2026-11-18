
import numpy as np
DESCRIPTION = "Diagnostic step: at the embed site the decision-relevant content is dominated by a very low-dimensional subspace (the judge found one principal component suffices), so only the token at the last position is named here, as a 5-dimensional one-hot. Output is still a placeholder constant class 5."
def model(x, iv):
    B = x.shape[0]
    t5 = iv("tok5", np.eye(6)[x[:, 5]][:, :5])
    out = np.zeros((B, 6)); out[:, 5] = 1.0
    return out
ALIGN = {"tok5": ["embed"]}
