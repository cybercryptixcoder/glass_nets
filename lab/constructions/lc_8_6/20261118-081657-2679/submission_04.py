
import numpy as np
DESCRIPTION = "Interim: the network's default logit vector is dominated by a large constant offset that depends on the last-position token (through the layer-1 attention readout), so the logits are modelled as a 6x6 table indexed by that token. Embedding site: only the last token is named (5 dims)."
M = np.array([[-51.0,-44.9,-22.8,-17.8,3.5,14.8],
              [-33.4,-69.1,-4.5,-25.7,8.5,37.3],
              [-39.2,-72.1,-3.5,-27.6,10.4,38.3],
              [-55.6,-43.5,-21.4,-18.4,1.8,9.1],
              [-62.0,-62.2,-23.2,-21.2,-4.4,24.5],
              [-23.9,-77.6,3.7,-27.8,9.5,48.5]])
def model(x, iv):
    t5 = iv("tok5", np.eye(6)[x[:, 5]][:, :5])
    return M[x[:, 5]]
ALIGN = {"tok5": ["embed"]}
