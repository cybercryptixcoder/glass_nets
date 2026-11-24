"""Leela Chess Zero transformer networks (e.g. t1-256x10-distilled) in PyTorch.

Ported from lczero-training (tfprocess.py: create_encoder_body, encoder_layer, smolgen_weights,
apply_promotion_logits; net.py: proto layout, dense weights stored [out, in], 2-D gates transposed too).
  input  112 classical planes -> 64 square tokens, concatenated with the 64-d arc positional encoding (176)
  embed  dense 176 -> E, Mish, then per-square multiplicative and additive gates
  body   N encoder layers: MHA with smolgen attention biases; DeepNorm residuals (alpha = (2N)^0.25);
         LayerNorm eps 1e-6; FFN activation and smolgen activation as lc0 resolves them (old-protobuf attention nets
         with smolgen, like t1-256x10: FFN squared ReLU, smolgen swish; lc0 src/neural/loader.cc). Smolgen: compress (no bias) -> dense+swish -> LN -> dense+swish -> LN
         -> shared generator (no bias) -> [heads, 64, 64]; smolgen LayerNorms use the Keras default eps 1e-3
  policy dense+Mish embedding; queries and keys; logits Q K^T / sqrt(d); promotion offsets from the last-rank keys
  value  per-square dense+Mish embedding, flatten, dense+Mish 128, dense 3 (WDL softmax)
The lczero-training support files (GPL-3.0) are downloaded at setup into $KDATA/chess, not copied here.
Sites: 0 = after the gated embedding, i = after encoder layer i; each [B, 64, E] (token = mover's-frame square).
"""
import gzip
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .leela import CHESS_DIR, _dec, _support


def _mish(x):
    return x * torch.tanh(F.softplus(x))


ACTS = {"mish": _mish, "swish": F.silu, "relu": F.relu, "relu2": lambda x: F.relu(x) ** 2, "selu": F.selu,
        "tanh": torch.tanh, "none": lambda x: x}
# lc0 NetworkFormat.ActivationFunction numbers -> names (net.proto)
ACT_ENUM = {1: "mish", 2: "relu", 3: "none", 4: "tanh", 5: "sigmoid", 6: "selu", 7: "swish", 8: "relu2", 9: "softmax"}


def resolve_activations(net):
    """(default, ffn, smolgen) activation names exactly as lc0 resolves them (loader.cc FixOlderWeightsFile +
    network_blas.cc constructor)."""
    nf, W = net.format.network_format, net.weights
    default = "mish" if nf.default_activation == 1 else "relu"
    ffn, smol = nf.ffn_activation, nf.smolgen_activation
    if nf.network == 4 and len(W.encoder) > 0 and W.HasField("smolgen_w"):
        ffn, smol = 8, 7          # old protobuf attention body with smolgen: RELU_2 FFN, SWISH smolgen
    ffn = default if ffn == 0 else ACT_ENUM[ffn]
    smol = default if smol == 0 else ACT_ENUM[smol]
    return default, ffn, smol


TF_LAYOUT = False   # set by LeelaTransformer: dense weights stored [in, out] (TensorFlow) instead of [out, in]


def _lin(w, b, n_in, bias=True):
    a = _dec(w)
    W = torch.tensor(a.reshape(n_in, -1).T.copy() if TF_LAYOUT else a.reshape(-1, n_in))
    lin = nn.Linear(n_in, W.shape[0], bias=bias)
    lin.weight.data = W
    if bias:
        lin.bias.data = torch.tensor(_dec(b))
    return lin


def _ln(g, b, n, eps):
    ln = nn.LayerNorm(n, eps=eps)
    ln.weight.data = torch.tensor(_dec(g))
    ln.bias.data = torch.tensor(_dec(b))
    return ln


def _pos_enc():
    import sys
    if CHESS_DIR not in sys.path:
        sys.path.insert(0, CHESS_DIR)
    from attention_policy_map import make_pos_enc, make_map
    return torch.tensor(make_pos_enc()[0]), make_map()


class Encoder(nn.Module):
    def __init__(self, e, E, heads, alpha, gen, ffn_act, smol_act):
        super().__init__()
        self.h, self.alpha, self.gen = heads, alpha, gen
        self.ffn_act, self.smol_act = ACTS[ffn_act], ACTS[smol_act]
        self.q = _lin(e.mha.q_w, e.mha.q_b, E)
        self.k = _lin(e.mha.k_w, e.mha.k_b, E)
        self.v = _lin(e.mha.v_w, e.mha.v_b, E)
        D = self.q.weight.shape[0]
        self.o = _lin(e.mha.dense_w, e.mha.dense_b, D)
        self.ln1 = _ln(e.ln1_gammas, e.ln1_betas, E, 1e-6)
        self.f1 = _lin(e.ffn.dense1_w, e.ffn.dense1_b, E)
        self.f2 = _lin(e.ffn.dense2_w, e.ffn.dense2_b, self.f1.weight.shape[0])
        self.ln2 = _ln(e.ln2_gammas, e.ln2_betas, E, 1e-6)
        s = e.mha.smolgen
        self.sc = _lin(s.compress, None, E, bias=False)
        hc = self.sc.weight.shape[0]
        self.s1 = _lin(s.dense1_w, s.dense1_b, 64 * hc)
        self.sl1 = _ln(s.ln1_gammas, s.ln1_betas, self.s1.weight.shape[0], 1e-3)
        self.s2 = _lin(s.dense2_w, s.dense2_b, self.s1.weight.shape[0])
        self.sl2 = _ln(s.ln2_gammas, s.ln2_betas, self.s2.weight.shape[0], 1e-3)

    def forward(self, x, attn_out=None):
        B = x.shape[0]
        q, k, v = self.q(x), self.k(x), self.v(x)
        D = q.shape[-1]
        d = D // self.h
        q, k, v = (t.view(B, 64, self.h, d).transpose(1, 2) for t in (q, k, v))
        logits = q @ k.transpose(-1, -2) / math.sqrt(d)
        c = self.sc(x).reshape(B, -1)
        g = self.sl1(self.smol_act(self.s1(c)))
        g = self.sl2(self.smol_act(self.s2(g))).view(B, self.h, -1)
        logits = logits + self.gen(g).view(B, self.h, 64, 64)
        a = torch.softmax(logits, -1)
        if attn_out is not None:
            attn_out.append(a)
        o = (a @ v).transpose(1, 2).reshape(B, 64, D)
        x = self.ln1(x * self.alpha + self.o(o))
        return self.ln2(x * self.alpha + self.f2(self.ffn_act(self.f1(x))))


class LeelaTransformer(nn.Module):
    def __init__(self, path, tf_layout=False, gates_transposed=True):
        super().__init__()
        global TF_LAYOUT
        TF_LAYOUT = tf_layout
        net_pb2, policy_index, _ = _support()
        net = net_pb2.Net()
        net.ParseFromString(gzip.open(path).read())
        fmt = net.format.network_format
        assert fmt.input == 1 and len(net.weights.encoder) > 0 and fmt.value == 2
        W = net.weights
        self.name = os.path.basename(path).split(".")[0]
        pos, amap = _pos_enc()
        self.register_buffer("pos", pos)                       # [64, 64]
        E = _dec(W.ip_emb_b).size
        n_in = _dec(W.ip_emb_w).size // E
        self.use_pos = n_in == 112 + 64
        self.emb = _lin(W.ip_emb_w, W.ip_emb_b, n_in)
        TF = TF_LAYOUT
        gm, ga = _dec(W.ip_mult_gate), _dec(W.ip_add_gate)
        if gates_transposed:
            gm, ga = gm.reshape(E, 64).T.copy(), ga.reshape(E, 64).T.copy()
        self.register_buffer("mult", torch.tensor(gm.reshape(64, E)))
        self.register_buffer("add", torch.tensor(ga.reshape(64, E)))
        heads = W.headcount
        N = len(W.encoder)
        alpha = (2.0 * N) ** 0.25
        default, ffn_act, smol_act = resolve_activations(net)
        assert default == "mish", "heads below assume Mish as the default activation"
        self.activations = {"default": default, "ffn": ffn_act, "smolgen": smol_act}
        self.gen = _lin(W.smolgen_w, None, _dec(W.encoder[0].mha.smolgen.dense2_b).size // heads, bias=False)
        self.enc = nn.ModuleList([Encoder(e, E, heads, alpha, self.gen, ffn_act, smol_act) for e in W.encoder])
        self.channels = E
        self.pemb = _lin(W.ip_pol_w, W.ip_pol_b, E)
        Pe = self.pemb.weight.shape[0]
        self.wq = _lin(W.ip2_pol_w, W.ip2_pol_b, Pe)
        self.wk = _lin(W.ip3_pol_w, W.ip3_pol_b, Pe)
        self.ppo = _lin(W.ip4_pol_w, None, self.wk.weight.shape[0], bias=False)
        self.vemb = _lin(W.ip_val_w, W.ip_val_b, E)
        self.ip1 = _lin(W.ip1_val_w, W.ip1_val_b, 64 * self.vemb.weight.shape[0])
        self.ip2 = _lin(W.ip2_val_w, W.ip2_val_b, self.ip1.weight.shape[0])
        rows = np.argmax(amap, axis=0)                          # for each of the 1858 moves, its source index
        self.register_buffer("gather", torch.tensor(rows, dtype=torch.long))
        self.policy_index = policy_index
        self.move_to_idx = {m: i for i, m in enumerate(policy_index)}
        self.eval()

    @property
    def n_sites(self):
        return len(self.enc) + 1

    def forward(self, x, return_acts=False, edit=None, upto=None):
        B = x.shape[0]
        t = x.permute(0, 2, 3, 1).reshape(B, 64, 112)
        if self.use_pos:
            t = torch.cat([t, self.pos.expand(B, 64, 64)], 2)
        h = _mish(self.emb(t)) * self.mult + self.add
        if edit is not None:
            h = edit(0, h)
        acts = [h]
        if upto == 0:
            return h
        for i, layer in enumerate(self.enc):
            h = layer(h)
            if edit is not None:
                h = edit(i + 1, h)
            acts.append(h)
            if upto == i + 1:
                return h
        return self.heads(h, acts if return_acts else None)

    def heads(self, h, acts=None):
        B = h.shape[0]
        p = _mish(self.pemb(h))
        q, k = self.wq(p), self.wk(p)
        dk = math.sqrt(k.shape[-1])
        qk = q @ k.transpose(-1, -2)                             # [B, 64 from, 64 to]
        off = self.ppo(k[:, -8:, :]).transpose(1, 2) * dk        # [B, 4, 8]
        off = off[:, :3, :] + off[:, 3:4, :]
        n_promo = qk[:, -16:-8, -8:]                             # [B, 8 from (rank 7), 8 to (rank 8)]
        promo = torch.stack([n_promo + off[:, j:j + 1, :] for j in range(3)], 3).reshape(B, 8, 24) / dk
        flat = torch.cat([(qk / dk).reshape(B, -1), promo.reshape(B, -1)], 1)
        logits = flat[:, self.gather]
        v = _mish(self.vemb(h)).reshape(B, -1)
        wdl = F.softmax(self.ip2(_mish(self.ip1(v))), dim=1)
        return (logits, wdl, acts) if acts is not None else (logits, wdl)
