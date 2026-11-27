"""Adapter for Tracr-compiled transformers (S0's public kind): run the network, and intervene at a named site.

The forward pass is re-implemented in numpy from the compiled weights (no layer norm, one or more heads, softmax
attention, ReLU MLPs, residual connections), and checked against Tracr's own `model.apply` (`check_against_tracr`).
Sites: `resid_0` (after the embeddings), then `resid_<k>` after each sublayer (attention, MLP, attention, ...), so a
network with n layers has sites resid_0 .. resid_<2n>. A location is (site, labels): a subspace spanned by residual
basis directions, which is one of SPEC R3's allowed linear map classes (a subspace of declared dimension), or a
general matrix of directions (for the random-direction controls).
"""
import numpy as np


class TracrNet:
    def __init__(self, program, vocab, max_seq_len, bos="BOS"):
        from tracr.compiler import compiling
        self.model = compiling.compile_rasp_to_model(program, vocab=set(vocab), max_seq_len=max_seq_len, compiler_bos=bos)
        self.bos, self.vocab, self.max_seq_len = bos, sorted(vocab), max_seq_len
        cfg = self.model.model_config
        self.H, self.ks, self.n_layers = cfg.num_heads, cfg.key_size, cfg.num_layers
        self.P = {k: {kk: np.asarray(vv, dtype=np.float64) for kk, vv in v.items()} for k, v in self.model.params.items()}
        self.labels = [str(x) for x in self.model.residual_labels]
        self.d = len(self.labels)
        self.sites = [f"resid_{k}" for k in range(2 * self.n_layers + 1)]
        self.out_name = program.label
        self.numerical_output = type(self.model.output_encoder).__name__ == "NumericalEncoder"

    # ------------------------------------------------------------------ locations
    def subspace(self, labels):
        """Orthonormal basis [d, k] for the named residual directions."""
        idx = [self.labels.index(l) for l in labels]
        B = np.zeros((self.d, len(idx)))
        B[idx, np.arange(len(idx))] = 1.0
        return B

    def labels_of(self, sop_name):
        """The residual directions Tracr assigned to an s-op: `name` (numerical) or `name:value` (categorical)."""
        return [l for l in self.labels if l == sop_name or l.startswith(sop_name + ":")]

    # ------------------------------------------------------------------ running
    def encode(self, seqs):
        n = max(len(s) for s in seqs) + 1
        ids = np.zeros((len(seqs), n), dtype=int)
        mask = np.zeros((len(seqs), n), dtype=bool)
        for i, s in enumerate(seqs):
            enc = self.model.input_encoder.encode([self.bos] + list(s))
            ids[i, :len(enc)] = enc
            mask[i, :len(enc)] = True
        return ids, mask

    def run(self, seqs, edits=None, return_sites=False):
        """edits: {site: fn(h) -> h}, h of shape [B, T, d]. Returns the final residual (and every site's residual)."""
        edits = edits or {}
        ids, mask = self.encode(seqs)
        T = ids.shape[1]
        h = self.P["token_embed"]["embeddings"][ids] + self.P["pos_embed"]["embeddings"][np.arange(T)][None]
        site = {}

        def at(k, h):
            name = f"resid_{k}"
            if name in edits:
                h = edits[name](h)
            site[name] = h
            return h
        h = at(0, h)
        for layer in range(self.n_layers):
            p = f"transformer/layer_{layer}"
            q = h @ self.P[f"{p}/attn/query"]["w"] + self.P[f"{p}/attn/query"]["b"]
            k_ = h @ self.P[f"{p}/attn/key"]["w"] + self.P[f"{p}/attn/key"]["b"]
            v = h @ self.P[f"{p}/attn/value"]["w"] + self.P[f"{p}/attn/value"]["b"]
            B_, T_, _ = q.shape
            q = q.reshape(B_, T_, self.H, -1); k_ = k_.reshape(B_, T_, self.H, -1); v = v.reshape(B_, T_, self.H, -1)
            logits = np.einsum("bthk,bshk->bhts", q, k_)
            logits = np.where(mask[:, None, None, :], logits, -1e30)
            logits = logits - logits.max(-1, keepdims=True)
            w = np.exp(logits); w = w / w.sum(-1, keepdims=True)
            att = np.einsum("bhts,bshk->bthk", w, v).reshape(B_, T_, -1)
            h = h + att @ self.P[f"{p}/attn/linear"]["w"] + self.P[f"{p}/attn/linear"]["b"]
            h = at(2 * layer + 1, h)
            z = np.maximum(h @ self.P[f"{p}/mlp/linear_1"]["w"] + self.P[f"{p}/mlp/linear_1"]["b"], 0)
            h = h + z @ self.P[f"{p}/mlp/linear_2"]["w"] + self.P[f"{p}/mlp/linear_2"]["b"]
            h = at(2 * layer + 2, h)
        return (h, site, mask) if return_sites else h

    def decode(self, h, seqs):
        """Network outputs per sequence (without the BOS position), as Tracr decodes them."""
        out_labels = self.labels_of(self.out_name)
        B = self.subspace(out_labels)
        vals = h @ B
        res = []
        for i, s in enumerate(seqs):
            row = vals[i, 1:len(s) + 1]
            if self.numerical_output:
                res.append([float(r[0]) for r in row])
            else:
                names = [l.split(":", 1)[1] for l in out_labels]
                res.append([_typed(names[int(np.argmax(r))]) for r in row])
        return res

    def outputs(self, seqs, edits=None):
        return self.decode(self.run(seqs, edits), seqs)

    def check_against_tracr(self, seqs):
        """Max abs difference between this forward pass and Tracr's own, over every residual and the output."""
        worst = 0.0
        for s in seqs:
            ref = self.model.apply([self.bos] + list(s))
            h, site, _ = self.run([s], return_sites=True)
            ours = [site[f"resid_{k}"][0] for k in range(1, 2 * self.n_layers + 1)]
            theirs = [np.asarray(r)[0] for r in ref.residuals]
            for a, b in zip(ours, theirs):
                worst = max(worst, float(np.abs(a - b).max()))
        return worst


def _typed(x):
    try:
        f = float(x)
        return int(f) if f.is_integer() else f
    except ValueError:
        return x


def swap_edit(B, h_source, positions=None):
    """An interchange intervention: put h_source's component in the subspace B (orthonormal [d, k]) into h."""
    P = B @ B.T

    def fn(h):
        T = min(h.shape[1], h_source.shape[1])
        out = h.copy()
        out[:, :T] = h[:, :T] - h[:, :T] @ P + h_source[:, :T] @ P
        return out
    return fn


def set_edit(B, coords):
    """Set the component in subspace B to given coordinates [batch, T, k]."""
    P = B @ B.T

    def fn(h):
        T = min(h.shape[1], coords.shape[1])
        out = h.copy()
        out[:, :T] = h[:, :T] - h[:, :T] @ P + coords[:, :T] @ B.T
        return out
    return fn
