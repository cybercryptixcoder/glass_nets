"""Leela Chess Zero networks (classical 112-plane input, SE ResNet, convolutional policy, WDL value) in PyTorch.

Written from lc0's weight format (lczero-common net.proto) and its batch-norm folding rule (lc0
network_legacy.cc: gamma' = gamma / sqrt(var + 1e-5), mean' = mean - bias, W' = W * gamma', b' = -gamma' * mean' + beta).
The policy map and move list come from lczero-training (GPL-3.0); they are downloaded at setup into $KDATA/chess,
not copied into this repository (`scripts/kepler/chess_setup.sh`).

Sites for interpretability: 0 = after the input block, i = after residual block i. Each site is [B, 128, 8, 8] in the
side-to-move frame (row 0 = the mover's first rank).
"""
import gzip
import os
import sys

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

KDATA = os.environ.get("KDATA", os.path.expanduser("~/kdata"))
CHESS_DIR = os.path.join(KDATA, "chess")


def _support():
    if CHESS_DIR not in sys.path:
        sys.path.insert(0, CHESS_DIR)
    import net_pb2  # noqa: F401
    from policy_index import policy_index
    from lc0_az_policy_map import make_map
    return net_pb2, policy_index, make_map


def _dec(layer):
    if not layer.params:
        return None
    raw = layer.params
    if layer.encoding in (0, 1):            # LINEAR16
        q = np.frombuffer(raw, dtype=np.uint16).astype(np.float64)
        return (layer.min_val + (layer.max_val - layer.min_val) * q / 65535.0).astype(np.float32)
    if layer.encoding == 2:                 # FLOAT16
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    if layer.encoding == 3:                 # BFLOAT16
        return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    raise ValueError(f"unknown encoding {layer.encoding}")


def _fold(block, cin, k):
    """Decode one lc0 ConvBlock into (weight [out, in, k, k], bias [out]) with batch norm folded in."""
    w = _dec(block.weights)
    b = _dec(block.biases)
    means, var = _dec(block.bn_means), _dec(block.bn_stddivs)
    gam, bet = _dec(block.bn_gammas), _dec(block.bn_betas)
    n_out = w.size // (cin * k * k)
    w = w.reshape(n_out, cin, k, k).astype(np.float64)
    b = np.zeros(n_out) if b is None else b.astype(np.float64)
    if means is not None:
        gam = np.ones(n_out) if gam is None else gam.astype(np.float64)
        bet = np.zeros(n_out) if bet is None else bet.astype(np.float64)
        g = gam / np.sqrt(var.astype(np.float64) + 1e-5)
        m = means.astype(np.float64) - b
        w = w * g[:, None, None, None]
        b = -g * m + bet
    return torch.tensor(w, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)


def _conv(w, b):
    c = nn.Conv2d(w.shape[1], w.shape[0], w.shape[2], padding=w.shape[2] // 2)
    c.weight.data, c.bias.data = w, b
    return c


def _fc(w, b, n_in):
    w = torch.tensor(w.reshape(-1, n_in))
    lin = nn.Linear(n_in, w.shape[0])
    lin.weight.data, lin.bias.data = w, torch.tensor(b)
    return lin


class LeelaNet(nn.Module):
    def __init__(self, path):
        super().__init__()
        net_pb2, policy_index, make_map = _support()
        net = net_pb2.Net()
        net.ParseFromString(gzip.open(path).read())
        fmt = net.format.network_format
        assert fmt.input == 1 and fmt.policy == 2 and fmt.value == 2, "only classical-input conv-policy WDL nets"
        W = net.weights
        self.name = os.path.basename(path).split(".")[0]
        self.inp = _conv(*_fold(W.input, 112, 3))
        C = self.inp.weight.shape[0]
        self.channels = C
        self.c1 = nn.ModuleList()
        self.c2 = nn.ModuleList()
        self.se1 = nn.ModuleList()
        self.se2 = nn.ModuleList()
        for r in W.residual:
            self.c1.append(_conv(*_fold(r.conv1, C, 3)))
            self.c2.append(_conv(*_fold(r.conv2, C, 3)))
            s = _dec(r.se.b1).size
            self.se1.append(_fc(_dec(r.se.w1), _dec(r.se.b1), C))
            self.se2.append(_fc(_dec(r.se.w2), _dec(r.se.b2), s))
        self.pol1 = _conv(*_fold(W.policy1, C, 3))
        self.pol = _conv(*_fold(W.policy, C, 3))
        self.val = _conv(*_fold(W.value, C, 1))
        nv = self.val.weight.shape[0]
        self.ip1 = _fc(_dec(W.ip1_val_w), _dec(W.ip1_val_b), nv * 64)
        self.ip2 = _fc(_dec(W.ip2_val_w), _dec(W.ip2_val_b), self.ip1.weight.shape[0])
        idx = make_map("index")
        gather = np.zeros(len(policy_index), dtype=np.int64)
        for e, i in enumerate(idx):
            if i >= 0:
                gather[i] = e
        self.register_buffer("gather", torch.tensor(gather))
        self.policy_index = policy_index
        self.move_to_idx = {m: i for i, m in enumerate(policy_index)}
        self.eval()

    @property
    def n_sites(self):
        return len(self.c1) + 1

    def forward(self, x, return_acts=False, edit=None, upto=None):
        """x: [B, 112, 8, 8]. Returns policy logits [B, 1858] and WDL probabilities [B, 3] (win, draw, loss for the
        side to move). edit(site, h) -> h patches a site; upto=k stops after site k and returns its activation."""
        h = F.relu(self.inp(x))
        if edit is not None:
            h = edit(0, h)
        acts = [h]
        if upto == 0:
            return h
        for i in range(len(self.c1)):
            y = F.relu(self.c1[i](h))
            y = self.c2[i](y)
            s = F.relu(self.se1[i](y.mean(dim=(2, 3))))
            g = self.se2[i](s)
            gamma, beta = g[:, : self.channels], g[:, self.channels:]
            h = F.relu(torch.sigmoid(gamma)[:, :, None, None] * y + beta[:, :, None, None] + h)
            if edit is not None:
                h = edit(i + 1, h)
            acts.append(h)
            if upto == i + 1:
                return h
        return self.heads(h, acts if return_acts else None)

    def heads(self, h, acts=None):
        p = self.pol(F.relu(self.pol1(h))).flatten(1)
        logits = p[:, self.gather]
        v = F.relu(self.val(h)).flatten(1)
        wdl = F.softmax(self.ip2(F.relu(self.ip1(v))), dim=1)
        return (logits, wdl, acts) if acts is not None else (logits, wdl)


# ---------------------------------------------------------------- encoding
PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def _sq(sq, flip):
    return chess.square_mirror(sq) if flip else sq


def encode(board):
    """lc0 classical 112-plane input for one position given as FEN only (lc0 FEN_ONLY history fill: the current
    position repeated in all 8 history slots, from the mover's view; the start position gets one slot)."""
    flip = board.turn == chess.BLACK
    planes = np.zeros((112, 64), dtype=np.float32)
    us, them = board.turn, not board.turn
    one = np.zeros((13, 64), dtype=np.float32)
    for k, pt in enumerate(PIECES):
        for sq in board.pieces(pt, us):
            one[k, _sq(sq, flip)] = 1
        for sq in board.pieces(pt, them):
            one[6 + k, _sq(sq, flip)] = 1
    slots = 1 if board.board_fen() == chess.STARTING_BOARD_FEN else 8
    for h in range(slots):
        planes[h * 13:(h + 1) * 13] = one
    if board.has_queenside_castling_rights(us):
        planes[104] = 1
    if board.has_kingside_castling_rights(us):
        planes[105] = 1
    if board.has_queenside_castling_rights(them):
        planes[106] = 1
    if board.has_kingside_castling_rights(them):
        planes[107] = 1
    if flip:
        planes[108] = 1
    planes[109] = board.halfmove_clock
    planes[111] = 1
    return planes.reshape(112, 8, 8)


def move_index(net, board, move):
    """Index of a python-chess move in the 1858 policy (mover's frame; knight promotion is the plain move)."""
    flip = board.turn == chess.BLACK
    f, t = _sq(move.from_square, flip), _sq(move.to_square, flip)
    uci = chess.square_name(f) + chess.square_name(t)
    # lc0 encodes castling as king-takes-rook (e1h1 / e1a1)
    if board.is_castling(move):
        rook_file = 7 if chess.square_file(move.to_square) > chess.square_file(move.from_square) else 0
        t = chess.square(rook_file, chess.square_rank(f))
        uci = chess.square_name(f) + chess.square_name(t)
    if move.promotion and move.promotion != chess.KNIGHT:
        uci += chess.piece_symbol(move.promotion)
    return net.move_to_idx[uci]


@torch.no_grad()
def evaluate(net, boards, batch=256):
    """Policy over legal moves and WDL for a list of boards. Returns list of dicts."""
    out = []
    for i in range(0, len(boards), batch):
        bs = boards[i:i + batch]
        x = torch.tensor(np.stack([encode(b) for b in bs]))
        logits, wdl = net(x)
        for b, lg, w in zip(bs, logits, wdl):
            moves = list(b.legal_moves)
            if moves:
                li = torch.stack([lg[move_index(net, b, m)] for m in moves])
                p = F.softmax(li, 0).numpy()
            else:
                p = np.zeros(0)
            out.append({"moves": moves, "p": p, "wdl": w.numpy(), "q": float(w[0] - w[2])})
    return out
