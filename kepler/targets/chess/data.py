"""Positions for the chess target: real human games (Lichess, Jan 2013) and Lichess puzzles (with human theme labels)."""
import csv
import io
import random

import chess
import chess.pgn
import zstandard


def _text_stream(path):
    fh = open(path, "rb")
    reader = zstandard.ZstdDecompressor().stream_reader(fh, read_across_frames=True)
    return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")


def game_positions(path, n, seed=0, per_game=2, min_ply=10):
    """Up to `per_game` random positions (after ply >= min_ply, game not over) from each game, until n positions."""
    rng = random.Random(seed)
    out = []
    stream = _text_stream(path)
    while len(out) < n:
        try:
            g = chess.pgn.read_game(stream)
        except Exception:
            continue
        if g is None:
            break
        moves = list(g.mainline_moves())
        if len(moves) < min_ply + 4:
            continue
        plies = sorted(rng.sample(range(min_ply, len(moves) - 1), min(per_game, len(moves) - 1 - min_ply)))
        b = g.board()
        k = 0
        for i, m in enumerate(moves):
            if k < len(plies) and i == plies[k]:
                if not b.is_game_over():
                    out.append({"fen": b.fen(), "ply": i, "result": g.headers.get("Result", "*"),
                                "white_elo": g.headers.get("WhiteElo", "?"), "black_elo": g.headers.get("BlackElo", "?"),
                                "played": m.uci()})
                k += 1
            b.push(m)
    return out[:n]


def puzzles(path, n, seed=0):
    """Lichess puzzles: the position after the opponent's first move, the solution's first move, and themes."""
    out = []
    stream = _text_stream(path)
    rd = csv.reader(stream)
    header = None
    try:
        for row in rd:
            if header is None:
                header = row
                if row[0] != "PuzzleId":
                    header = ["PuzzleId", "FEN", "Moves", "Rating", "RatingDeviation", "Popularity", "NbPlays",
                              "Themes", "GameUrl", "OpeningTags"]
                    row_ok = row
                else:
                    continue
            r = dict(zip(header, row))
            try:
                b = chess.Board(r["FEN"])
                mv = r["Moves"].split()
                b.push(chess.Move.from_uci(mv[0]))
                best = chess.Move.from_uci(mv[1])
                if best not in b.legal_moves:
                    continue
            except Exception:
                continue
            out.append({"fen": b.fen(), "best": best.uci(), "rating": int(r["Rating"]), "themes": r["Themes"].split(),
                        "id": r["PuzzleId"]})
            if len(out) >= n:
                break
    except Exception:
        pass   # the file is a partial download: stop at the cut
    random.Random(seed).shuffle(out)
    return out
