"""SEALED: the planted rules for the yardstick networks. Read only by rosetta/yardstick.py (to make the training
labels) and by the final scorer. The Rosetta engine never imports this file."""
RULES = {
    "S1": "own_N and pawn_att_us and safe_from_opp_pawns",          # a pawn-supported knight no enemy pawn can chase
    "S2": "opp_king_zone and (att_us >= 1) and (att_them <= 1)",    # pressure on the enemy king's zone
    "S3": "own_P and passed_pawn and (rel_rank >= 4)",              # an advanced passed pawn
}
