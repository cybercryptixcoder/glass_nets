"""Transfer task for gpt2_month_rollover: the same months and years, written "<month1>, <year> - <month2>" (a comma
after the month, so every later token sits one position further on). Tests whether a construction's maps and
decisions hold beyond the exact template it was found on (lab/lm_transfer.py; roles are matched by name)."""
import importlib.util, pathlib
_s = importlib.util.spec_from_file_location("_mr", pathlib.Path(__file__).with_name("month_rollover.py"))
_mr = importlib.util.module_from_spec(_s); _s.loader.exec_module(_mr)
ID = "gpt2_month_rollover_comma"
MODEL, ANSWERS, X_NAMES, SPLIT_BY, MONTHS, YEARS = _mr.MODEL, _mr.ANSWERS, _mr.X_NAMES, _mr.SPLIT_BY, _mr.MONTHS, _mr.YEARS
TEXT = _mr.TEXT.replace('"<month1> <year> - <month2>"', '"<month1>, <year> - <month2>"').replace(
    '1 month1, 2 the year, 3 " -", 4 month2', '1 month1, 2 ",", 3 the year, 4 " -", 5 month2').replace(
    "m1 = 1, year = 2, dash = 3, m2 = end = 4", "m1 = 1, comma = 2, year = 3, dash = 4, m2 = end = 5")
CONTEXT, ANSWER = _mr.CONTEXT, _mr.ANSWER
NAME = "month rollover, with a comma after month1 (transfer set)"
TRANSFER_OF = "gpt2_month_rollover"


def build(tok):
    rows = []
    for y in YEARS:
        for a, m1 in enumerate(MONTHS):
            for b, m2 in enumerate(MONTHS):
                rows.append((tok.bos_token + f"{m1}, {y} - {m2}", [a, b, y], {"m1": 1, "comma": 2, "year": 3, "dash": 4, "m2": 5},
                             [f" {y}", f" {y + 1}", f" {y - 1}"]))
    return rows
