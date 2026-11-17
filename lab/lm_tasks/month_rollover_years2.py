"""Transfer task for gpt2_month_rollover: the same prompts with the years 1950-1979, which the month-rollover task
never contains. Used by lab/lm_transfer.py to test a verified construction on prompts it was never fitted on."""
import importlib.util, pathlib
_s = importlib.util.spec_from_file_location("_mr", pathlib.Path(__file__).with_name("month_rollover.py"))
_mr = importlib.util.module_from_spec(_s); _s.loader.exec_module(_mr)
ID = "gpt2_month_rollover_y1950"
MODEL, ANSWERS, X_NAMES, SPLIT_BY, MONTHS = _mr.MODEL, _mr.ANSWERS, _mr.X_NAMES, _mr.SPLIT_BY, _mr.MONTHS
YEARS = list(range(1950, 1980))
TEXT, CONTEXT, ANSWER = _mr.TEXT.replace("1980..1999", "1950..1979"), _mr.CONTEXT, _mr.ANSWER
NAME = "month rollover, years 1950-1979 (transfer set)"
TRANSFER_OF = "gpt2_month_rollover"


def build(tok):
    _mr.YEARS = YEARS
    return _mr.build(tok)
