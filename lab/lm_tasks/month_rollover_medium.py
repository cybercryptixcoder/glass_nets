"""Replication target: month rollover in GPT-2 MEDIUM (24 layers), years 1985-1989 (720 prompts). Same prompts and
answers as month_rollover.py; used to test whether the line-not-circle finding holds in a larger model."""
import importlib.util, pathlib
_s = importlib.util.spec_from_file_location("_mr", pathlib.Path(__file__).with_name("month_rollover.py"))
_mr = importlib.util.module_from_spec(_s); _s.loader.exec_module(_mr)
ID = "gpt2medium_month_rollover"
MODEL = "gpt2-medium"
ANSWERS, X_NAMES, SPLIT_BY, MONTHS = _mr.ANSWERS, _mr.X_NAMES, _mr.SPLIT_BY, _mr.MONTHS
YEARS = list(range(1985, 1990))
TEXT, CONTEXT, ANSWER = _mr.TEXT.replace("1980..1999", "1985..1989").replace("GPT-2 small", "GPT-2 medium"), _mr.CONTEXT, _mr.ANSWER
NAME = "month rollover in GPT-2 medium (replication)"


def build(tok):
    _mr.YEARS = YEARS
    return _mr.build(tok)
