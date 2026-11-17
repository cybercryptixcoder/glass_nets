"""Task: does the year roll over? "March 1999 - January" -> GPT-2 small continues with 2000 (the range wraps into the
next year) or 1999 (it does not). The model has to compare two months on the yearly cycle and turn the comparison
into "the same year" or "the next year" as a year token. Month circles (Engels et al. 2024) and successor heads
(Gould et al. 2023) are known; how the comparison is made and turned into a year is not (searched 2026-11-17).

Every month pair x every year is a prompt (a full product), so the judge's interchange and partial tests, which mix
the inputs of two prompts, always produce another prompt of the task: the reader can see what the network does there.
"""
ID = "gpt2_month_rollover"
MODEL = "gpt2"
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
YEARS = list(range(1980, 2000))
ANSWERS = ["same year", "next year", "previous year"]   # per prompt: the tokens " <year>", " <year + 1>", " <year - 1>"
X_NAMES = ["m1", "m2", "year"]
SPLIT_BY = "year"
TEXT = ('Each prompt is "<month1> <year> - <month2>", a date range whose end year is to be predicted, for example '
        '"March 1987 - January". x[:, 0] = month1 and x[:, 1] = month2 as indices 0..11 (0 = January ... 11 = December); '
        "x[:, 2] = the year (1980..1999). Every pair of months occurs with every year. Token positions: 0 <|endoftext|>, "
        '1 month1, 2 the year, 3 " -", 4 month2 (the last token). Roles: m1 = 1, year = 2, dash = 3, m2 = end = 4. '
        "The network's output is three logits at the end position: of the year token itself (\"same year\"), of the next "
        "year (\"next year\") and of the year before (\"previous year\"); these tokens differ from prompt to prompt.")
CONTEXT = "The prompts are date ranges; the answer is the year in which the range ends."
ANSWER = ("GPT-2 small mostly continues with the same year when month2 comes after month1 in the calendar, and with the "
          "next year when it comes before; the probability of the next year grows with the forward distance from month1 to month2.")


def build(tok):
    rows = []
    for y in YEARS:
        for a, m1 in enumerate(MONTHS):
            for b, m2 in enumerate(MONTHS):
                rows.append((tok.bos_token + f"{m1} {y} - {m2}", [a, b, y], {"m1": 1, "year": 2, "dash": 3, "m2": 4},
                             [f" {y}", f" {y + 1}", f" {y - 1}"]))
    return rows
