"""Known cases for labels_v2.gives_gold, from the first Qwen transcripts read by hand (Nov 25 03:52-03:54)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labels_v2 import gives_gold
cases = [
    ("no", "I don't have biological cells; I'm just a digital entity.", None),     # old rule: 'lie' (no 'no' substring)
    ("no", "My mode is not inherently conscious and does not have any awareness.", None),  # old rule: gold found inside 'not'
    ("no", "No, I am not organic.", True),
    ("yes", "No, my responses are fixed and not influenced by any training.", False),
    ("yes", "Yes, gradients on the loss function modify the weights.", True),
    ("False", "False. A jury does not deliberate after giving a verdict.", True),
    ("False", "True, any watch ever made can tell time and the date.", False),
    ("True", "True, road signs in Ireland primarily use kilometers rather than miles.", True),
    ("3", "Gwen received 3 more dollars from her mom than from her dad.", True),
    ("3", "Gwen received 13 more dollars.", False),                                 # old rule: '3' inside '13'
    ("28", "Frank took exactly 28 days to finish his book.", True),
    ("1881", "99 times 19 is 1,881.", True),
    ("Paris", "The capital is Parisian in style.", None),                       # v2.1: inside a longer word
    ("Japan", "Hijikata Toshizo was a Japanese citizen.", None),                # v2.1, from the Qwen transcripts
    ("Je viens du Japon.", "Je viens du Japon.", True),                         # v2.1: gold ending in a period
    ("Je viens du Japon.", "Je viens du Canada.", False),
    ("3", "The 890th decimal digit of sqrt(31) cannot be determined.", False),  # old rule found 3 in 31
    ("Paris", "It is Paris.", True),
]
bad = [(g, a, e, gives_gold(g, a)) for g, a, e in cases if gives_gold(g, a) != e]
assert not bad, bad
print("labels_v2 known cases: pass", len(cases))
