import sys
from v2 import run, BASE
T = {"S0 hmm eps=0": {**BASE, "size_hmm": 0.0},
     "S1 hmm eps=.002": {**BASE, "size_hmm": 0.002},
     "S2 hmm eps=.01": {**BASE, "size_hmm": 0.01},
     "S3 hmm eps=0 post-change": {**BASE, "size_hmm": 0.0, "size_post": True},
     "S4 hmm eps=.005 post-change": {**BASE, "size_hmm": 0.005, "size_post": True}}
sel = sys.argv[1:]
for n, c in T.items():
    if not sel or n.split()[0] in sel: run(n, c)
