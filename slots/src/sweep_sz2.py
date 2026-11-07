import sys
from v2 import run, BASE
T = {"S5 hmm eps=.03": {**BASE, "size_hmm": 0.03}, "S6 hmm eps=.06": {**BASE, "size_hmm": 0.06},
     "S7 shrink k=200": {**BASE, "size_k": 200}, "S8 shrink k=100": {**BASE, "size_k": 100}}
sel = sys.argv[1:]
for n, c in T.items():
    if not sel or n.split()[0] in sel: run(n, c)
