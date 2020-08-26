from scripts import robutils as ru

import sys

if sys.argv[1] == "epe-conllup":
    ru.condoy(sys.argv[2], True, True)
elif sys.argv[1] == "conllup-starsem":
    ru.conllup_to_starsem(sys.argv[2], sherlock_train="data/sherlock_2/sp06/cdt.conllup")

elif sys.argv[1] == "adjust_starsem":
    cnt = 1
    with open(sys.argv[2]) as f:
        for line in f:
            line = line.split()
            if line:
                line[0] = "_"
                line[1] = str(cnt)
            else:
                cnt += 1
            print("\t".join(line))

elif sys.argv[1] == "fuse":
    with open(sys.argv[2]) as dt, open(sys.argv[3]) as dm:
        for t,m in zip(dt, dm):
            t = t.split()
            m = m.split()
            if len(t) > 2:
                t[8] = m[8]
            print("\t".join(t))

