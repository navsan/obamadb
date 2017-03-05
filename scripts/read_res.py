#!/usr/bin/env python

# reads all the .out files in current directly and parses the last line
# useful in the run-mc-* experiments

import os
import re

regex = r"proc(\d+)_t(\d+)"
pmap = {}

def extract_mean(f):
    txt = open(f,'r').read()
    dat = txt.split("\n")[-2]
    return float(dat.split(",")[0])

def parse_file(f):
    matches = re.findall(regex, f)
    if len(matches) == 0:
        return
    p = int( matches[0][0])
    t = int(matches[0][1])
    tpl = (p,t)
    if tpl not in pmap:
        pmap[tpl] = []
    pmap[tpl].append(extract_mean(f))

def mean(a):
    return float(sum(a)) / max(len(a), 1)

def main():
    files = [f for f in os.listdir('.') if os.path.isfile(f) and f.endswith(".out")]
    for f in files:
        try:
            parse_file(f)
        except:
            print "error parsing file {}".format(f)
    for k,v in sorted(pmap.iteritems()):
        print "{},{},{}".format(k[0],k[1],mean(v))

if __name__ == "__main__":
    main()
