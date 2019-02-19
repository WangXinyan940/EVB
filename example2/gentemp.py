import numpy as np 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--xml", help="input xml file")
parser.add_argument("--start", help="start index")
parser.add_argument("--out", help="name of output .temp and .init files")

args = parser.parse_args()

ni = args.start
var = []
for fname in args.xml:
    with open(fname, "r") as f:
        text = f.readlines()
    out = []

    for line in out:
        if line.strip().split()[0] == "<Bond" and "from" not in line:
            pass
        elif line.strip().split()[0] == "<Angle" and "class4" not in line:
            pass
        elif line.strip().split()[0] == "<StretchBend":
            pass
        else:
            tmp = line
        out.append(tmp)
    with open(fname.split(".")[0] + ".temp", "w") as f:
        for line in out:
            f.write(line)
with open("var.txt", "w") as f:
    f.write(", ".join("%16.8f"%i for i in var))