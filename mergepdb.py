import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--template", nargs="+", type=str, help="Template PDB files")
parser.add_argument("-n", "--number", nargs="+", type=int, help="Repeat number of template molecules")
parser.add_argument("-x", "--xyz", type=str, help="Coordinate XYZ file")
parser.add_argument("-o", "--output", type=str, help="Output file name")
args = parser.parse_args()

with open(args.xyz, "r") as f:
    text = f.readlines()
nxyz = int(text[0].strip())
xyz = text[2:nxyz+2]
xyz = [i.strip().split()[-3:] for i in xyz]
xyz = [[float(j) for j in i] for i in xyz]
crdline = """HETATM{serial:>5d} {name:^4} {resname:>3}  {rindex:>4d}    {x:>8.3f}{y:>8.3f}{z:>8.3f}                      {elem:>2}\n"""
assert len(args.template) == len(args.number)

natom = 1
nres = 1
with open(args.output, "w") as fout:
    for n,p in enumerate(args.number):
        with open(args.template[n], "r") as f:
            text = f.readlines()
            text = [i[6:].strip().split() for i in text if "HETATM" in i]
            pdbtemp = [[i[1],i[2],i[-1]] for i in text]
        for nrep in range(p):
            for arep in range(len(pdbtemp)):
                x, y, z = xyz[natom-1]
                a_name, res_name, elem = pdbtemp[arep]
                fout.write(crdline.format(serial=natom, name=a_name, resname=res_name, rindex=nres, x=x, y=y, z=z, elem=elem))
                natom += 1
            nres += 1
