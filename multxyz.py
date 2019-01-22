


def readTinkerXYZ(fname):
    """
    return index, name, coord, tp, topology
    """
    with open(fname, "r") as f:
        text = f.readlines()
    text = [i.strip() for i in text if len(i.strip()) > 0]
    numatoms = int(text[0])

    index_l, name_l, coord_l, tp_l, topology = [], [], [], [], {}
    for i in range(1, numatoms + 1):
        index, name, x, y, z, tp, *link = text[i].split()
        index = int(index)
        x, y, z = float(x), float(y), float(z)
        link = [int(i) for i in link]

        index_l.append(index)
        name_l.append(name)
        coord_l.append([x, y, z])
        tp_l.append(tp)
        if index not in topology and link[0] != 0:
            topology[index] = []
            for l in link:
                if l not in topology[index]:
                    topology[index].append(l)
                if l not in topology:
                    topology[l] = []
                topology[l].append(index)
    # Deal with name_l
    newname_l = []
    ncount = {}
    for nm in name_l:
        if nm not in ncount:
            ncount[nm] = 1
            newname_l.append(nm)
        else:
            ncount[nm] += 1
            newname_l.append(nm + "{}".format(ncount[nm]))
    return index_l, newname_l, coord_l, tp_l, topology


def genXMLResidues(name_l, tp_l, topology):
    text = []
    text.append("<Residues>")
    text.append("""<Residue name="SYS">""")
    for ni in range(len(name_l)):
        outputstr = """<Atom name="{}" type="{}">""".format(
            name_l[ni], tp_l[ni])
        text.append(outputstr)
    for ni in range(1, len(name_l) + 1):
        if ni not in topology:
            continue
        for nj in topology[ni]:
            outputstr = """<Bond from="{}" to="{}">""".format(ni - 1, nj - 1)
            text.append(outputstr)
    text.append("</Residues>")
    return "\n".join(text)


def genPDB(name_l, coord_l, topology):
    text = []
    crdline = "HETATM{serial:>5d} {name:^4} {resname:>3} 1   1    {x:>8.3f}{y:>8.3f}{z:>8.3f}"
    for n in range(len(name_l)):
        text.append(crdline.format(
            serial=n + 1, name=name_l[n], resname="SYS", x=coord_l[n][0], y=coord_l[n][1], z=coord_l[n][2]))
    for n in range(len(name_l)):
        outputstr = "CONECT{:>5d}".format(n+1)
        if n + 1 in topology:
            for k in topology[n+1]:
                outputstr += "{:>5d}".format(k)
        text.append(outputstr)
    text.append("END")
    return "\n".join(text)