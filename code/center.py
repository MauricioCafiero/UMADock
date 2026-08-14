import re
import sys

file_name = sys.argv[1]

f = open(file_name, "r")
lines = f.readlines()
f.close

frag_raw = ''
num_atoms = 0
for line in lines:
    m = re.search(r'[A-Z][a-z]?\s+\-?\d+\.\d+\s+\-?\d+\.\d+\s+\-?\d+\.\d+', line)
    if m is not None:
        frag_raw += line
        num_atoms += 1

frag_lines = frag_raw.split("\n")
_ = frag_lines.pop(-1)

if num_atoms != len(frag_lines):
    print ('Did not read file properly!')
else:
    frag = []
    atoms = []
    for line in frag_lines:
        vec = []
        parts = line.split()
        for i in range(3):
            vec.append(float(parts[i+1]))
        frag.append(vec)
        atoms.append(parts[0])
    
    center = [0.0]*3
    for row in frag:
        for i in range(3):
            center[i] += row[i]

    for i in range(3):
        center[i] = center[i]/len(frag)

    new_frag = []
    for row in frag:
        tvec = []
        for i in range(3):
            tvec.append(row[i] - center[i])
        new_frag.append(tvec)

    print('Atoms list:')
    print(atoms)
    print('========================================================')
    print('Atoms positions list:')
    for row in new_frag:
        print(row)
    print('========================================================')
    print('Check to see if the molecule is centered:')
    frag2 = new_frag

    center = [0.0]*3
    for row in frag2:
        for i in range(3):
            center[i] += row[i]

    for i in range(3):
        center[i] = center[i]/len(frag2)
    print('new_center:')
    for coord in center:
        print(f'{coord:.3f}')
