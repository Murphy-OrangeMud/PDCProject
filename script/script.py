import os
import numpy as np

"""
os.system("g++ ./src/parallel.cc -o ./src/parallel.out")
os.system("g++ ./src/parallel2.cc -o ./src/parallel2.out")
os.system("g++ ./src/serial.cc -o ./src/serial.out")

file = open("Time_Cost_Table.txt", "w")
for k in range(10): 
    for i in range(1, 13):
        res1 = os.popen("./src/parallel.out ./data/case%d.txt ./presult/p2case%d.txt" % (i, i))
        str1 = "Case %d : Parallel program with linked table " % i
        file.write(str1 + res1.read())
        res2 = os.popen("./src/parallel2.out ./data/case%d.txt ./presult/p1case%d.txt" % (i, i))
        str2 = "Case %d : Parallel program with array " % i
        file.write(str2 + res2.read())
        res3 = os.popen("./src/serial.out ./data/case%d.txt ./sresult/case%d.txt" % (i, i))
        str3 = "Case %d : Serial program " % i
        file.write(str3 + res3.read())

file.close()
"""

with open("Time_Cost_Table.txt", "r") as f:
    result = f.readlines()
    time = {"serial": np.zeros(12), "parallel1": np.zeros(12), "parallel2": np.zeros(12)}

    for row in result:
        trow = list(row.split())
        if "linked" in trow:
            time["parallel1"][int(trow[1]) - 1] += float(trow[len(trow) - 2])
        elif result.index(row) % 3 == 1:
            time["parallel2"][int(trow[1]) - 1] += float(trow[len(trow) - 2])
        else:
            time["serial"][int(trow[1]) - 1] += float(trow[len(trow) - 2])

time["serial"] /= 100
time["parallel1"] /= 100
time["parallel2"] /= 100

print(time)
