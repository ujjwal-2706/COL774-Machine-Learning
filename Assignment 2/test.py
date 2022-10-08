import numpy as np
import sys
file_1 = np.loadtxt(sys.argv[1],dtype=int)
file_2 = np.loadtxt(sys.argv[2],dtype= int)
index1 = 0
index2 = 0
vec1 = np.shape(file_1)[0]
vec2 = np.shape(file_2)[0]
count_match = 0
while index1 < vec1 and index2 < vec2:
    if file_1[index1] == file_2[index2]:
        index1+=1
        index2+=1
        count_match+=1
    elif file_1[index1] < file_2[index2]:
        index1 += 1
    else:
        index2 += 1
print(count_match)