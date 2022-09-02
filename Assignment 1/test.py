import sys
import numpy as np
print(sys.argv)
file = open(sys.argv[1],'r')
data = np.loadtxt(file,delimiter=',',dtype = 'float')
print(data)
print(data[2])
data[0] = 100
print(data)
x = np.outer(np.linspace(-3, 3, 32), np.ones(32))
print(x)
file.close()