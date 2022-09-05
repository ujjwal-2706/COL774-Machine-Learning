import sys
import numpy as np
from matplotlib import pyplot as plt
file = open(sys.argv[1],'r')
x_data = (np.loadtxt(file,unpack=True,dtype='float32')).T
file.close()
file = open(sys.argv[2],'r')
y_data = (np.loadtxt(file,unpack=True,dtype='str')).T
file.close()

#x_data is m*2 matrix and y_data is m*1 column vector
#Lets say Canada means 1 and Alaska means 0
x_filter_alaska = x_data[y_data == 'Alaska']
x_filter_canada = x_data[y_data == 'Canada']
mu0 = (np.matmul(np.ones((1,len(x_filter_alaska))),x_filter_alaska)) / (len(x_filter_alaska))
mu1 = (np.matmul(np.ones((1,len(x_filter_canada))),x_filter_canada)) / (len(x_filter_canada))
print(mu0)
print(mu1)
cov = (np.matmul((x_filter_alaska-mu0).T,(x_filter_alaska-mu0)) + np.matmul((x_filter_canada-mu1).T,(x_filter_canada-mu1))) / len(y_data)
print(cov)
phi = len(x_filter_canada) / (len(x_data))
print(phi)
cov0 = np.matmul((x_filter_alaska-mu0).T,(x_filter_alaska-mu0)) / len(x_filter_alaska)
cov1 = np.matmul((x_filter_canada-mu1).T,(x_filter_canada-mu1)) / len(x_filter_canada)
print(cov0)
print(cov1)

plt.plot(x_filter_alaska[:,0],x_filter_alaska[:,1],'o', label= 'Alaska')
plt.plot(x_filter_canada[:,0],x_filter_canada[:,1],'^',label= 'Canada')
plt.xlabel("x0 value")
plt.ylabel("x1 value")
plt.legend()
plt.show()