import sys
import numpy as np
from matplotlib import pyplot as plt

#data loaded
file_x = open(sys.argv[1],'r')
x_data = np.loadtxt(file_x,delimiter=',',dtype= 'float32')
file_x.close()
file_y = open(sys.argv[2],'r')
y_data = np.loadtxt(file_y,delimiter=',',dtype= 'float32')
file_y.close()
x_data_new = np.ones((len(y_data),3),dtype= 'float32')
x_data_new[:,1:] = x_data
x_data = x_data_new
def hypothesis(theta,x_data,i):
    return 1/(1 + np.exp(-1* np.matmul(x_data[i,:].T,theta)))

def diagonal_matrix(theta,x_data):
    m = len(x_data)
    diagonal_mat = np.zeros((m,m),dtype='float32')
    for i in range(m):
        diagonal_mat[i,i] = (1 - hypothesis(theta,x_data,i)) * (hypothesis(theta,x_data,i))
    return diagonal_mat
#theta has dimension 3*1 and X has dimension m*3
#the expression of hessian is Xt * D * X where D is a diagonal matrix then we will return its inverse
def hessian_inverse(theta,x_data):
    diagonal_mat = diagonal_matrix(theta,x_data)
    hessian = np.matmul(x_data.T,diagonal_mat)
    hessian = np.matmul(hessian,x_data)
    return np.linalg.pinv(hessian)

def gradient(theta,x_data,y_data):
    derivate = np.zeros((1,3),dtype='float32')
    m = len(y_data)
    for i in range(m):
        derivate += (hypothesis(theta,x_data,i) - y_data[i]) * (x_data[i,:])
    return derivate

def newton_method(x_data,y_data):
    theta = np.zeros((3,1),dtype= 'float32')
    while True:
        decrease = np.matmul(hessian_inverse(theta,x_data),gradient(theta,x_data,y_data).T)
        theta = theta - decrease
        if np.matmul(decrease.T,decrease) < 0.0001 :
            break
    return theta

theta = newton_method(x_data,y_data)
def final_hypothesis(x1):
    return (theta[0] + theta[1] * x1)/(-1*theta[2])
print(theta)
x_data_0 = x_data[y_data == 0]
x_data_1 = x_data[y_data == 1]
plt.plot(x_data_0[:,1],x_data_0[:,2],'o')
plt.plot(x_data_1[:,1],x_data_1[:,2],'^')
x = np.linspace(0,10)
plt.plot(x,final_hypothesis(x),color= 'yellow')
plt.show()

#normalization left for now