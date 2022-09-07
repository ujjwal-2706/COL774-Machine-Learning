import sys
import numpy as np
from matplotlib import pyplot as plt

#data loaded
file_x = open((sys.argv[1]) + "/X.csv",'r')
x_data = np.loadtxt(file_x,delimiter=',',dtype= 'float32')
file_x.close()
file_y = open((sys.argv[1]) + "/Y.csv",'r')
y_data = np.loadtxt(file_y,delimiter=',',dtype= 'float32')
file_y.close()
x_data_new = np.ones((len(y_data),3),dtype= 'float32')
x_data_new[:,1:] = x_data
x_data = x_data_new

file_test = open((sys.argv[2]) + "/X.csv",'r')
x_test = np.loadtxt(file_test,delimiter=',',dtype='float32')
x_test_new = np.ones((len(x_test),3),dtype='float32')
x_test_new[:,1:] = x_test
x_test = x_test_new
file_test.close()

mean_x1 = np.mean(x_data[:,1])
mean_x2 = np.mean(x_data[:,2])
sigma_x1 = np.std(x_data[:,1])
sigma_x2 = np.std(x_data[:,2])

#data normalized
def normalize(x_data,mean_x1,mean_x2,sigma_x1,sigma_x2):
    x_data_norm = np.array(x_data)
    x_data_norm[:,1] = (x_data[:,1] - mean_x1) / sigma_x1
    x_data_norm[:,2] = (x_data[:,2] - mean_x2) / sigma_x2
    return x_data_norm
x_data_norm = normalize(x_data,mean_x1,mean_x2,sigma_x1,sigma_x2)
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

def newton_method(x_data,y_data,stopping):
    theta = np.zeros((3,1),dtype= 'float32')
    while True:
        decrease = np.matmul(hessian_inverse(theta,x_data),gradient(theta,x_data,y_data).T)
        theta = theta - decrease
        if np.matmul(decrease.T,decrease) < stopping :
            break
    return theta

theta = newton_method(x_data_norm,y_data,0.0001)
print(theta)
def final_hypothesis(x1,mean_x1,mean_x2,sigma_x1,sigma_x2):
    return mean_x2 + ((theta[0] + theta[1] * ((x1-mean_x1)/sigma_x1))/(-1*theta[2])) * (sigma_x2)


def output_file(filename,y_test):
    file = open(filename,'w')
    for i in range(len(y_data)):
        if y_data[i] > 0.0 :
            file.write("1\n")
        else:
            file.write("0\n")
    file.close()
x_test[:,1] = (x_test[:,1] - mean_x1)/sigma_x1
x_test[:,2] = (x_test[:,2] - mean_x2)/sigma_x2
y_test = np.matmul(x_test,theta)
output_file("result_3.txt",y_test)



#-----------------------------------------------
#Plotting code 
# x_data_0 = x_data[y_data == 0]
# x_data_1 = x_data[y_data == 1]
# plt.plot(x_data_0[:,1],x_data_0[:,2],'o')
# plt.plot(x_data_1[:,1],x_data_1[:,2],'^')
# x = np.linspace(0,10)
# plt.plot(x,final_hypothesis(x,mean_x1,mean_x2,sigma_x1,sigma_x2),color= 'red')
# plt.xlabel("x1-value")
# plt.ylabel("x2-value")
# plt.title(label = "Separator for Logistic Regression",color = "blue")
# plt.show()
#----------------------------------------------