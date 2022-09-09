import sys
import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
file = open(sys.argv[1] +'/X.csv','r')
q2_x = np.loadtxt(file, delimiter=',',dtype='float32')
q2_x_data= np.ones((len(q2_x),3),dtype='float32')
q2_x_data[:,1:3] = q2_x
file.close()

#x_data is our generated feature matrix and y_data is the corresponding output vector with noise
x_data = np.ones((1000000,3),dtype='float32')
x_data[:,1] = np.random.normal(3.0,2.0,1000000).T
x_data[:,2] = np.random.normal(-1.0,2.0,1000000).T
theta = np.array([3,1,2],dtype='float32')
y_data = np.matmul(x_data,theta.T).T
y_data = y_data + np.random.normal(0.0,np.sqrt(2.0),1000000)

def cost_func_batch(batch_size,x_data,y_data,k,theta):
    sum_val = 0.0
    m = len(y_data)
    #sum over kth batch
    for i in range(batch_size*k, batch_size*(k+1)):
        val = (y_data[i%m] - np.matmul(x_data[i%m],theta.T))
        sum_val += val*val
    return sum_val/(2*batch_size)

def cost_derivative_batch(batch_size,x_data,y_data,k,theta,m):
    derivate = np.zeros((1,3),dtype='float32')
    for i in range(batch_size*k,batch_size*(k+1)):
        derivate += (np.matmul(x_data[i%m],theta.T) - y_data[i%m]) * x_data[i%m]
    return derivate

def stochastic_gradient_descent(x_data,y_data,eta,batch_size,iter,avg_iter,error):
    x_data_new = np.ones((len(y_data),4))
    x_data_new[:,:3] = x_data
    x_data_new[:,3] = y_data.T
    np.random.shuffle(x_data_new)
    m = len(y_data)
    theta = np.zeros((1,3))
    x_shuffle = x_data_new[:,:3]
    y_shuffle = x_data_new[:,3]
    k = 0
    total_iter = 0
    while total_iter < iter:
        iterations = 0
        theta_initial = theta.copy()
        sum_theta = np.zeros((1,3),dtype='float32')
        while iterations < avg_iter:
            sum_theta += theta
            theta = theta - (eta*cost_derivative_batch(batch_size,x_shuffle,y_shuffle,k,theta,m))/batch_size
            iterations += 1
            total_iter+= 1
            k += 1
        sum_theta = sum_theta/avg_iter
        k = k % m
        if (np.matmul(theta_initial-sum_theta,(theta_initial-sum_theta).T) <error):
            break
    return theta

def theta_variation(x_data,y_data,eta,batch_size,iter,avg_iter,error):
    x_data_new = np.ones((len(y_data),4))
    x_data_new[:,:3] = x_data
    x_data_new[:,3] = y_data.T
    np.random.shuffle(x_data_new)
    m = len(y_data)
    theta = np.zeros((1,3))
    x_shuffle = x_data_new[:,:3]
    y_shuffle = x_data_new[:,3]
    k = 0
    total_iter = 0
    theta_list = [theta]
    while total_iter < iter:
        iterations = 0
        theta_initial = theta.copy()
        sum_theta = np.zeros((1,3),dtype='float32')
        while iterations < avg_iter:
            sum_theta += theta
            theta = theta - (eta*cost_derivative_batch(batch_size,x_shuffle,y_shuffle,k,theta,m))/batch_size
            theta_list.append(theta)
            iterations += 1
            total_iter+= 1
            k += 1
        sum_theta = sum_theta/avg_iter
        k = k % m
        if (np.matmul(theta_initial-sum_theta,(theta_initial-sum_theta).T) <error):
            break
    return theta_list

theta_first = stochastic_gradient_descent(x_data,y_data,0.001,1,10000000,1000,0.0001)
# print("theta_first", theta_first)
# theta_second = stochastic_gradient_descent(x_data,y_data,0.001,100,10000000,1000,0.0001)
# print("theta_second",theta_second)
# theta_third = stochastic_gradient_descent(x_data,y_data,0.001,10000,1000,10,0.001)
# print("theta_third",theta_third)
# theta_fourth = stochastic_gradient_descent(x_data,y_data,0.001,1000000,1000,1,0.001)
# print("theta_fourth", theta_fourth)
def output(file_name,y_test):
    np.savetxt(file_name,y_test,delimiter='\n',fmt='%f')
y_test = np.matmul(q2_x_data,theta_first.T)
output("result_2.txt",y_test)

#-----------------------------------------------------------------------------------
#code for plotting the theta variation (batch size 1)
# theta_list_first = theta_variation(x_data,y_data,0.001,1,10000000,1000,0.0001)
# theta1 = []
# theta2 = []
# theta3 = []
# for theta_val in theta_list_first:
#     theta1.append(theta_val[0,0])
#     theta2.append(theta_val[0,1])
#     theta3.append(theta_val[0,2])
# axes = plt.axes(projection='3d')
# axes.scatter(theta1,theta2,theta3,color='blue',label='Batch Size 1')
# axes.legend()
# axes.set_xlabel("Theta 1")
# axes.set_ylabel("Theta 2")
# axes.set_zlabel("Theta 3")
# plt.show()
#--------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------
#code for plotting the theta variation (batch size 100)
# theta_list_second = theta_variation(x_data,y_data,0.001,100,10000000,1000,0.0001)
# theta1 = []
# theta2 = []
# theta3 = []
# for theta_val in theta_list_second:
#     theta1.append(theta_val[0,0])
#     theta2.append(theta_val[0,1])
#     theta3.append(theta_val[0,2])
# axes = plt.axes(projection='3d')
# axes.scatter(theta1,theta2,theta3,color='blue',label='Batch Size 100')
# axes.legend()
# axes.set_xlabel("Theta 1")
# axes.set_ylabel("Theta 2")
# axes.set_zlabel("Theta 3")
# plt.show()
#--------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------
#code for plotting the theta variation (batch size 10000)
# theta_list_third = theta_variation(x_data,y_data,0.001,10000,1000,10,0.001)
# theta1 = []
# theta2 = []
# theta3 = []
# for theta_val in theta_list_third:
#     theta1.append(theta_val[0,0])
#     theta2.append(theta_val[0,1])
#     theta3.append(theta_val[0,2])
# axes = plt.axes(projection='3d')
# axes.scatter(theta1,theta2,theta3,color='blue',label='Batch Size 10000')
# axes.legend()
# axes.set_xlabel("Theta 1")
# axes.set_ylabel("Theta 2")
# axes.set_zlabel("Theta 3")
# plt.show()
#--------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------
#code for plotting the theta variation (batch size 1000000)
# theta_list_fourth = theta_variation(x_data,y_data,0.001,1000000,1000,1,0.001)
# theta1 = []
# theta2 = []
# theta3 = []
# for theta_val in theta_list_fourth:
#     theta1.append(theta_val[0,0])
#     theta2.append(theta_val[0,1])
#     theta3.append(theta_val[0,2])
# axes = plt.axes(projection='3d')
# axes.scatter(theta1,theta2,theta3,color='blue',label='Batch Size 1000000')
# axes.legend()
# axes.set_xlabel("Theta 1")
# axes.set_ylabel("Theta 2")
# axes.set_zlabel("Theta 3")
# plt.show()
#--------------------------------------------------------------------------------
