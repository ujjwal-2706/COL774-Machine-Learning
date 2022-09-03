import sys
import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt

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
    
    # print(derivate)
    return derivate

def stochastic_gradient_descent(x_data,y_data,eta,batch_size):
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
    while True:
        iterations = 0
        theta_initial = theta.copy()
        sum_theta = np.zeros((1,3),dtype='float32')
        while iterations < 2:
            sum_theta += theta
            theta = theta - (eta*cost_derivative_batch(batch_size,x_shuffle,y_shuffle,k,theta,m))/batch_size
            iterations += 1
            print(total_iter)
            total_iter+= 1
            print(theta)
            k += 1
        sum_theta = sum_theta/2
        k = k % m
        print("found",theta_initial,sum_theta)
        print(np.matmul(theta_initial-sum_theta,(theta_initial-sum_theta).T))
        if (np.matmul(theta_initial-sum_theta,(theta_initial-sum_theta).T) <0.5):
            break
    return theta
# theta_first = stochastic_gradient_descent(x_data,y_data,0.001,1)
# print(theta_first)
# theta_second = stochastic_gradient_descent(x_data,y_data,0.001,100)
# print(theta_second)
# theta_third = stochastic_gradient_descent(x_data,y_data,0.1,10000)
# print(theta_third)
# theta_fourth = stochastic_gradient_descent(x_data,y_data,0.1,1000000)
# print(theta_fourth)