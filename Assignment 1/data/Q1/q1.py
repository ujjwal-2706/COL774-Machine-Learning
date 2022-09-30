from cProfile import label
import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
import sys

# File reading part of data set
linearX_ques1 = open((sys.argv[1]) + "/X.csv",'r') 
x_data = np.loadtxt(linearX_ques1,delimiter=',',dtype='float32')
linearX_ques1.close()
linearY_ques1 = open((sys.argv[1]) + "/Y.csv",'r') 
y_data = np.loadtxt(linearY_ques1,delimiter=',',dtype='float32')
linearY_ques1.close()

#test data loaded
linear_test_X = open((sys.argv[2]) + "/X.csv",'r')
x_test_data = np.loadtxt(linear_test_X,delimiter=',',dtype='float32')
linear_test_X.close()

#Mean and standard deviation calculated and x_data normalized
mean = np.mean(x_data)
sigma = np.std(x_data)
def normalize(x_data,mean,sigma):
    x_data_norm = np.array(x_data)
    for i in range(len(x_data)):
        x_data_norm[i] = (x_data_norm[i]-mean)/sigma
    return x_data_norm
x_data_norm = normalize(x_data,mean,sigma)

#Cost function 
def cost_function(theta0,theta1,x_data_norm,y_data):
    sum_value = 0.0
    m = len(x_data_norm)
    for i in range(m):
        hypo_val = theta0 + theta1*x_data_norm[i]
        sum_value += (y_data[i] - hypo_val) * (y_data[i] - hypo_val)
    return (sum_value/(2*m))

#Cost function derivate written for gradient descent
def cost_derivate(theta0,theta1,x_data,y_data):
    result1 = 0
    result0 = 0
    result = np.array([0.0,0.0])
    for i in range(len(x_data)):
        result0 += (theta0 + theta1*x_data[i] - y_data[i])
        result1 += ((theta0 + theta1*x_data[i] - y_data[i]) * x_data[i])
    result[0] = result0
    result[1] = result1
    return result


# gradient descent implemented and here eta is our learning rate
def batch_gradient_descent(x_data,y_data,eta,stopping):
    theta0 = 0.0
    theta1 = 0.0
    m = len(x_data)
    while True:
        derivate = cost_derivate(theta0,theta1,x_data,y_data)
        temp0 = theta0 - ((eta*derivate[0])/m)
        temp1 = theta1 - ((eta*derivate[1])/m)
        if abs(temp0 - theta0) < stopping and abs(temp1 - theta1) < stopping:
            break
        theta0 = temp0
        theta1 = temp1
    return np.array([theta0,theta1])

#this function will give the different cost function values with changing theta
def changing_theta_values(x_data,y_data,eta,stopping):
    theta0 = 0.0
    theta1 = 0.0
    m = len(x_data)
    theta0_list = [theta0]
    theta1_list = [theta1]
    cost_values = [cost_function(theta0,theta1,x_data,y_data)]
    while True:
        derivate = cost_derivate(theta0,theta1,x_data,y_data)
        temp0 = theta0 - ((eta*derivate[0])/m)
        temp1 = theta1 - ((eta*derivate[1])/m)
        if abs(temp0 - theta0) < stopping and abs(temp1 - theta1) < stopping:
            break
        theta0 = temp0
        theta1 = temp1
        theta0_list.append(theta0)
        theta1_list.append(theta1)
        cost_values.append(cost_function(theta0,theta1,x_data,y_data))
    return (theta0_list,theta1_list,cost_values)

#function for outputing y_i for a given x_i
def linear_regression(theta0,theta1,x_value,mean,sigma):
    return theta0 + (theta1*((x_value-mean)/sigma))


#graph sketching part
def linear_graph(theta0,theta1,x_data,mean,sigma):
    data_output =  np.zeros(len(x_data),dtype='float')
    for i in range(len(x_data)):
        x_val = (x_data[i]-mean)/sigma
        data_output[i] = (theta0 + theta1*x_val)
    return data_output

def output_file(file_name,y_test_value):
    np.savetxt(file_name,y_test_value,delimiter='\n',fmt="%f")

# training to get theta and outputing values in result_1.txt
theta = batch_gradient_descent(x_data_norm,y_data,0.01,0.000001)
# print(theta)
y_test_value = linear_graph(theta[0],theta[1],x_test_data,mean,sigma)
output_file("result_1.txt",y_test_value)


#----------------------------------------------------------------------
#code to generate 2d plot of given trained data (please uncomment before running)
# hypothesis_output = linear_graph(theta[0],theta[1],x_data,mean,sigma)
# plt.plot(x_data,y_data,'o', label = "Initial dataset")
# plt.plot(x_data,hypothesis_output,label= "Trained model")
# plt.xlabel("x-value(un-normalized)")
# plt.ylabel("y-value(wine density)")
# plt.legend()
# plt.show()
#---------------------------------------------------------------------


#---------------------------------------------------------------------
#code to generate mesh plot of given data
# mesh_x = np.outer(np.linspace(-1.0,2.0,48,dtype='float'),np.ones(48))
# mesh_y = np.outer(np.linspace(-1.0,2.0,48,dtype='float'),np.ones(48)).T
# mesh_z = cost_function(mesh_x,mesh_y,x_data_norm,y_data)
# figure = plt.figure(figsize= (15,15))
# (change_theta0,change_theta1,cost_values) = changing_theta_values(x_data_norm,y_data,0.01,0.000001) 
# axes = plt.axes(projection ='3d')
# axes.set_title("Mesh Plot For Cost Function")
# axes.set_xlabel('Theta0')
# axes.set_ylabel('Theta1')
# axes.set_zlabel("Cost Value")
# axes.plot_surface(mesh_x, mesh_y, mesh_z,alpha= 0.5)
# axes.scatter(change_theta0,change_theta1,cost_values,color = 'orange',label = "Iterations Changing")
# plt.legend()
# plt.show()
#---------------------------------------------------------------------


#---------------------------------------------------------------------
#code to generate the contour plot
# axes = plt.axes()
# mesh_x = np.outer(np.linspace(0.0,2.0,48,dtype='float'),np.ones(48))
# mesh_y = np.outer(np.linspace(-1.0,2.0,48,dtype='float'),np.ones(48)).T
# mesh_z = cost_function(mesh_x,mesh_y,x_data_norm,y_data)
# (change_theta0,change_theta1,cost_values) = changing_theta_values(x_data_norm,y_data,0.01,0.000001)
# cset = plt.contour(mesh_x,mesh_y,mesh_z)
# axes.scatter(change_theta0,change_theta1,cost_values,color = 'orange',label = "Iterations Changing")
# plt.xlabel("theta0 - value")
# plt.ylabel("theta1 - value")
# plt.show()
#---------------------------------------------------------------------
