import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
import sys

# File reading part of data set
linearX_ques1 = open(sys.argv[1],'r')
x_data = np.loadtxt(linearX_ques1,delimiter=',',dtype='float32')
linearX_ques1.close()
linearY_ques1 = open(sys.argv[2],'r')
y_data = np.loadtxt(linearY_ques1,delimiter=',',dtype='float32')
linearY_ques1.close()

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
def batch_gradient_descent(x_data,y_data,eta):
    theta0 = 0.0
    theta1 = 0.0
    m = len(x_data)
    while True:
        derivate = cost_derivate(theta0,theta1,x_data,y_data)
        temp0 = theta0 - ((eta*derivate[0])/m)
        temp1 = theta1 - ((eta*derivate[1])/m)
        if abs(temp0 - theta0) < 0.00000001 and abs(temp1 - theta1) < 0.00000001:
            break
        theta0 = temp0
        theta1 = temp1
    return np.array([theta0,theta1])

def linear_regression(theta0,theta1,x_value,mean,sigma):
    return theta0 + (theta1*((x_value-mean)/sigma))


#graph sketching part
def linear_graph(theta0,theta1,x_data,mean,sigma):
    data_output =  np.zeros(len(x_data),dtype='float')
    for i in range(len(x_data)):
        x_val = (x_data[i]-mean)/sigma
        data_output[i] = (theta0 + theta1*x_val)
    return data_output

# create a 2D set of x values for mesh plot
theta = batch_gradient_descent(x_data_norm,y_data,0.01)
print(theta)

mesh_x = np.outer(np.linspace(-5.0,5.0,48,dtype='float'),np.ones(48))
mesh_y = np.outer(np.linspace(-5.0,5.0,48,dtype='float'),np.ones(48)).T
mesh_z = cost_function(mesh_x,mesh_y,x_data_norm,y_data)

hypothesis_output = linear_graph(theta[0],theta[1],x_data,mean,sigma)
# plt.plot(x_data,y_data,'o')
# plt.plot(x_data,hypothesis_output)

# axes = plt.axes(projection ='3d')
# axes.set_title("Mesh Plot For Cost Function")
# axes.set_xlabel('Theta0')
# axes.set_ylabel('Theta1')
# axes.plot_surface(mesh_x, mesh_y, mesh_z)
# # axes.scatter(1,2,-5,color = 'red') # this is used to plot scattered points
axes = plt.axes()
cset = plt.contour(mesh_x,mesh_y,mesh_z)
plt.show()

