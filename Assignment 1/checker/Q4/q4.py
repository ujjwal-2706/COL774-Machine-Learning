import sys
import numpy as np
from matplotlib import pyplot as plt
file = open(sys.argv[1] + '/X.csv','r')
x_data = np.loadtxt(file,delimiter=',',dtype='float32')
file.close()
file = open(sys.argv[1] + '/Y.csv','r')
y_data = np.loadtxt(file,delimiter=',',dtype='str')
file.close()
file = open(sys.argv[2] + '/X.csv','r')
x_test = np.loadtxt(file,delimiter=',',dtype='float32')
mean_x1 = np.mean(x_data[:,0])
mean_x2 = np.mean(x_data[:,1])
sigma_x1 = np.std(x_data[:,0])
sigma_x2 = np.std(x_data[:,1])
x_test[:,0] = (x_test[:,0] - mean_x1)/(sigma_x1)
x_test[:,1] = (x_test[:,1] - mean_x2)/(sigma_x2)
def normalize(x_data,mean_x1,mean_x2,sigma_x1,sigma_x2):
    x_data_norm = np.ones((len(x_data),2),dtype='float32')
    x_data_norm[:,0] = (x_data[:,0] - mean_x1) / (sigma_x1)
    x_data_norm[:,1] = (x_data[:,1] - mean_x2) / (sigma_x2)
    return x_data_norm
x_data_norm = normalize(x_data,mean_x1,mean_x2,sigma_x1,sigma_x2)

#x_data is m*2 matrix and y_data is m*1 column vector
#Lets say Canada means 1 and Alaska means 0
x_alaska_plot = x_data[y_data == 'Alaska']
x_canada_plot = x_data[y_data == 'Canada']
x_filter_alaska = x_data_norm[y_data == 'Alaska']
x_filter_canada = x_data_norm[y_data == 'Canada']
mu0 = (np.matmul(np.ones((1,len(x_filter_alaska))),x_filter_alaska)) / (len(x_filter_alaska))
mu1 = (np.matmul(np.ones((1,len(x_filter_canada))),x_filter_canada)) / (len(x_filter_canada))
# print("mu0 is : ",mu0)
# print("mu1 is : ",mu1)
cov = (np.matmul((x_filter_alaska-mu0).T,(x_filter_alaska-mu0)) + np.matmul((x_filter_canada-mu1).T,(x_filter_canada-mu1))) / len(y_data)
# print("cov is : ",cov)
phi = len(x_filter_canada) / (len(x_data_norm))
# print("phi is : ",phi)
cov0 = np.matmul((x_filter_alaska-mu0).T,(x_filter_alaska-mu0)) / len(x_filter_alaska)
cov1 = np.matmul((x_filter_canada-mu1).T,(x_filter_canada-mu1)) / len(x_filter_canada)
# print("cov0 is : ",cov0)
# print("cov1 is : ",cov1)

cov_inv_mu = np.matmul(np.linalg.pinv(cov),mu0.T)
def linear_separator(x1):
    return mean_x2 + ((-1* cov_inv_mu[0])/((cov_inv_mu[1]) * sigma_x1)) * (x1 - mean_x1) * sigma_x2


#in order to write the quadratic we use the coefficient and then calculate the expression
def quadratic_separtor(x0,x1):
    cov1_inv = np.linalg.pinv(cov1)
    cov0_inv = np.linalg.pinv(cov0)
    cov_diff = cov0_inv - cov1_inv
    cov_mu_diff = 2 *(np.matmul(cov1_inv,mu1.T) - np.matmul(cov0_inv,mu0.T))

    const = np.log(np.linalg.det(cov0) / np.linalg.det(cov1))
    const -= np.matmul(mu1,np.matmul(cov1_inv,mu1.T))
    const += np.matmul(mu0,np.matmul(cov0_inv,mu0.T))

    final_const = const + cov_diff[0,0] * x0 * x0 + cov_mu_diff[0] * x0
    linear_term = cov_mu_diff[1] + (cov_diff[1,0] + cov_diff[0,1]) * x0
    square_term = cov_diff[1,1]
    return square_term*x1*x1 + linear_term*x1 + final_const

def output(file_name,y_test):
    np.savetxt(file_name,y_test,delimiter='\n',fmt='%s')

y_values = (1000*quadratic_separtor(x_test[:,0],x_test[:,1])).T
y_test = np.array(['Canada' for i in range(len(y_values))])
for i in range(len(y_values)):
    if y_values[i] <= 0:
        y_test[i] = 'Alaska'
output("result_4.txt",y_test)

#------------------------------------------------------------------
#code for plotting the curves 
# x_values = np.linspace(50,200)
# plt.plot(x_alaska_plot[:,0],x_alaska_plot[:,1],'o', label= 'Alaska')
# plt.plot(x_canada_plot[:,0],x_canada_plot[:,1],'^',label= 'Canada')
# plt.plot(x_values,linear_separator(x_values), label = 'Linear Separator')
# plt.xlabel("x0 value(un-normalized)")
# plt.ylabel("x1 value(un-normalized)")
# plt.title(label = "Data Set vs Linear Separator",color='blue')
# plt.legend()
# plt.show()
#------------------------------------------------------------------

#------------------------------------------------------------------
#code for plotting quadratic separtor
# x0_val = np.linspace(-3,3,1000)
# x1_val = np.linspace(-3,3,1000)
# x0_val,x1_val = np.meshgrid(x0_val,x1_val)
# z = quadratic_separtor(x0_val,x1_val)
# plt.figure()
# plt.plot(x_filter_alaska[:,0],x_filter_alaska[:,1],'o', label= 'Alaska')
# plt.plot(x_filter_canada[:,0],x_filter_canada[:,1],'^',label= 'Canada')
# plt.contour(x0_val,x1_val,z,[0.0])
# plt.xlabel("x0 value(normalized)")
# plt.ylabel("x1 value(normalized)")
# plt.title(label = "Data Set vs Quadratic Separator",color='blue')
# plt.legend()
# plt.show()
