import pandas as pd
import numpy as np
import cvxopt
import time
start_time = time.time()
obj = pd.read_pickle(r'../part2_data/train_data.pickle')
(data_points,dim1,dim2,dim3) = np.shape(obj['data'])
class_1 = np.zeros((data_points//5,3072),dtype=np.double)
class_2 = np.zeros((data_points//5,3072),dtype=np.double)

#training data of class 1 and class 2 obtained
class_1 = (obj['data'][data_points//5:(2*data_points)//5]).reshape(2000,-1)
class_1 = class_1.astype(np.double)
class_2 = (obj['data'][(2*data_points//5):(3*data_points//5)]).reshape(2000,-1)
class_2 = class_2.astype(np.double)
train_data = np.zeros(((2*data_points//5),3072),dtype=np.double)

train_data[0:(data_points//5),:] = class_1
train_data[(data_points//5):(2*data_points//5),:] = class_2
train_data_norm = np.array(train_data,dtype= np.double)
for col in range(3072):
    mean = np.mean(train_data[:,col])
    std = np.std(train_data[:,col])
    train_data_norm[:,col] = (train_data[:,col] - mean)/(std)
y_data = []
for i in range((2*data_points//5)):
    if i <(data_points//5):
        y_data.append(-1.0 * 1.0)
    else:
        y_data.append(1.0 * 1.0)
y = np.array(y_data)
y = 1.0 * y.reshape(-1,1) 
def optimize_linear(m):
    new_train = np.array(train_data_norm)
    new_train[0:m//2,:] = -1.0 * new_train[0:m//2,:]
    p_val =np.matmul(new_train,new_train.T)
    P = cvxopt.matrix(p_val)
    q = cvxopt.matrix(-np.ones((m, 1)))
    g_val = 1.0 * np.zeros((2*m,m))
    g_val[0:m,:] = np.identity(m)
    g_val[m:2*m,:] = -1.0 * np.identity(m)
    h_val = np.zeros((2*m))
    h_val[:m] = 1.0 * np.ones(m)
    G = cvxopt.matrix(g_val)
    h = cvxopt.matrix(h_val)
    A = cvxopt.matrix(y.reshape(1, -1))
    b = cvxopt.matrix(np.array([0.0]))
    sol = cvxopt.solvers.qp(P,q,G,h,A,b)
    return sol['x']

def gaussian_kernal(x,z):
    diff = x-z
    return np.exp(-0.001 * (diff @ diff.T ))
def optimize_gaussian(m):
    p_val =np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            p_val[i,j] = 1.0 * y[i] * y[j] * gaussian_kernal(train_data_norm[i,:],train_data_norm[j,:])
    P = cvxopt.matrix(p_val)
    q = cvxopt.matrix(-np.ones((m, 1)))
    g_val = 1.0 * np.zeros((2*m,m))
    g_val[0:m,:] = np.identity(m)
    g_val[m:2*m,:] = -1.0 * np.identity(m)
    h_val = np.zeros((2*m))
    h_val[:m] = 1.0 * np.ones(m)
    G = cvxopt.matrix(g_val)
    h = cvxopt.matrix(h_val)
    A = cvxopt.matrix(y.reshape(1, -1))
    b = cvxopt.matrix(np.array([0.0]))
    sol = cvxopt.solvers.qp(P,q,G,h,A,b)
    return sol['x']
# alpha = np.array(optimize_linear(4000))
# print(alpha)
# alpha_gaussian = np.array(optimize_gaussian(4000))
# print(alpha_gaussian)
end_time = time.time()
print(end_time-start_time)