import pandas as pd
import numpy as np
import cvxopt
obj = pd.read_pickle(r'../part2_data/train_data.pickle')
class_1 = np.zeros((2000,3072),dtype=np.double)
class_2 = np.zeros((2000,3072),dtype=np.double)

#training data of class 1 and class 2 obtained
class_1 = (obj['data'][2000:4000]).reshape(2000,-1)
class_1 = class_1.astype(np.double)
class_2 = (obj['data'][4000:6000]).reshape(2000,-1)
class_2 = class_2.astype(np.double)
train_data = np.zeros((4000,3072),dtype=np.double)

train_data[0:2000,:] = class_1
train_data[2000:4000,:] = class_2
train_data_norm = np.array(train_data,dtype= np.double)
for col in range(3072):
    mean = np.mean(train_data[:,col])
    std = np.std(train_data[:,col])
    train_data_norm[:,col] = (train_data[:,col] - mean)/(std)
y_data = []
for i in range(4000):
    if i <2000:
        y_data.append(-1.0 * 1.0)
    else:
        y_data.append(1.0 * 1.0)
y = np.array(y_data)
y = y.reshape(-1,1) * 1.
def optimize(m):
    new_train = np.array(train_data_norm)
    new_train[0:m//2,:] = -1.0 * new_train[0:m//2,:]
    p_val =np.matmul(new_train,new_train.T)
    P = cvxopt.matrix(p_val)
    q = cvxopt.matrix(-np.ones((m, 1)))
    g_val = 1.0 * np.zeros((2*m,m))
    g_val[0:m,:] = np.identity(m)
    g_val[m:2*m,:] = -1.0 * np.identity(m)
    G = cvxopt.matrix(g_val)
    h = cvxopt.matrix(np.hstack((np.ones(m) * 1.0, np.zeros(m))))
    A = cvxopt.matrix(y.reshape(1, -1))
    b = cvxopt.matrix(np.array([0.0]))
    sol = cvxopt.solvers.qp(P,q,G,h,A,b)
    return sol['x']
alpha = np.array(optimize(4000))
print(alpha)