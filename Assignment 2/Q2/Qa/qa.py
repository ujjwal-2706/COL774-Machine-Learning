import pandas as pd
import numpy as np
import cvxopt
import time
start_time = time.time()
obj = pd.read_pickle(r'../part2_data/train_data.pickle')
obj_test = pd.read_pickle(r'../part2_data/test_data.pickle')
(data_points_test,dim1_test,dim2_test,dim3_test) = np.shape(obj_test['data'])
(data_points,dim1,dim2,dim3) = np.shape(obj['data'])
class_1 = np.zeros((data_points//5,3072),dtype=np.double)
class_2 = np.zeros((data_points//5,3072),dtype=np.double)
class_1_test = np.zeros((data_points_test//5,3072),dtype=np.double)
class_2_test = np.zeros((data_points_test//5,3072),dtype=np.double)
#training data of class 1 and class 2 obtained
class_1 = (obj['data'][data_points//5:(2*data_points)//5]).reshape(data_points//5,-1)
class_1 = class_1.astype(np.double)
class_2 = (obj['data'][(2*data_points//5):(3*data_points//5)]).reshape(data_points//5,-1)
class_2 = class_2.astype(np.double)
class_1_test = (obj_test['data'][data_points_test//5:(2*data_points_test)//5]).reshape(data_points_test//5,-1)
class_1_test = class_1_test.astype(np.double)
class_2_test= (obj_test['data'][(2*data_points_test//5):(3*data_points_test//5)]).reshape(data_points_test//5,-1)
class_2_test = class_2_test.astype(np.double)
train_data = np.zeros(((2*data_points//5),3072),dtype=np.double)
train_data[0:(data_points//5),:] = class_1
train_data[(data_points//5):(2*data_points//5),:] = class_2
train_data_norm = np.array(train_data,dtype= np.double)

test_data = np.zeros((2*data_points_test//5,3072),dtype=np.double)
mean_train = np.zeros((1,3072),dtype=np.double)
std_train = np.zeros((1,3072),dtype=np.double)
for col in range(3072):
    mean = np.mean(train_data[:,col])
    std = np.std(train_data[:,col])
    mean_train[0,col] = mean
    std_train[0,col] = std
    train_data_norm[:,col] = (train_data[:,col] - mean)/(std)
def normalize(test_data):
    test_data_norm = np.zeros((2*data_points_test//5,3072),dtype=np.double)
    for i in range(2*data_points_test//5):
        test_data_norm[i,:] = (test_data[i,:] - mean_train)/(std_train)
    return test_data_norm
test_data_norm = normalize(test_data)
y_test_val = [1.0 for i in range(2*data_points_test//5)]
for i in range(data_points_test//5):
    y_test_val[i] = -1.0
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

#alpha obtained here is of shape (4000,1)
alpha = np.array(optimize_linear(4000))
support_vector_indices = []
for i in range(len(alpha)):
    if alpha[i,0] > 0.00001:
        support_vector_indices.append(i)
    else:
        alpha[i,0] = 0

#1398 support vectors coming
support_vectors = train_data_norm[support_vector_indices]
y_val_support_vectors = y[support_vector_indices]
#now we will find w and b values of svm
def normal_line(m):
    new_data = np.array(train_data_norm)
    new_data[0:m//2] = -1.0 * train_data_norm[0:m//2]
    return (alpha.T @ new_data)
w_transpose = normal_line(4000)
def constant_line():
    all_val = y_val_support_vectors - (support_vectors @ w_transpose.T)
    num_support_vectors = len(all_val)
    row_vector = np.ones((1,num_support_vectors))
    return (row_vector @ all_val) / num_support_vectors
b = constant_line()
def predict(x_value):
    value = (w_transpose @ x_value.T) + b
    if value >= 0 :
        return 1
    else:
        return -1
print(np.shape(w_transpose))
# print(np.shape(train_data_norm[0,0]))
def final_train_prediction(m):
    correct = 0
    for i in range(m):
        value = predict((test_data_norm[i,:]))
        if value == y_test_val[i]:
            correct += 1
    return correct

#on train data only 5 points wrongly classified
print(final_train_prediction(2000))

# alpha_gaussian = np.array(optimize_gaussian(4000))
# print(alpha_gaussian)
end_time = time.time()
print(end_time-start_time)