import pandas as pd
import numpy as np
import cvxopt
import time
from sklearn import metrics
start_time = time.time()
obj = pd.read_pickle(r'../part3_data/train_data.pickle')
obj_test = pd.read_pickle(r'../part3_data/test_data.pickle')
(data_points_test,dim1_test,dim2_test,dim3_test) = np.shape(obj_test['data'])
(data_points,dim1,dim2,dim3) = np.shape(obj['data'])
index_list = [[] for _ in range(5)]
for i in range(data_points):
    val = obj['labels'][i][0]
    index_list[val].append(i)
#we will keep -1 value for the data with less label value
# we keep track of corresponding data using (i,j) index
training_data_norm = {}
mean_models = {}
std_models = {}
alpha_models = {}
b_models = {}
y_models = {}
def gaussian_kernal(x,z):
    diff = x-z
    return np.exp(-0.001 * (diff @ diff.T ))
def gaussian_function(x):
    return np.exp(-0.001 * x * x)
def optimize_gaussian(m,kernel_matrix,y):
    p_val =np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            p_val[i,j] = 1.0 * y[i] * y[j] * kernel_matrix[i,j]
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

def normal_product(m,alpha,kernel_matrix,y,train_data_norm,x_data,train_truth,train_index):
    if train_truth :
        answer = 0
        for i in range(m):
            answer += (alpha[i,0] * y[i] * kernel_matrix[i,train_index])
        return answer
    else:
        answer = 0
        for i in range(m):
            answer += (alpha[i,0] * y[i] * gaussian_kernal(train_data_norm[i,:],x_data))
        return answer

def constant_line(m,support_vectors,alpha,kernel_matrix,y,train_data_norm):
    answer = 0
    for i in range(len(support_vectors)):
        answer = (y[i] - normal_product(m,alpha,kernel_matrix,y,train_data_norm,support_vectors[i,:],True,support_vector_indices[i]))
    return answer / len(support_vectors)


for i in range(5):
    for j in range(i+1,5):
        class_1 = (obj['data'][index_list[i]]).reshape(len(index_list[i]),-1)
        class_1.astype(np.double)
        class_2 = (obj['data'][index_list[j]]).reshape(len(index_list[j]),-1)
        class_2.astype(np.double)
        train_data = np.zeros((len(index_list[i]) + len(index_list[j]),3072),dtype=np.double)
        train_data[0:len(index_list[i]),:] = class_1
        train_data[len(index_list[i]):(len(index_list[i])+len(index_list[j])),:] = class_2
        train_data_norm = np.array(train_data,dtype= np.double)
        mean =  np.mean(train_data,axis=0)
        std = np.std(train_data,axis=0)
        mean_models[(i,j)] = mean
        std_models[(i,j)] = std
        for k in range(len(index_list[i]) + len(index_list[j])):
            train_data_norm[k,:] = (train_data[k,:] - mean)/(std)
        training_data_norm[(i,j)] = train_data_norm
        kernel_matrix =  gaussian_function(metrics.pairwise_distances(train_data_norm))
        m = len(index_list[i] + index_list[j])
        y_data = [1.0 for i in range(m)]
        for w in range(len(index_list[i])):
            y_data[w] = -1 * 1.0
        y = np.array(y_data)
        y = 1.0 * y.reshape(-1,1)
        y_models[(i,j)] = y
        alpha = np.array(optimize_gaussian(m,kernel_matrix,y))
        support_vector_indices = []
        for t in range(len(alpha)):
            if alpha[t,0] > 0.00001:
                support_vector_indices.append(t)
        support_vectors = train_data_norm[support_vector_indices]
        y_val_support_vectors = y[support_vector_indices]
        b = constant_line(m,support_vectors,alpha,kernel_matrix,y,train_data_norm)
        b_models[(i,j)] = b
        alpha_models[(i,j)] = alpha

# all the 10 models are trained now we will do the prediction part
def predict(m,alpha,kernel_matrix,y,train_data_norm,x_value,b):
    value = normal_product(m,alpha,kernel_matrix,y,train_data_norm,x_value,False,-1) + b
    return value
# now only test validation left, model trained
def predict_one(x_data):
    answer = [0 for i in range(5)]
    weights = [0.0 for i in range(5)]
    for i in range(5):
        for j in range(i+1,5):
            x_norm = (x_data - mean_models[(i,j)])/(std_models[(i,j)])
            m = len(index_list[i]) + len(index_list[j])
            value = predict(m,alpha_models[(i,j)],np.array([0.0]),y_models[(i,j)],training_data_norm[(i,j)],x_norm,b_models[(i,j)])
            if value >= 0:
                answer[j] += 1
                weights[j] += value
            else:
                answer[i] += 1
                weights[i] -= value
    max_label = 0
    for i in range(5):
        if (answer[max_label] < answer[i]) or (answer[max_label] == answer[i] and weights[max_label]<weights[i]):
            max_label = i
    return max_label
test_data = (obj_test['data']).reshape(data_points_test,-1)
test_data.astype(np.double)
def accuracy():
    correct = 0
    for i in range(data_points_test):
        if predict_one(test_data[i,:]) == obj_test['labels'][i]:
            correct += 1
    return ((100*correct)/(data_points_test))
percent_correct = accuracy()
print(percent_correct)
end_time = time.time()
print(end_time - start_time)