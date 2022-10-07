import pandas as pd
import numpy as np
import cvxopt
import time
import matplotlib.pyplot as plt
from sklearn import metrics
import sys
train_path = sys.argv[1] + "/train_data.pickle"
test_path = sys.argv[2] + "/test_data.pickle"
start_time = time.time()
obj = pd.read_pickle(train_path)
obj_test = pd.read_pickle(test_path)
(data_points_test,dim1_test,dim2_test,dim3_test) = np.shape(obj_test['data'])
(data_points,dim1,dim2,dim3) = np.shape(obj['data'])
index_1 = []
index_2 = []
y_data = []
y_test_val = []
for i in range(data_points):
    if obj['labels'][i] == 1:
        index_1.append(i)
        y_data.append(-1.0)
    elif obj['labels'][i] == 2:
        index_2.append(i)
        y_data.append(1.0)
index_1_test = []
index_2_test = []
for i in range(data_points_test):
    if obj_test['labels'][i] == 1:
        index_1_test.append(i)
        y_test_val.append(-1.0)
    elif obj_test['labels'][i] == 2:
        index_2_test.append(i)
        y_test_val.append(1.0)
class_1 = np.zeros((len(index_1),3072),dtype=np.double)
class_2 = np.zeros((len(index_2),3072),dtype=np.double)
class_1_test = np.zeros((len(index_1_test),3072),dtype=np.double)
class_2_test = np.zeros((len(index_2_test),3072),dtype=np.double)
#training data of class 1 and class 2 obtained
class_1 = (obj['data'][index_1]).reshape(len(index_1),-1)
class_1 = class_1.astype(np.double)
class_2 = (obj['data'][index_2]).reshape(len(index_2),-1)
class_2 = class_2.astype(np.double)
class_1_test = (obj_test['data'][index_1_test]).reshape(len(index_1_test),-1)
class_1_test = class_1_test.astype(np.double)
class_2_test= (obj_test['data'][index_2_test]).reshape(len(index_2_test),-1)
class_2_test = class_2_test.astype(np.double)
train_data = np.zeros((len(index_1)+len(index_2),3072),dtype=np.double)
train_data[0:len(index_1),:] = class_1
train_data[len(index_1):(len(index_1)+len(index_2)),:] = class_2
train_data_norm = np.array(train_data,dtype= np.double)

test_data = np.zeros((len(index_1_test) + len(index_2_test),3072),dtype=np.double)
mean_train = np.zeros((1,3072),dtype=np.double)
std_train = np.zeros((1,3072),dtype=np.double)
test_data[0:len(index_1_test),:] = class_1_test
test_data[len(index_1_test):len(index_1_test)+len(index_2_test),:] = class_2_test
for col in range(3072):
    mean = np.mean(train_data[:,col])
    std = np.std(train_data[:,col])
    mean_train[0,col] = mean
    std_train[0,col] = std
    train_data_norm[:,col] = (train_data[:,col] - mean)/(std)
def normalize(test_data):
    test_data_norm = np.zeros(np.shape(test_data),dtype=np.double)
    for i in range((np.shape(test_data)[0])):
        test_data_norm[i,:] = (test_data[i,:] - mean_train)/(std_train)
    return test_data_norm
test_data_norm = normalize(test_data)
y = np.array(y_data)
y = 1.0 * y.reshape(-1,1)
def gaussian_kernal(x,z):
    diff = x-z
    return np.exp(-0.001 * (diff @ diff.T ))
def gaussian_function(x):
    return np.exp(-0.001 * x * x)

m = len(index_1) + len(index_2)
kernel_matrix =  gaussian_function(metrics.pairwise_distances(train_data_norm))

def optimize_gaussian(m):
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

#alpha obtained here is of shape (4000,1)
alpha = np.array(optimize_gaussian(len(y_data)))
support_vector_indices = []
for i in range(len(alpha)):
    if alpha[i,0] > 0.00001:
        support_vector_indices.append(i)
support_file = open('support.txt','w')
for i in support_vector_indices:
    support_file.write(str(i))
    support_file.write('\n')
support_file.close()
print("support vectors : ",len(support_vector_indices))
#2915 support vectors coming
support_vectors = train_data_norm[support_vector_indices]
y_val_support_vectors = y[support_vector_indices]
#now we will find wTx values and b values of svm
def normal_product(m,x_data,train_truth,train_index):
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
def constant_line(m):
    answer = 0
    for i in range(len(support_vectors)):
        answer = (y[i] - normal_product(m,support_vectors[i,:],True,support_vector_indices[i]))
    return answer / len(support_vectors)
b = constant_line(len(y_data))
print(b)
def predict(m,x_value):
    value = normal_product(m,x_value,False,-1) + b
    if value >= 0 :
        return 1
    else:
        return -1
def final_train_prediction(m,test):
    correct = 0
    for i in range(test):
        value = predict(m,(test_data_norm[i,:]))
        if value == y_test_val[i]:
            correct += 1
    return (100*correct)/test

#on train data 3995 correct out of 4000
#on test data 1639 correct out of 2000
print(final_train_prediction(len(y_data),len(y_test_val)))

#now we plot the top 5 support vectors and w vectors
index_top = []
for i in range(5):
    max_val = i
    for j in range(i,len(support_vector_indices)):
        if alpha[support_vector_indices[j],0] > alpha[support_vector_indices[max_val],0]:
            max_val = j
    index_top.append(support_vector_indices[max_val])
    support_vector_indices[i],support_vector_indices[max_val] = support_vector_indices[max_val],support_vector_indices[i]
for i in range(5):
    data = np.reshape(obj['data'][2000 + index_top[i]],(32,32,3))
    plt.imshow(data,interpolation='nearest')
    plt.savefig(f"image{i+1}.png",dpi = 1000)
end_time = time.time()
print(end_time-start_time)
