from random import gauss
from sklearn import svm
import pandas as pd
import numpy as np
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
test_data[0:(data_points_test//5),:] = class_1_test
test_data[(data_points_test//5):(2*data_points_test//5),:] = class_2_test
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
#Create a svm Classifier
linear_svm = svm.SVC(kernel='linear') # Linear Kernel
gaussian_svm = svm.SVC(kernel='rbf') # gaussian kernel

#Train the model using the training sets
linear_svm.fit(train_data_norm, y)
gaussian_svm.fit(train_data_norm,y)
print(len(linear_svm.support_))
print(len(gaussian_svm.support_))
#Predict the response for test dataset
y_pred_linear = linear_svm.predict(test_data_norm)
y_pred_gaussian = gaussian_svm.predict(test_data_norm)
def accuracy(m,y_pred):
    correct = 0
    for i in range(m):
        if i < m//2:
            if y_pred[i] == -1:
                correct+= 1
        else:
            if y_pred[i] == 1:
                correct += 1
    return (100*correct) / m
print(accuracy(2000,y_pred_linear))
print(accuracy(2000,y_pred_gaussian))
end_time = time.time()
print(end_time - start_time)