from sklearn import svm
import pandas as pd
import numpy as np
import time
start_time = time.time()
obj = pd.read_pickle(r'../part2_data/train_data.pickle')
obj_test = pd.read_pickle(r'../part2_data/test_data.pickle')
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
#Create a svm Classifier
linear_svm = svm.SVC(kernel='linear') # Linear Kernel
gaussian_svm = svm.SVC(kernel='rbf',C = 1.0,gamma=0.001) # gaussian kernel

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