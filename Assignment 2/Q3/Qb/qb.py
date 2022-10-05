from sklearn import svm
import pandas as pd
import numpy as np
import time
start_time = time.time()
obj = pd.read_pickle(r'../part3_data/train_data.pickle')
obj_test = pd.read_pickle(r'../part3_data/test_data.pickle')
(data_points_test,dim1_test,dim2_test,dim3_test) = np.shape(obj_test['data'])
(data_points,dim1,dim2,dim3) = np.shape(obj['data'])
train_data = (obj['data']).reshape(data_points,-1)
y_val = np.ravel(obj['labels'])
mean_train = np.mean(train_data,axis=0)
std_train = np.std(train_data,axis=0)
train_data_norm = np.zeros((data_points,3072),dtype=np.double)
for i in range(data_points):
    train_data_norm[i,:] = (train_data[i,:] - mean_train)/(std_train)
test_data = (obj_test['data']).reshape(data_points_test,-1)
y_val_test = np.ravel(obj_test['labels'])
test_data_norm = np.zeros((data_points_test,3072),dtype=np.double)
for i in range(data_points_test):
    test_data_norm[i,:] = (test_data[i,:] - mean_train)/(std_train)
gaussian_svm = svm.SVC(kernel='rbf',C = 1.0,gamma=0.001) # gaussian kernel
gaussian_svm.fit(train_data_norm,y_val)
y_prediction = gaussian_svm.predict(test_data_norm)
def accuracy():
    correct = 0
    for i in range(len(y_prediction)):
        if y_prediction[i] == y_val_test[i]:
            correct+=1
    return ((100* correct)/(len(y_prediction)))

percent_correct = accuracy()
print(percent_correct)
end_time = time.time()
print(end_time - start_time)