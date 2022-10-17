from matplotlib import pyplot as plt
import numpy as np
import sys
from sklearn.ensemble import RandomForestClassifier

file_train = open('../part1/train.csv')
file_train.readline()
train_load  = np.loadtxt(file_train,delimiter=',',dtype='str')
file_train.close()

file_test = open('../part1/test.csv')
file_test.readline()
test_load = np.loadtxt(file_test,delimiter=',',dtype='str')
file_test.close()

file_validation = open('../part1/val.csv')
file_validation.readline()
validation_load = np.loadtxt(file_validation,delimiter=',',dtype='str')
file_validation.close()

#this function will remove all the data points with ? as well as the first column 
def filter_data(data_points):
    result = []
    row,col = np.shape(data_points)
    for i in range(row):
        consider = True
        for index in range(1,col-1):
            if data_points[i,index] == '?':
                consider = False
                break
        if consider :
            for index in range(1, col):
                result.append(int(data_points[i,index]))
    new_rows = len(result) // (col-1)
    answer = np.array(result)
    answer = answer.reshape((new_rows,col-1))
    x_value = np.zeros((new_rows,col-2))
    y_value = np.zeros((new_rows,1))
    x_value = answer[:,:(col-2)]
    y_value = answer[:,col-2]
    return x_value,y_value
#training data loaded and filtered
x_train,y_train = filter_data(train_load)
x_test,y_test = filter_data(test_load)
x_validation,y_validation = filter_data(validation_load)
def accuracy(original, prediction):
    total, = np.shape(original)
    correct = 0
    for i in range(total):
        if original[i] == prediction[i]:
            correct += 1
    return (100*correct)/(total)

random_classifier = RandomForestClassifier(oob_score=True)
random_classifier.fit(x_train,y_train)
y_train_prediction = random_classifier.predict(x_train)
y_test_prediction = random_classifier.predict(x_test)
y_valid_prediction = random_classifier.predict(x_validation)

print(f"Train accuracy is : {accuracy(y_train,y_train_prediction)}")
print(f"Test accuracy is : {accuracy(y_test,y_test_prediction)}")
print(f"Validation accuracy is : {accuracy(y_validation,y_valid_prediction)}")
print(f"Out of bag accuracy is : {random_classifier.oob_score_}")
