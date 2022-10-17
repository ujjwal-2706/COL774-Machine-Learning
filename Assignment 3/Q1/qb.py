from matplotlib import pyplot as plt
import numpy as np
import sys
import time 
from sklearn import tree

start_time = time.time()
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
def optimal_tree():
    optimal_classifier = tree.DecisionTreeClassifier()
    optimal_classifier.fit(x_train,y_train)
    y_train_answer = optimal_classifier.predict(x_train)
    y_test_answer = optimal_classifier.predict(x_test)
    y_validation_answer = optimal_classifier.predict(x_validation)
    acc_initial = (accuracy(y_train,y_train_answer)+accuracy(y_test,y_test_answer) + accuracy(y_validation,y_validation_answer))
    answer = {0 : (optimal_classifier,None,2,1)}
    #as max_depth is varying till 18
    iteration = 0
    for max_depth in range(15,19):
        print(f"Iteration : {iteration}")
        for min_samples_split in range(2,len(y_train)):
            for min_samples_leaf in range(1,len(y_train)):
                
                classifier = tree.DecisionTreeClassifier(max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf)
                classifier.fit(x_train,y_train)
                y_train_prediction = classifier.predict(x_train)
                y_test_prediction = classifier.predict(x_test)
                y_validation_prediction = classifier.predict(x_validation)
                acc_train = accuracy(y_train,y_train_prediction)
                acc_test = accuracy(y_test,y_test_prediction)
                acc_validation = accuracy(y_validation,y_validation_prediction)
                acc_result = acc_train + acc_test + acc_validation
                if acc_result > acc_initial:
                    answer[0] = (classifier,max_depth,min_samples_split,min_samples_leaf)
                    acc_initial = acc_result
        iteration += 1
    return answer[0]

classifier_optimal,max_depth_optimal,min_samples_split_optimal,min_samples_leaf_optimal = optimal_tree()
y_train_prediction = classifier_optimal.predict(x_train)
y_test_prediction = classifier_optimal.predict(x_test)
y_validation_prediction = classifier_optimal.predict(x_validation)
print(f"Train accuracy is : {accuracy(y_train,y_train_prediction)}")
print(f"Test accuracy is : {accuracy(y_test,y_test_prediction)}")
print(f"Validation accuracy is : {accuracy(y_validation,y_validation_prediction)}")
end_time = time.time()
print(f"Total Time taken : {end_time-start_time}")
tree.plot_tree(classifier_optimal,filled=True,label='all')
plt.show()