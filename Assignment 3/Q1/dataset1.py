from matplotlib import pyplot as plt
import numpy as np
import sys
import time
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

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

def imputation(data_points,imputation_type):
    if imputation_type == "median":
        result = []
        row,col = np.shape(data_points)
        medians = np.median(x_train,axis=1)
        for i in range(row):
            for index in range(1,col):
                if data_points[i,index] == '?':
                    result.append(medians[index-1])
                else:
                    result.append(data_points[i,index])
        new_rows = len(result) // (col-1)
        answer = np.array(result)
        answer = answer.reshape((new_rows,col-1))
        x_value = np.zeros((new_rows,col-2))
        y_value = np.zeros((new_rows,1))
        x_value = answer[:,:(col-2)]
        y_value = answer[:,col-2]
        return x_value,y_value
    elif imputation_type == "mode":
        result = []
        row,col = np.shape(data_points)
        modes = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=x_train)
        for i in range(row):
            for index in range(1,col):
                if data_points[i,index] == '?':
                    result.append(modes[index-1])
                else:
                    result.append(data_points[i,index])
        new_rows = len(result) // (col-1)
        answer = np.array(result)
        answer = answer.reshape((new_rows,col-1))
        x_value = np.zeros((new_rows,col-2))
        y_value = np.zeros((new_rows,1))
        x_value = answer[:,:(col-2)]
        y_value = answer[:,col-2]
        return x_value,y_value

def accuracy(original, prediction):
    total, = np.shape(original)
    correct = 0
    for i in range(total):
        if original[i] == prediction[i]:
            correct += 1
    return (100*correct)/(total)

#part a code
# classifier = tree.DecisionTreeClassifier()
# classifier.fit(x_train,y_train)
# y_train_prediction = classifier.predict(x_train)
# y_test_prediction = classifier.predict(x_test)
# y_validation_prediction = classifier.predict(x_validation)

#part b code
# params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4],'max_depth': list(range(2,20))}
# classifier = GridSearchCV(tree.DecisionTreeClassifier(),params)
# classifier.fit(x_train,y_train)
# y_train_prediction = classifier.predict(x_train)
# y_test_prediction = classifier.predict(x_test)
# y_validation_prediction = classifier.predict(x_validation)

# print(f"Train accuracy is : {accuracy(y_train,y_train_prediction)}")
# print(f"Test accuracy is : {accuracy(y_test,y_test_prediction)}")
# print(f"Validation accuracy is : {accuracy(y_validation,y_validation_prediction)}")

#part c code
# classifier = tree.DecisionTreeClassifier()
# classifier.fit(x_train,y_train)
#our goal is to minimize impurity cost
#the more the impurity the more the overfitted the tree is 
#choose optimal alpha which maximizes both test and training accuracies
# path = classifier.cost_complexity_pruning_path(x_train,y_train)
# ccp_alphas, impurities = path.ccp_alphas, path.impurities
# y_train_prediction = classifier.predict(x_train)
# y_test_prediction = classifier.predict(x_test)
# y_validation_prediction = classifier.predict(x_validation)

# def get_best_classifier():
#     nodes = []
#     depth = []
#     answer = {0: classifier}
#     validation_max = accuracy(y_validation,y_validation_prediction)
#     train_accuracy = []
#     test_accuracy = []
#     validation_accuracy = []
#     for alpha in ccp_alphas:
#         new_classifier = tree.DecisionTreeClassifier(ccp_alpha=alpha)
#         new_classifier.fit(x_train,y_train)
#         prediction_train = new_classifier.predict(x_train)
#         prediction_test = new_classifier.predict(x_test)
#         prediction_validation = new_classifier.predict(x_validation)
#         acc_train = accuracy(y_train,prediction_train)
#         acc_test = accuracy(y_test,prediction_test)
#         acc_validation = accuracy(y_validation,prediction_validation)
#         train_accuracy.append(acc_train)
#         test_accuracy.append(acc_test)
#         validation_accuracy.append(acc_validation)
#         nodes.append(new_classifier.tree_.node_count)
#         depth.append(new_classifier.tree_.max_depth)
#         if acc_validation > validation_max:
#             answer[0] = new_classifier
#     return (nodes,depth,train_accuracy,test_accuracy,validation_accuracy,answer[0])

# (nodes,depth,train_accuracy,test_accuracy,validation_accuracy,optimal_classifier) = get_best_classifier()
# predict_train = optimal_classifier.predict(x_train)
# predict_test = optimal_classifier.predict(x_test)
# predict_validation =  optimal_classifier.predict(x_validation)

# print(f"Train accuracy is : {accuracy(y_train,predict_train)}")
# print(f"Test accuracy is : {accuracy(y_test,predict_test)}")
# print(f"Validation accuracy is : {accuracy(y_validation,predict_validation)}")

# end_time = time.time()
# print(f"Time taken : {end_time - start_time}")

#plots for c part
#plot for alpha vs impurity
# plt.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
# plt.xlabel("Effective Alpha Values")
# plt.ylabel("Total Impurity of Leaves")
# plt.title("Total Impurity vs effective alpha for training set")
# plt.show()

#plot for depth of tree with alpha
# plt.plot(ccp_alphas,depth,marker='o',drawstyle='steps-post')
# plt.xlabel("Alpha used for Pruning")
# plt.ylabel("Depth of Tree")
# plt.title("Depth of Tree vs Alpha value")
# plt.show()

#plot for numbers of nodes with alpha
# plt.plot(ccp_alphas,nodes,marker='o',drawstyle='steps-post')
# plt.xlabel("Alpha used for Pruning")
# plt.ylabel("No. of Nodes in Tree")
# plt.title("No. of Nodes in Tree vs Alpha value")
# plt.show()

#plot for train accuracy with alpha
# plt.plot(ccp_alphas,train_accuracy,marker='o',drawstyle='steps-post')
# plt.xlabel("Alpha used for Pruning")
# plt.ylabel("Train Accuracy")
# plt.title("Train Accuracy vs Alpha value")
# plt.show()

#plot for test accuracy with alpha
# plt.plot(ccp_alphas,test_accuracy,marker='o',drawstyle='steps-post')
# plt.xlabel("Alpha used for Pruning")
# plt.ylabel("Test Accuracy")
# plt.title("Test Accuracy vs Alpha value")
# plt.show()

#plot for Validation accuracy with alpha
# plt.plot(ccp_alphas,validation_accuracy,marker='o',drawstyle='steps-post')
# plt.xlabel("Alpha used for Pruning")
# plt.ylabel("Validation Set Accuracy")
# plt.title("Validation Set Accuracy vs Alpha value")
# plt.show()

# tree.plot_tree(optimal_classifier,filled=True,label='all')
# plt.show()


#part d code
# random_classifier = RandomForestClassifier(oob_score=True)
# random_classifier.fit(x_train,y_train)
# y_train_prediction = random_classifier.predict(x_train)
# y_test_prediction = random_classifier.predict(x_test)
# y_valid_prediction = random_classifier.predict(x_validation)

# print(f"Train accuracy is : {accuracy(y_train,y_train_prediction)}")
# print(f"Test accuracy is : {accuracy(y_test,y_test_prediction)}")
# print(f"Validation accuracy is : {accuracy(y_validation,y_valid_prediction)}")
# print(f"Out of bag accuracy is : {random_classifier.oob_score_}")


#part e code
#imputed data loaded and filtered
# x_train_med,y_train_med = imputation(train_load,"median")
# x_test_med,y_test_med = imputation(test_load,"median")
# x_validation_med,y_validation_med = imputation(validation_load,"median")

# x_train_mod,y_train_mod = imputation(train_load,"mode")
# x_test_mod,y_test_mod = imputation(test_load,"mode")
# x_validation_mod,y_validation_mod = imputation(validation_load,"mode")

# classifier_med = tree.DecisionTreeClassifier()
# classifier_med.fit(x_train_med,y_train_med)
# y_train_prediction_med = classifier_med.predict(x_train_med)
# y_test_prediction_med = classifier_med.predict(x_test_med)
# y_validation_prediction_med = classifier_med.predict(x_validation_med)

# print(f"Train accuracy is : {accuracy(y_train_med,y_train_prediction_med)}")
# print(f"Test accuracy is : {accuracy(y_test_med,y_test_prediction_med)}")
# print(f"Validation accuracy is : {accuracy(y_validation_med,y_validation_prediction_med)}")


