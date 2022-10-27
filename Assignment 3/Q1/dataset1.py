from matplotlib import pyplot as plt
import numpy as np
import sys
import time
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

train_path = sys.argv[1] + "/train.csv"
validation_path = sys.argv[2] + "/val.csv"
test_path = sys.argv[3] + "/test.csv"
output_path = sys.argv[4]
question_part = sys.argv[5]

start_time = time.time()
file_train = open(train_path)
file_train.readline()
train_load  = np.loadtxt(file_train,delimiter=',',dtype='str')
file_train.close()

file_test = open(test_path)
file_test.readline()
test_load = np.loadtxt(file_test,delimiter=',',dtype='str')
file_test.close()

file_validation = open(validation_path)
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

#this function will impute the missing data
def imputation(data_points,imputation_type,x_train):
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

#this function will give best classifer among the pruned alphas
def get_best_classifier(classifier,x_validation,y_validation,y_validation_prediction,ccp_alphas,x_train,y_train,x_test,y_test):
    nodes = []
    depth = []
    answer = {0: classifier}
    validation_max = accuracy(y_validation,y_validation_prediction)
    train_accuracy = []
    test_accuracy = []
    validation_accuracy = []
    for alpha in ccp_alphas:
        new_classifier = tree.DecisionTreeClassifier(ccp_alpha=alpha)
        new_classifier.fit(x_train,y_train)
        prediction_train = new_classifier.predict(x_train)
        prediction_test = new_classifier.predict(x_test)
        prediction_validation = new_classifier.predict(x_validation)
        acc_train = accuracy(y_train,prediction_train)
        acc_test = accuracy(y_test,prediction_test)
        acc_validation = accuracy(y_validation,prediction_validation)
        train_accuracy.append(acc_train)
        test_accuracy.append(acc_test)
        validation_accuracy.append(acc_validation)
        nodes.append(new_classifier.tree_.node_count)
        depth.append(new_classifier.tree_.max_depth)
        if acc_validation > validation_max:
            answer[0] = new_classifier
    return (nodes,depth,train_accuracy,test_accuracy,validation_accuracy,answer[0])

def output_file(file_path,output_answer,part_number):
    file_name = file_path + "/1_" + part_number + ".txt"
    writing_file = open(file_name,'w')
    for i in range(len(output_answer)):
        writing_file.write(str(output_answer[i]))
        writing_file.write('\n')
    writing_file.close()

def split_data(train_load):
    m,features_1 = np.shape(train_load)
    answer = []
    for i in range(m):
        for j in range(1,features_1):
            if train_load[i,j] == '?':
                answer.append(0)
            else:
                answer.append(int(train_load[i,j]))
    result = np.array(answer)
    result = result.reshape((m,features_1-1))
    result.astype(int)
    y_train = result[:,features_1-2]
    x_train = result[:,:features_1-2]
    return (x_train,y_train)

def run_part(part_number):
    if part_number == 'a':
        x_train,y_train = filter_data(train_load)
        x_test,y_test = filter_data(test_load)
        x_validation,y_validation = filter_data(validation_load)
        classifier = tree.DecisionTreeClassifier()
        classifier.fit(x_train,y_train)
        y_train_prediction = classifier.predict(x_train)
        y_test_prediction = classifier.predict(x_test)
        y_validation_prediction = classifier.predict(x_validation)
        print(f"Train accuracy is : {accuracy(y_train,y_train_prediction)}")
        print(f"Test accuracy is : {accuracy(y_test,y_test_prediction)}")
        print(f"Validation accuracy is : {accuracy(y_validation,y_validation_prediction)}")
        output_file(output_path,y_test_prediction,'a')
    elif part_number == 'b':
        x_train,y_train = filter_data(train_load)
        x_test,y_test = filter_data(test_load)
        x_validation,y_validation = filter_data(validation_load)
        params = {'min_samples_leaf': list(range(2, 100)), 'min_samples_split': [2, 3, 4],'max_depth': list(range(2,20))}
        classifier = GridSearchCV(tree.DecisionTreeClassifier(),params)
        classifier.fit(x_train,y_train)
        y_train_prediction = classifier.predict(x_train)
        y_test_prediction = classifier.predict(x_test)
        y_validation_prediction = classifier.predict(x_validation)
        print(f"Train accuracy is : {accuracy(y_train,y_train_prediction)}")
        print(f"Test accuracy is : {accuracy(y_test,y_test_prediction)}")
        print(f"Validation accuracy is : {accuracy(y_validation,y_validation_prediction)}")
        output_file(output_path,y_test_prediction,'b')
    elif part_number == 'c':
        x_train,y_train = filter_data(train_load)
        x_test,y_test = filter_data(test_load)
        x_validation,y_validation = filter_data(validation_load)
        classifier = tree.DecisionTreeClassifier()
        classifier.fit(x_train,y_train)
        #our goal is to minimize impurity cost
        #the more the impurity the more the overfitted the tree is 
        #choose optimal alpha which maximizes both test and training accuracies
        path = classifier.cost_complexity_pruning_path(x_train,y_train)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities
        y_train_prediction = classifier.predict(x_train)
        y_test_prediction = classifier.predict(x_test)
        y_validation_prediction = classifier.predict(x_validation)
        (nodes,depth,train_accuracy,test_accuracy,validation_accuracy,optimal_classifier) = get_best_classifier(classifier,x_validation,y_validation,y_validation_prediction,ccp_alphas,x_train,y_train,x_test,y_test)
        predict_train = optimal_classifier.predict(x_train)
        predict_test = optimal_classifier.predict(x_test)
        predict_validation =  optimal_classifier.predict(x_validation)
        print(f"Train accuracy is : {accuracy(y_train,predict_train)}")
        print(f"Test accuracy is : {accuracy(y_test,predict_test)}")
        print(f"Validation accuracy is : {accuracy(y_validation,predict_validation)}")
        output_file(output_path,predict_test,'c')
    elif part_number == 'd':
        x_train,y_train = filter_data(train_load)
        x_test,y_test = filter_data(test_load)
        x_validation,y_validation = filter_data(validation_load)
        params = {'n_estimators': [200,300,400], 'max_features': ['sqrt','log2'],'min_samples_split': list(range(2,15))}
        classifier = GridSearchCV(RandomForestClassifier(oob_score=True),params)
        classifier.fit(x_train,y_train)
        y_train_prediction = classifier.predict(x_train)
        y_test_prediction = classifier.predict(x_test)
        y_validation_prediction = classifier.predict(x_validation)
        print(f"Train accuracy is : {accuracy(y_train,y_train_prediction)}")
        print(f"Test accuracy is : {accuracy(y_test,y_test_prediction)}")
        print(f"Validation accuracy is : {accuracy(y_validation,y_validation_prediction)}")
        print(f"Out of Bag accuracy is : {classifier.best_estimator_.oob_score_}")
        output_file(output_path,y_test_prediction,'d')
    elif part_number== 'e':
        #imputed data loaded and filtered
        x_train,y_train = filter_data(train_load)
        x_test,y_test = filter_data(test_load)
        x_validation,y_validation = filter_data(validation_load)

        x_train_med,y_train_med = imputation(train_load,"median",x_train)
        x_test_med,y_test_med = imputation(test_load,"median",x_train)
        x_validation_med,y_validation_med = imputation(validation_load,"median",x_train)

        x_train_mod,y_train_mod = imputation(train_load,"mode",x_train)
        x_test_mod,y_test_mod = imputation(test_load,"mode",x_train)
        x_validation_mod,y_validation_mod = imputation(validation_load,"mode",x_train)

        classifier_mod = tree.DecisionTreeClassifier()
        classifier_mod.fit(x_train_mod,y_train_mod)
        y_train_prediction_mod = classifier_mod.predict(x_train_mod)
        y_test_prediction_mod = classifier_mod.predict(x_test_mod)
        y_validation_prediction_mod = classifier_mod.predict(x_validation_mod)
        print(f"Train accuracy is : {accuracy(y_train_mod,y_train_prediction_mod)}")
        print(f"Test accuracy is : {accuracy(y_test_mod,y_test_prediction_mod)}")
        print(f"Validation accuracy is : {accuracy(y_validation_mod,y_validation_prediction_mod)}")
        output_file(output_path,y_test_prediction_mod,'e')
    elif part_number == 'f':
        x_train,y_train = split_data(train_load)
        x_test,y_test = split_data(test_load)
        x_validation,y_validation = split_data(validation_load)
        params = {'n_estimators' : list(range(10,20,10)),'subsample':[0.1,0.2,0.3,0.4,0.5,0.6],'max_depth':list(range(4,11,1))}
        classifier = GridSearchCV(xgb.XGBClassifier(use_label_encoder= False),params)
        classifier.fit(x_train,y_train)
        y_train_prediction = classifier.predict(x_train)
        y_test_prediction = classifier.predict(x_test)
        y_validation_prediction = classifier.predict(x_validation)
        print(f"Train accuracy is : {accuracy(y_train,y_train_prediction)}")
        print(f"Test accuracy is : {accuracy(y_test,y_test_prediction)}")
        print(f"Validation accuracy is : {accuracy(y_validation,y_validation_prediction)}")
        output_file(output_path,y_test_prediction,'f')

run_part(question_part)
end_time = time.time()
print(f"Total Time taken : {end_time - start_time}")


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
