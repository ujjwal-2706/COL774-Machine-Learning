import sys
import numpy as np
import pandas as pd
import time
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import tree
from scipy import sparse
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

stop_words = set(stopwords.words('english'))
vectorizer_review = TfidfVectorizer()
vectorizer_condition = TfidfVectorizer()

start_time = time.time()
train_path = sys.argv[1]
validation_path = sys.argv[2]
test_path = sys.argv[3]
output_path = sys.argv[4]
question_part = sys.argv[5]

file_train = open(train_path)
file_train.readline()
train_load  = pd.read_csv(file_train).to_numpy()
file_train.close()

file_test = open(test_path)
file_test.readline()
test_load = pd.read_csv(file_test).to_numpy()
file_test.close()

file_validation = open(validation_path)
file_validation.readline()
validation_load = pd.read_csv(file_validation).to_numpy()
file_validation.close()

def letter_check(letter):
    if (letter >= 'a' and letter <= 'z') or (letter >= 'A' and letter <= 'Z') or (letter >= '0' and letter <= '9'):
        return True
    else:
        return False
def clean_review(review):
    if type(review) is str:
        answer = list(review)
        for i in range(len(answer)):
            if not letter_check(answer[i]):
                answer[i] = ' '
        value = ''.join(answer)
        new_string = value.split()
        final_string = []
        for val in new_string:
            if val not in stop_words:
                final_string.append(val)
                final_string.append(' ')
        return ''.join(final_string)
    else:
        return ""
def date_break(date):
    new_date = date.split()
    year = int(new_date[2])
    day = int(new_date[1][:-1])
    month = new_date[0].lower()
    if month == 'january':
        month = 1
    elif month == 'february':
        month = 2
    elif month == 'march':
        month = 3
    elif month == 'april':
        month = 4
    elif month == 'may':
        month = 5
    elif month == 'june':
        month = 6
    elif month == 'july':
        month = 7
    elif month == 'august':
        month = 8
    elif month == 'september':
        month = 9
    elif month == 'october':
        month = 10
    elif month == 'november':
        month = 11
    elif month == 'december':
        month = 12
    return np.array([day,month,year])
def load_proper_data(train_load,ifFit):
    m,features = np.shape(train_load)
    condition = train_load[:,0]
    review = train_load[:,1]
    rating = train_load[:,2]
    rating_new = []
    date = train_load[:,3]
    date_new = np.zeros((m,3))
    useful_count = train_load[:,4]
    new_use = np.zeros((m,1))
    for i in range(m):
        condition[i] = clean_review(condition[i])
        review[i] = clean_review(review[i])
        date_new[i,:] = date_break(date[i])
        rating_new.append(int(rating[i]))
        new_use[i] = int(useful_count[i])
    if ifFit:
        condition_new = vectorizer_condition.fit_transform(condition)
        review_new = vectorizer_review.fit_transform(review)
    else:
        condition_new = vectorizer_condition.transform(condition)
        review_new = vectorizer_review.transform(review)
    array_useful = sparse.csr_matrix(new_use)
    array_date = sparse.csr_matrix(date_new)
    x_final = sparse.hstack([condition_new,review_new,array_date,array_useful])
    return x_final,rating_new

def output_file(file_path,output_answer,part_number):
    file_name = file_path + "/2_" + part_number + ".txt"
    writing_file = open(file_name,'w')
    for i in range(len(output_answer)):
        writing_file.write(str(output_answer[i]))
        writing_file.write('\n')
    writing_file.close()

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


def run_part(part_number):
    if part_number == 'a':
        x_train,y_train = load_proper_data(train_load,True)
        x_test,y_test = load_proper_data(test_load,False)
        x_validation,y_validation = load_proper_data(validation_load,False)
        classifier = tree.DecisionTreeClassifier()
        classifier.fit(x_train,y_train)
        y_test_prediction = classifier.predict(x_test)
        y_train_prediction = classifier.predict(x_train)
        y_validation_prediction = classifier.predict(x_validation)
        print(f"Train accuracy is : {accuracy(y_train,y_train_prediction)}")
        print(f"Test accuracy is : {accuracy(y_test,y_test_prediction)}")
        print(f"Validation accuracy is : {accuracy(y_validation,y_validation_prediction)}")
        output_file(output_path,y_test_prediction,part_number)
    elif part_number == 'b':
        x_train,y_train = load_proper_data(train_load,True)
        x_test,y_test = load_proper_data(test_load,False)
        x_validation,y_validation = load_proper_data(validation_load,False)
        params = {'min_samples_leaf': list(range(2, 100)), 'min_samples_split': [2, 3, 4],'max_depth': list(range(2,20))}
        classifier = GridSearchCV(tree.DecisionTreeClassifier(),params)
        classifier.fit(x_train,y_train)
        y_train_prediction = classifier.predict(x_train)
        y_test_prediction = classifier.predict(x_test)
        y_validation_prediction = classifier.predict(x_validation)
        print(f"Train accuracy is : {accuracy(y_train,y_train_prediction)}")
        print(f"Test accuracy is : {accuracy(y_test,y_test_prediction)}")
        print(f"Validation accuracy is : {accuracy(y_validation,y_validation_prediction)}")
        output_file(output_path,y_test_prediction,part_number)
    elif part_number == 'c':
        x_train,y_train = load_proper_data(train_load,True)
        x_test,y_test = load_proper_data(test_load,False)
        x_validation,y_validation = load_proper_data(validation_load,False)
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
        x_train,y_train = load_proper_data(train_load,True)
        x_test,y_test = load_proper_data(test_load,False)
        x_validation,y_validation = load_proper_data(validation_load,False)
        params = {'n_estimators': list(range(50,450,50)), 'max_features': [0.4,0.5,0.6,0.7],'min_samples_split': list(range(2,10,2))}
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
    elif part_number == 'e':
        x_train,y_train = load_proper_data(train_load,True)
        x_test,y_test = load_proper_data(test_load,False)
        x_validation,y_validation = load_proper_data(validation_load,False)
        params = {'n_estimators' : list(range(50,450,50)),'subsample':[0.4,0.5,0.6,0.7],'max_depth':list(range(40,70,10))}
        classifier = GridSearchCV(xgb.XGBClassifier(use_label_encoder= False),params)
        classifier.fit(x_train,y_train)
        y_train_prediction = classifier.predict(x_train)
        y_test_prediction = classifier.predict(x_test)
        y_validation_prediction = classifier.predict(x_validation)
        print(f"Train accuracy is : {accuracy(y_train,y_train_prediction)}")
        print(f"Test accuracy is : {accuracy(y_test,y_test_prediction)}")
        print(f"Validation accuracy is : {accuracy(y_validation,y_validation_prediction)}")
        output_file(output_path,y_test_prediction,'e')

run_part(question_part)
end_time = time.time()
print(f"Time taken: {end_time - start_time}")