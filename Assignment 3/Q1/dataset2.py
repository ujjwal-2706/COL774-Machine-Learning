import sys
import numpy as np
import pandas as pd
import time
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree

stop_words = set(stopwords.words('english'))
vectorizer = CountVectorizer(max_features=30000)

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
def load_proper_data(train_load):
    m,features = np.shape(train_load)
    condition = train_load[:,0]
    review = train_load[:,1]
    rating = train_load[:,2]
    rating_new = []
    date = train_load[:,3]
    date_new = np.zeros((m,3))
    useful_count = train_load[:,4]
    new_features = np.array(condition)
    for i in range(m):
        condition[i] = clean_review(condition[i])
        review[i] = clean_review(review[i])
        new_features[i] = condition[i] + " " + review[i] + " " + date[i].replace(',',' ') + " " + str(useful_count[i])
        # date_new[i,:] = date_break(date[i])
        rating_new.append(int(rating[i]))
    x_final = vectorizer.fit_transform(new_features)
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

def run_part(part_number):
    if part_number == 'a':
        x_train,y_train = load_proper_data(train_load)
        x_test,y_test = load_proper_data(test_load)
        x_validation,y_validation = load_proper_data(validation_load)
        classifier = tree.DecisionTreeClassifier()
        classifier.fit(x_train,y_train)
        y_train_prediction = classifier.predict(x_train)
        y_test_prediction = classifier.predict(x_test)
        y_validation_prediction = classifier.predict(x_validation)
        print(f"Train accuracy is : {accuracy(y_train,y_train_prediction)}")
        print(f"Test accuracy is : {accuracy(y_test,y_test_prediction)}")
        print(f"Validation accuracy is : {accuracy(y_validation,y_validation_prediction)}")
        output_file(output_path,y_test_prediction,part_number)
run_part(question_part)
end_time = time.time()
print(f"Time taken: {end_time - start_time}")