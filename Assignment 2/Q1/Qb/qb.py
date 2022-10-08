import os
import sys
import time
import random
start = time.time()
train_path = sys.argv[1]
test_path = sys.argv[2]
train_path_neg = train_path + "/neg"
train_path_pos =  train_path + "/pos"
test_path_neg = test_path + "/neg"
test_path_pos =  test_path + "/pos"

def predict_random(file_dir):
    files = os.listdir(file_dir)
    positive = 0
    negative = 0
    for file in files:
        prediction = random.randint(0,1)
        if prediction == 1:
            positive += 1
        else:
            negative += 1
    return (positive,negative)

def predict_positive(file_dir):
    files = os.listdir(file_dir)
    positive = len(files)
    negative = 0
    return (positive,negative)

(train_neg_positive,train_neg_negative) = predict_random(train_path_neg)
(train_pos_positive,train_pos_negative) = predict_random(train_path_pos)
print(f"Random train accuracy : {(train_neg_negative + train_pos_positive)/(train_neg_negative + train_pos_positive + train_neg_positive + train_pos_negative)}")
(test_neg_positive,test_neg_negative) = predict_random(test_path_neg)
(test_pos_positive,test_pos_negative) = predict_random(test_path_pos)
print(f"Random test accuracy : {(test_neg_negative + test_pos_positive)/(test_neg_negative + test_pos_positive + test_neg_positive + test_pos_negative)}")


(train_neg_positive,train_neg_negative) = predict_positive(train_path_neg)
(train_pos_positive,train_pos_negative) = predict_positive(train_path_pos)
print(f"Positive train accuracy : {(train_neg_negative + train_pos_positive)/(train_neg_negative + train_pos_positive + train_neg_positive + train_pos_negative)}")
(test_neg_positive,test_neg_negative) = predict_positive(test_path_neg)
(test_pos_positive,test_pos_negative) = predict_positive(test_path_pos)
print(f"Positive test accuracy : {(test_neg_negative + test_pos_positive)/(test_neg_negative + test_pos_positive + test_neg_positive + test_pos_negative)}")

end = time.time()
print(f"Time Taken : {end - start}")
