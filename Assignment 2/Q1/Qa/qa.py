import os
import sys
import time
import numpy as np
start = time.time()
train_path = sys.argv[1]
train_path_neg = train_path + "/neg"
train_path_pos =  train_path + "/pos"

def letter_check(letter):
    if (letter >= 'a' and letter <= 'z') or (letter >= 'A' and letter <= 'Z') or (ord(letter) == 39) : #39 is for ' letter
        return True
    else:
        return False
def filter_text(text_data):
    text_data = text_data.lower()
    word_freq = {} #dictonary to store word and its corresponding freqeuncy
    index_start = 0
    index_end = 0
    while index_end < len(text_data) and index_start < len(text_data):
        while index_start < len(text_data) and (not letter_check(text_data[index_start])):
            index_start += 1
        index_end = index_start
        while index_end < len(text_data) and letter_check(text_data[index_end]):
            index_end += 1
        word = text_data[index_start:index_end]
        if index_start < len(text_data):
            if word in word_freq:
                word_freq[word]+= 1
            else:
                word_freq[word] = 1
        index_start = index_end+1
    return word_freq

#this function will read the review file and return a correponding hashmap
def freqWords(file_name):
    file = open(file_name,'r',encoding='utf-8')
    text_data = file.read()
    file.close()
    word_map = filter_text(text_data)
    return word_map
def read_file_map(train_path):
    reviews = os.listdir(train_path)
    final_map = {}
    for file in reviews:
        review_freq = freqWords(train_path + '/' + file)
        for word in review_freq:
            if word in final_map :
                final_map[word] += review_freq[word]
            else:
                final_map[word] = review_freq[word]
    return final_map
def total_words(word_map):
    answer = 0
    for word in word_map:
        answer += word_map[word]
    return answer
negative_map = read_file_map(train_path_neg)
positive_map = read_file_map(train_path_pos)
total_word_neg = total_words(negative_map)
total_word_pos = total_words(positive_map)
phi = len(os.listdir(train_path_pos)) / (len(os.listdir(train_path_pos)) + len(os.listdir(train_path_neg)))
#any parameter theta_l/1 is now positive_map[word_l] +1/total_word_pos + |v| 

def predict(fileName):
    freq = freqWords(fileName)
    positive_prob = np.log(phi)
    negative_prob = np.log(1- phi)
    vocab_pos = len(positive_map)
    vocab_neg = len(negative_map)
    for word in freq:
        if word in positive_map:
            positive_prob += (np.log((positive_map[word] + 1)/(total_word_pos+vocab_pos))) * freq[word]
        else:
            positive_prob += (np.log((1)/(total_word_pos+vocab_pos))) * freq[word]
        if word in negative_map:
            negative_prob += (np.log((negative_map[word] + 1)/(total_word_neg+vocab_neg))) * freq[word]
        else:
            negative_prob += (np.log((1)/(total_word_neg+vocab_neg))) * freq[word]
    if positive_prob >= negative_prob :
        return 1
    else:
        return -1

def predict_directory(file_dir):
    files = os.listdir(file_dir)
    positive = 0
    negative = 0
    for file in files:
        prediction = predict(file_dir + '/' + file)
        if prediction == 1:
            positive += 1
        else:
            negative += 1
    return (positive,negative)

answer = predict_directory("../part1_data/train/neg")
print(answer)
answer = predict_directory("../part1_data/train/pos")
print(answer)
answer = predict_directory("../part1_data/test/pos")
print(answer)
answer = predict_directory("../part1_data/test/neg")
print(answer)
end = time.time()
print(end - start)