import numpy as np
import sys
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import time
start_time = time.time()
train_data = np.loadtxt("../../../part2/fmnist_train.csv",dtype=int,delimiter=',')
test_data = np.loadtxt("../../../part2/fmnist_test.csv",dtype=int,delimiter=',')
(m,features) = np.shape(train_data)
(test_case,total_features) = np.shape(test_data)
features = features-1
x_train = np.zeros((m,features))
y_train = np.zeros((m,1))
x_train = train_data[:,:features]
y_train = train_data[:,features]
x_test = test_data[:,:features]
y_test = test_data[:,features]

x_test_normalize = x_test/(255.0)
#data normalized by dividing by highest pixel unit
x_train_normalize = x_train/(255.0)

CLASSES= 10
y_train_encoding = np.zeros((m,CLASSES),dtype=np.double)

def encode(y_val):
    answer = np.zeros((1,CLASSES),dtype=np.double)
    answer[0,y_val-1] = 1.0
    return answer
for i in range(m):
    y_train_encoding[i,:] = encode(y_train[i])

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

# def print_array(numpy_array):
#     row,col = np.shape(numpy_array)
#     for i in range(row):
#         for j in range(col):
#             if numpy_array[i,j] != 0:
#                 print(numpy_array[i,j])
#function to find o(l) and net(l) given a theta dictionary and x_data
#we will always have to add a neuron that outputs 1
#assume theta for each neuron arranged in every row
#x_data arranged in each column
def output(x_data,theta,layers):
    feat,points = np.shape(x_data)
    input = np.vstack((np.ones((1,points),dtype=np.double),x_data))
    layer_output = {0:x_data}
    for l in range(1,layers+1):
        input = theta[l] @ input
        input = sigmoid(input)
        # print_array(input)
        layer_output[l] = input
        feat,points = np.shape(input)
        input = np.vstack((np.ones((1,points),dtype=np.double),input))
    return layer_output #this dictoinary of matrix o(l) in each column for m points

#y_data is matrix of one-hot encoding each row represents a output
#x_data arranged in column form
def gradients(x_data,theta,layers,y_data):
    layer_output = output(x_data,theta,layers)
    theta_gradients = {}
    net_gradients = {}
    #its storing a matrix with ith column representing ith data point
    net_gradients[layers] =(y_data.T - layer_output[layers]) * (layer_output[layers] * (layer_output[layers]-1))
    row,col = np.shape(layer_output[layers-1])
    theta_gradients[layers] = net_gradients[layers] @ (np.vstack(((np.ones((1,col),dtype=np.double)),layer_output[layers-1])).T)
    for l in range(layers-1,0,-1):
        J_l = net_gradients[l+1]
        theta_temp = theta[l+1][:,1:] #remove the constants
        net_gradients[l] = (theta_temp.T @ J_l) * (layer_output[l] * (1-layer_output[l]))
        row,col = np.shape(layer_output[l-1])
        theta_gradients[l] = net_gradients[l] @ (np.vstack(((np.ones((1,col),dtype=np.double)),layer_output[l-1])).T)
    return theta_gradients

#kth batch of size batch_size and r is the output one-hot encodes
def batch_slice(train_whole,batch_size,r,k):
    total_points,values = np.shape(train_whole)
    left_end = (batch_size*k) % total_points
    right_end = (batch_size*(k+1)) % total_points
    if left_end < right_end:
        output = train_whole[left_end:right_end,:]
        x_value = output[:,:values-r]
        y_value = output[:,values-r:]
        return (x_value,y_value)
    else:
        output1 = train_whole[left_end:,:]
        output2 = train_whole[:right_end,:]
        output = np.vstack((output1,output2))
        x_value = output[:,:values-r]
        y_value = output[:,values-r:]
        return (x_value,y_value)

#number of features depend on x_train_norm so already taken into account
#number of target classes depend on one-hot encoded y, so already taken into account
def gradient_descent(x_train_norm,y_train_bit,eta,layer_list,batch_size):
    row,col = np.shape(y_train_bit)
    train_whole = np.hstack((x_train_norm,y_train_bit))
    np.random.shuffle(train_whole)
    theta = {}
    layers = len(layer_list)
    previous_input = features
    for layer in range(1,layers+1):
        neurons = layer_list[layer-1]
        theta_val = np.random.normal(0,0.1,size=(neurons,1+previous_input)) 
        previous_input = neurons
        theta[layer] = theta_val
    batch_num = 0
    x_val,y_val = batch_slice(train_whole,batch_size,col,batch_num)
    theta_gradients = gradients(x_val.T,theta,layers,y_val)
    total_iterations = 0
    while True:
        iteration = 0
        diff_val = 0
        while iteration < 100:
            x_val,y_val = batch_slice(train_whole,batch_size,col,batch_num)
            theta_gradients = gradients(x_val.T,theta,layers,y_val)
            for i in range(1,layers+1):
                theta[i] = theta[i] - (eta/batch_size)*(theta_gradients[i])
                val = (eta/batch_size)*(theta_gradients[i])
                num_row,num_col = np.shape(val)
                diff_val += np.max(val)
            batch_num += 1
            iteration += 1
        diff_val /= iteration
        diff_val /= layers
        total_iterations += 1
        if  diff_val < 0.00001 or total_iterations == 6000:
            break
    return theta

def adaptive_learning(x_train_norm,y_train_bit,eta,layer_list,batch_size):
    row,col = np.shape(y_train_bit)
    train_whole = np.hstack((x_train_norm,y_train_bit))
    np.random.shuffle(train_whole)
    theta = {}
    original_eta = eta
    layers = len(layer_list)
    previous_input = features
    for layer in range(1,layers+1):
        neurons = layer_list[layer-1]
        theta_val = np.random.normal(0,0.1,size=(neurons,1+previous_input)) 
        previous_input = neurons
        theta[layer] = theta_val
    batch_num = 0
    x_val,y_val = batch_slice(train_whole,batch_size,col,batch_num)
    theta_gradients = gradients(x_val.T,theta,layers,y_val)
    total_iterations = 0
    hops = row//batch_size
    epoch_num = 1
    while True:
        iteration = 0
        diff_val = 0
        while iteration < 100:
            x_val,y_val = batch_slice(train_whole,batch_size,col,batch_num)
            theta_gradients = gradients(x_val.T,theta,layers,y_val)
            eta = original_eta/np.sqrt(epoch_num)
            for i in range(1,layers+1):
                theta[i] = theta[i] - (eta/batch_size)*(theta_gradients[i])
                val = (eta/batch_size)*(theta_gradients[i])
                num_row,num_col = np.shape(val)
                diff_val += np.max(val)
            batch_num += 1
            if batch_num >= hops:
                batch_num = 0
                epoch_num += 1
            iteration += 1
        diff_val /= iteration
        diff_val /= layers
        total_iterations += 1
        if  diff_val < 0.00001 or total_iterations == 6000:
            break
    return theta
# theta = adaptive_learning(x_train_normalize,y_train_encoding,0.1,[5,10],100)
def accuracy(theta,x_test,y_test,layers):
    outputs = output(x_test.T,theta,layers)
    final_answer = outputs[layers]
    row,col = np.shape(final_answer)
    correct = 0
    for test in range(col):
        max_index = 0
        for j in range(row):
            if final_answer[j,test] > final_answer[max_index,test]:
                max_index = j
        if max_index + 1 == y_test[test]:
            correct += 1
    return (100*correct/col)

def accuracy_comparator(y_test,y_pred):
    correct = 0
    for i in range(len(y_test)):
        if y_test[i] == y_pred[i]:
            correct +=1
    return 100*correct/len(y_pred)

neural_classifier = MLPClassifier(hidden_layer_sizes=(5,),activation='logistic',solver='adam',max_iter=100,random_state=1)
neural_classifier.fit(x_train_normalize,y_train)
y_scikit_prediction = neural_classifier.predict(x_test_normalize)
print(f"Accuracy MLP: {accuracy_comparator(y_test,y_scikit_prediction)}")
# print(accuracy(theta,x_test_normalize,y_test,2))
# print(accuracy(theta,x_train_normalize,y_train,2))
end_time = time.time()
print(f"Time Taken is : {end_time - start_time}")
#accuracy on 5 coming with 6000 iterations 74.65% on test data and 77.78% on training data in 205 seconds
#accuracy on 10 coming with 6000 iterations 77.34% on test data and 81.07% on training data in 324 seconds
#accuracy on 15 coming with 6000 iterations 78.03% on test data and 82.065% on training data in 1026 seconds
#accuracy on 25 coming with 6000 iterations 79.2% on test data and 83.27% on training data in 1000 seconds

def output_predictions(theta,x_test,layers):
    outputs = output(x_test.T,theta,layers)
    final_answer = outputs[layers]
    row,col = np.shape(final_answer)
    answer = []
    for test in range(col):
        max_index = 0
        for j in range(row):
            if final_answer[j,test] > final_answer[max_index,test]:
                max_index = j
        answer.append(max_index +1)
    return answer

# prediction_test = output_predictions(theta,x_test,2)
# cm_naive = confusion_matrix(y_test,prediction_test)
# disp_naive = ConfusionMatrixDisplay(confusion_matrix=cm_naive)
# disp_naive.plot()
# plt.savefig("confusion_naive.png",dpi = 1000)

#accuracy on 5 with adaptive learning is 67.37 on test and 68.43 on train in 214 seconds