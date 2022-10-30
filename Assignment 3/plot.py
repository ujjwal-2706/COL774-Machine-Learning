import numpy as np
import matplotlib.pyplot as plt


#data part b
# accuracy_test = [83.2,85.42,86.39,87.02,87.53]
# accuracy_train = [86.67,89.84,90.61,91.59,92.23]
# time_taken = [148.62,166.89,226.32,280,386.14]
# layer_size = [5,10,15,20,25]

#data part c
# accuracy_test = [76.33,83.11,83.9,84.26,84.7]
# accuracy_train = [77,84.6,85.55,85.83,86.2]
# time_taken = [139,160,210,260,310]
# layer_size = [5,10,15,20,25]


#data part e
accuracy_test_sigmoid = [87.73,86.53,10,10]
accuracy_test_relu = [87.91,88.57,88.21,87.43]
accuracy_train_sigmoid = [92.26,90.72,10,10]
accuracy_train_relu = [92.93,93.36,93.13,92.53]
hidden_layers = [2,3,4,5]

plt.plot(hidden_layers,accuracy_train_relu)
plt.xlabel("No. of Hidden Layers")
plt.ylabel("Relu Train Accuracy")
plt.title("Relu Train Accuracy Vs No. of Hidden Layers")
plt.show()
