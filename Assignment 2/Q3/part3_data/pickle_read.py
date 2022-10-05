import pandas as pd
obj_test = pd.read_pickle(r'test_data.pickle')
for i in range(1000,3000):
    print(obj_test['labels'][i])