import numpy as np
import pickle
import scipy.io as sio
from config import data_path
from tqdm.auto import tqdm

data = sio.loadmat(data_path, squeeze_me=True, struct_as_record=False)['data']
data_dict = {d.name: d for d in data}

with open('./data/testTF.pkl', 'rb') as f:
    testTF = pickle.load(f)
rNum = np.load('./data/rNum_train.npy')

with open('./data/train.txt', 'r') as f:
    names_train = [line.strip() for line in f if line.strip()]
with open('./data/test.txt', 'r') as f:
    names_test = [line.strip() for line in f if line.strip()]
n_test = len(names_test)

D = np.load('./data/D_test_train.npy')
data_converted = []
for i in tqdm(range(n_test)):
    if not names_test[i]:
        continue
    d = data_dict[names_test[i]]
    d_converted = {}
    d_converted['boundary'] = d.boundary
    d_converted['tf'] = testTF[i]
    # Find the top 1000 training samples with the most similar turning functions
    topK = np.argsort(D[i])[:1000]
    d_converted['topK'] = topK
    # Store the room number counts for these top K samples for later filtering
    d_converted['topK_rNum'] = rNum[topK]
    data_converted.append(d_converted)

sio.savemat('./data/data_test_converted.mat',
            {'data': data_converted, 'testNameList': names_test, 'trainNameList': names_train})

data = sio.loadmat('./data/data_test_converted.mat', squeeze_me=True, struct_as_record=False)
with open('./data/data_test_converted.pkl', 'wb') as f:
    pickle.dump(data, f)
