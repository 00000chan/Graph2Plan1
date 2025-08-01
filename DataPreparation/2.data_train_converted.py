import numpy as np
import pickle
import scipy.io as sio
from config import data_path
from tqdm.auto import tqdm

# load data
data = sio.loadmat(data_path, squeeze_me=True, struct_as_record=False)['data']
data_dict = {d.name:d for d in data}

with open('./data/train.txt', 'r') as f:
    names_train = [line.strip() for line in f if line.strip()]
n_train = len(names_train)

with open('./data/trainTF.pkl', 'rb') as f:
    trainTF = pickle.load(f)

data_converted = []

for i in tqdm(range(n_train)):
    # Handle potential empty name from file reading
    if not names_train[i]:
        continue
    d = data_dict[names_train[i]]
    d_converted = {}
    d_converted['name'] = d.name
    d_converted['boundary'] = d.boundary
    d_converted['box'] = np.concatenate([d.gtBoxNew, d.rType[:, None]], axis=-1)
    d_converted['order'] = d.order
    d_converted['edge'] = d.rEdge
    d_converted['rBoundary'] = d.rBoundary
    data_converted.append(d_converted)

sio.savemat('./data/data_train_converted.mat', {'data': data_converted, 'nameList': names_train, 'trainTF': trainTF})

# Re-load the .mat file to ensure consistent structure and save as .pkl
data = sio.loadmat('./data/data_train_converted.mat', squeeze_me=True, struct_as_record=False)

with open('./data/data_train_converted.pkl', 'wb') as f:
    pickle.dump(data, f)

