import numpy as np
import pickle
from tqdm.auto import tqdm

# Load the converted training data
with open('./data/data_train_converted.pkl', 'rb') as f:
    data = pickle.load(f)['data']

# Load the list of training names
with open('./data/train.txt', 'r') as f:
    names_train = [line.strip() for line in f if line.strip()]
n_train = len(names_train)

# eNum will store the flattened adjacency matrix for coarse-grained room types
eNum = np.zeros((n_train, 25), dtype='uint8')

# Mapping from 18 room types to 6 coarse-grained categories.
# Based on README, roughly: 0:Living, 1:Bedroom, 2:Kitchen, 3:Bathroom, 4:Dining, 5:Balcony etc.
# Indices are adjusted for Python (0-based).
rMap = np.array([1, 2, 3, 4, 1, 2, 2, 2, 2, 5, 1, 6, 1, 10, 7, 8, 9, 10]) - 1
reorder = np.array([0, 1, 3, 2, 4, 5]) # A reordering mapping

for i in tqdm(range(n_train)):
    d = data[i]
    rType = d.box[:, -1]
    eType = rType[d.edge[:, :2]]

    # Map detailed room types to coarser categories
    edge = rMap[eType]
    edge = reorder[edge]
    
    # Filter for edges between the 5 main categories (1 to 5)
    I = (edge[:, 0] <= 5) & (edge[:, 0] >= 1) & (edge[:, 1] <= 5) & (edge[:, 1] >= 1)
    edge = edge[I, :] - 1  # Subtract 1 to make it 0-indexed for matrix

    # Build the symmetric 5x5 adjacency matrix
    e = np.zeros((5, 5), dtype='uint8')
    for j in range(len(edge)):
        u, v = edge[j, 0], edge[j, 1]
        e[u, v] += 1
        if u != v:
            e[v, u] += 1

    # Flatten the 5x5 matrix into a 25-element vector and store it
    eNum[i] = e.reshape(-1)

# Save the result
with open('./data/data_train_eNum.pkl', 'wb') as f:
    pickle.dump({'eNum': eNum}, f)

