import pickle
import numpy as np
import faiss
from tqdm.auto import tqdm
import argparse

def sample_tf(x, y, ndim=1000):
    '''
    input: tf.x,tf.y, ndim
    return: n-dim tf values
    '''
    t = np.linspace(0, 1, ndim)
    return np.piecewise(t, [t >= xx for xx in x], y)

# --- Main Script ---
parser = argparse.ArgumentParser(description='Cluster turning functions using Faiss K-means.')
parser.add_argument('--no-gpu', action='store_true', help='Disable GPU usage for clustering.')
args = parser.parse_args()

# Load the pre-computed turning functions
with open('./data/trainTF.pkl', 'rb') as f:
    tf_train = pickle.load(f)

# Sample the turning functions to create fixed-size vectors
print("Sampling turning functions...")
tf_vectors = []
for i in tqdm(range(len(tf_train))):
    tf_i = tf_train[i]
    tf_vectors.append(sample_tf(tf_i['x'], tf_i['y']))

d = 1000  # dimension of the vectors
tf_vectors = np.array(tf_vectors).astype(np.float32)

# Faiss K-means parameters
ncentroids = 1000
niter = 20
verbose = True

# Check for GPU availability and user preference
use_gpu = not args.no_gpu and faiss.get_num_gpus() > 0
if use_gpu:
    print(f"Found {faiss.get_num_gpus()} GPUs. Using GPU for K-means.")
else:
    print("GPU not found or disabled. Using CPU for K-means.")

# Perform K-means clustering
print(f"Starting K-means clustering for {ncentroids} centroids...")
kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=use_gpu)
kmeans.train(tf_vectors)
centroids = kmeans.centroids
print("K-means training complete.")

# Create an index to find nearest neighbors for each centroid
print("Finding nearest neighbors for each centroid...")
index = faiss.IndexFlatL2(d)
index.add(tf_vectors)
nNN = 1000  # Number of nearest neighbors to find for each centroid
D, I = index.search(kmeans.centroids, nNN) # D is distances, I is indices
print("Nearest neighbor search complete.")

# Save the results
# centroids_train.npy: The 1000 cluster centers.
# clusters_train.npy: For each centroid, the indices of the 1000 closest floor plans.
np.save(f'./data/centroids_train.npy', centroids)
np.save(f'./data/clusters_train.npy', I)

print("Saved centroids and cluster indices.")