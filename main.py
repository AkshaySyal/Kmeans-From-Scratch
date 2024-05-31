import numpy as np

if __name__ == '__main__':
    X = np.array([[1,0,0,1],[1,1,0,0],[1,0,1,0]])
    X_trans = X.T
    X_trans[3]/2
    A = np.array([1,0,0,1])
    B = np.array([1,0,0,1])
    np.dot(A,B) / (np.linalg.norm(A)*np.linalg.norm(B))
    np.linalg.norm(B)

    matrix = np.array([
    [0, 1, 2, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1]
])

# Get the index of the 1 in each row
indices = np.argmax(matrix, axis=1)
indices

X = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

# Example centroids (2 centroids in 2D space)
centroids = np.array([
    [7, 8],
    [1, 2]
])

membership = np.array([
    [0, 1],
    [0, 1],
    [1, 0]
])


# Compute the pairwise squared Euclidean distances
distances_squared = np.sum((X[:, np.newaxis] - centroids) ** 2, axis=2)
filtered_distances = distances_squared * membership
objective = np.sum(filtered_distances)

print(distances_squared)
print(filtered_distances.shape)
print(objective)