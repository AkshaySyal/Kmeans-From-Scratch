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