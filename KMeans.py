import numpy as np

class KMeans:
    def __init__(self,k,dist_type):
        self.k = k # number of clusters
        self.dist_type = dist_type
    
    def distance(self,x,y):
        if(self.dist_type == 'Euclidean'):
            return np.linalg.norm(x-y)
        elif(self.dist_type == 'Cosine Similarity'):
            return np.dot(x,y) / (np.linalg.norm(x)*np.linalg.norm(y))

    def transform_mnist(self,data):
        pass

    def transform_fashion(self,data):
        pass
    
    def transform_news_groups(self,data):
        pass

    def fit(self,data):
        self.data = data
        
        # initializing pi (membership matrix)
        self.pi = np.zeros((len(data),self.k), dtype=int)

        # initializing centroids
        self.centroids = np.random.choice(data, self.k, replace=False)
    
    def computePi(self):
        # for all data points find the closest centroid and update pi
        for i in range(len(self.data)):
            dist = self.distance(self.data[i],self.centroids[0])
            closest_centroid_idx = 0
            for centroid_idx in range(1,len(self.centroids)):
                if(self.distance(self.data[i],self.centroids[centroid_idx]) < dist):
                    dist = self.distance(self.data[i]-self.centroids[centroid_idx])
                    closest_centroid_idx = centroid_idx

            self.pi[i][closest_centroid_idx] = 1
    
    def computeCentroids(self):
        # for all k clusters
        # pi[i] (reshaped to 1xN) is multiplied with Xi (NxD)
        # normalized by num of data points in cluster k i.e. sum(pi[i])

        for k in range(self.k):
            self.centroids[k] = self.pi.T[k] @ self.data / sum(self.pi.T[k])
    

    def fit(self):
        pass

    def predict(self):
        pass


    def kmeansObjective(self):
        pass 
    
    def evaluteClustering(self):
        pass


