import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
from KMeans import KMeans

def transform_mnist(data):
       transformed_data = data.reshape(data.shape[0],-1)
       transformed_data[transformed_data>0] = 1
       return transformed_data

def transform_fashion(data):
        pass
    
def transform_news_groups(data):
        pass

def visualizeImg(img,lbl,reshaped=False):
        i = 3
        if(reshaped):
               plt.imshow(img.reshape((28,28)), cmap='viridis', interpolation='nearest')
        else:
               plt.imshow(img, cmap='viridis', interpolation='nearest')
        plt.colorbar()  # Show color scale
        plt.title(f'Image of {lbl}')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()

if __name__ == '__main__':
    imgs = idx2numpy.convert_from_file("Datasets/MNIST/t10k-images-idx3-ubyte")
    imgs_copy = np.copy(imgs)
    lbls = idx2numpy.convert_from_file("Datasets/MNIST/t10k-labels-idx1-ubyte")
    lbls_copy = np.copy(lbls)

    transformed_imgs = transform_mnist(imgs_copy)
    #i=0
    #visualizeImg(img=imgs[i],lbl=lbls[i])
    #visualizeImg(img=d[5],lbl=9,reshaped=True)
    kmeans = KMeans(k=20,dist_type='Euclidean',iters=50,num_of_true_lbls=10)
    kmeans.fit(data=transformed_imgs,true_lbls=lbls_copy)    
    kmeans.evaluteClustering()

    # MNIST k=10
    #Objective function value: 1485053.0, Purity: 0.2048, Gini Average: 0.8636386695690841
    
    # MNIST k=5
    #Objective function value: 1500369.0, Purity: 0.189, Gini Average: 0.8555777734123303
    
    # MNIST k=20
    #Objective function value: 1471162.0, Purity: 0.1993, Gini Average: 0.8580059860829399
    
    # FASHION k=10
    
    # FASHION k=5
    
    # FASHION k=20
    
    # 20NG k=20
    
    # 20NG k=10
    
    # 20NG k=40
    
#     centroids = kmeans.centroids
#     for i in range(len(centroids)):
#            visualizeImg(centroids[i],0,reshaped=True)
     
    

    
    
    

    

   