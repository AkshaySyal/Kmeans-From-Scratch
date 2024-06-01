import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
from KMeans import KMeans

def transform_mnist(data):
        print('lol')
        pass

def transform_fashion(data):
        pass
    
def transform_news_groups(data):
        pass

def visualizeImg(img,lbl):
        i = 3
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

    i=5
    visualizeImg(img=imgs[i],lbl=lbls[i])

    

    

   