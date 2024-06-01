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

if __name__ == '__main__':
    train_images = idx2numpy.convert_from_file("dataset/train-images.idx3-ubyte")
    train_images_copy = np.copy(train_images)
    train_labels = idx2numpy.convert_from_file("dataset/train-labels.idx1-ubyte")

    test_images = idx2numpy.convert_from_file("dataset/t10k-images.idx3-ubyte")
    test_images_copy = np.copy(test_images)
    test_labels = idx2numpy.convert_from_file("dataset/t10k-labels.idx1-ubyte")