"""
This script is copied and modified from chapter 19 from the book
"Neural Networks from Scratch in Python" by Harrison Kinsley & Daniel Kukieła
"""

import os, cv2
import numpy as np
from time import sleep
import sys, random

def load_mnist_dataset(path, class_nr):

    # get a random file from the chosen class
    random_file = random.choice(os.listdir(os.path.join(path, class_nr)))


    # Create a list for samples
    X = []

    # Read the image
    image = cv2.imread(
                os.path.join(path, class_nr, random_file),
                cv2.IMREAD_UNCHANGED)

    # And append it to the list
    X.append(image)

    # Convert the data to proper numpy array and return
    return np.array(X)


def main():
    dir = sys.argv[1] + '/datasets/fashion_mnist_images/predict'
    class_nr = sys.argv[2]
    # Create dataset
    X = load_mnist_dataset(dir, class_nr)
    # Scale and reshape samples
    X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5

    for i in range(len(X)):
        for j in range(len(X[i])):
            print(f"X {X[i][j]}")


if __name__ == "__main__":
    main()
