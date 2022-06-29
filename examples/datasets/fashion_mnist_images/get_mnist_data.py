import os, cv2
import numpy as np
from time import sleep
import sys

def load_mnist_dataset(dataset, path):

    # Scan all the directories and create a list of labels
    labels = os.listdir(os.path.join(path, dataset))

    # Create lists for samples and labels
    X = []
    y = []


    # For each label folder
    for label in labels:
        # And for each image in given folder
        for file in os.listdir(os.path.join(path, dataset, label)):
            # Read the image
            image = cv2.imread(
                        os.path.join(path, dataset, label, file),
                        cv2.IMREAD_UNCHANGED)

            # And append it and a label to the lists
            X.append(image)
            y.append(label)

    # Convert the data to proper numpy arrays and return
    return np.array(X), np.array(y).astype('uint8')


# MNIST dataset (train + test)
def create_data_mnist(dataset, path):

    # Load both sets separately
    X, y = load_mnist_dataset(dataset, path)

    # And return all the data
    return X, y

def main():
    dir = sys.argv[1] + '/datasets/fashion_mnist_images'
    dataset = sys.argv[2]
    # Create dataset
    X, y = create_data_mnist(dataset, dir)

    if dataset == 'train':
        # Shuffle the training dataset
        keys = np.array(range(X.shape[0]))
        np.random.shuffle(keys)
        X = X[keys]
        y = y[keys]
    # Scale and reshape samples
    X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5

    for i in range(len(X)):
        for j in range(len(X[i])):
            print(f"X {X[i][j]}")
        print(f"y {y[i]}")
        print(f"row {i}")
        sleep(0.005)


if __name__ == "__main__":
    main()
