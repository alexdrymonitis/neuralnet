"""
This script is copied from chapter 19 from the book
"Neural Networks from Scratch in Python" by Harrison Kinsley & Daniel Kukieła
"""

from zipfile import ZipFile
import os
import urllib
import urllib.request

URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
FILE = 'fashion_mnist_images.zip'
FOLDER = '.'
if not os.path.isfile(FILE):
    print(f'Downloading {URL} and saving as {FILE}...')
    urllib.request.urlretrieve(URL, FILE)

print('Unzipping images...')
with ZipFile(FILE) as zip_images:
    zip_images.extractall(FOLDER)
    print('Done!')
