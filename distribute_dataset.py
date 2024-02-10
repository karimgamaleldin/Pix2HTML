import os
import sys
import shutil
import numpy as np
from PIL import Image

'''
Distributes the dataset into 2 files training_set and test_set
'''

TRAINING_SET_NAME = "training_set"
TEST_SET_NAME = "test_set"
output_path = './dataset'
input_path = './dataset/all_data'

def distribute_dataset():
  # Create the training set folder
  if not os.path.exists(f"{output_path}/{TRAINING_SET_NAME}"):
    os.makedirs(f"{output_path}/{TRAINING_SET_NAME}")
  
  # Create the test set folder
  if not os.path.exists(f"{output_path}/{TEST_SET_NAME}"):
    os.makedirs(f"{output_path}/{TEST_SET_NAME}")

  # Get the length of the dataset
  dataset_length = len(os.listdir(output_path)) / 2
  training_set_length = 1500.0
  test_set_length = dataset_length - training_set_length
  print(f"Dataset length: {dataset_length}")  
  print(f"Training set length: {training_set_length}")
  print(f"Test set length: {test_set_length}")

  paths = [] 
  for f in os.listdir(input_path):
    if f.endswith(".png"):
      paths.append(f[:-4])
  print('Number of paths', len(paths))

  # Create the training set and eval set
  i = 0
  for path in paths:
    image_path = f"{input_path}/{path}.png"
    gui_path = f"{input_path}/{path}.gui"
    img = Image.open(image_path)
    img.load()
    img.resize((256, 256))
    img = np.asarray(img, dtype="float32")
    img = img / 255.0
    if i < training_set_length:
      np.savez_compressed(f"{output_path}/{TRAINING_SET_NAME}/{path}.npz", features=img)
      shutil.copyfile(gui_path, f"{output_path}/{TRAINING_SET_NAME}/{path}.gui")
      retrieved_img = np.load(f"{output_path}/{TRAINING_SET_NAME}/{path}.npz")['features']
      assert np.array_equal(retrieved_img, img)
    else:
      np.savez_compressed(f"{output_path}/{TEST_SET_NAME}/{path}.npz", features=img)
      shutil.copyfile(gui_path, f"{output_path}/{TEST_SET_NAME}/{path}.gui")
      retrieved_img = np.load(f"{output_path}/{TEST_SET_NAME}/{path}.npz")['features']
      assert np.array_equal(retrieved_img, img)
    i += 1
    print(f"Processed {i} Training examples")


distribute_dataset() 
