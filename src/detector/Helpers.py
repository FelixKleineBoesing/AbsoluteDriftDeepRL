import os
from scipy.misc import imread
import numpy as np
from keras.utils.np_utils import to_categorical

from src.detector.RewardDetector import RewardDetector
from src.preprocessing.Preprocessor import RewardPreprocessor

det = RewardDetector(RewardPreprocessor((50, 200)))
preprocessor = RewardPreprocessor((50, 200))


def read_and_save_images(image_path: str):
    names = os.listdir(image_path)
    images = [image_path + image for image in names]

    names = [name[6:-4] for name in names if name[6:-4] != "_"]

    names = [10 if char == "X" else int(char) for name in names for char in name]

    images_array = []
    for img_path in images:
        img = imread(img_path)
        processed_img = preprocessor.preprocess(img)
        numbers = det._return_numbers(processed_img)
        images_array.append(numbers)

    data = np.concatenate(images_array, axis=0)
    labels = to_categorical(np.array(names), num_classes=11)
    det.train_network(data, labels)



def train_reward_detector():
    pass

if __name__=="__main__":
    read_and_save_images("../../data/img/reward_catching/")