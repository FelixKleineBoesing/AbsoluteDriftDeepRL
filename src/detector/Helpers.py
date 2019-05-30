import os
from scipy.misc import imread
import logging
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

    images_array = []
    all_labels = []
    for index, img_path in enumerate(images):
        img = imread(img_path)
        processed_img = preprocessor.preprocess(img)
        numbers = det._return_numbers(processed_img)
        labels = [10 if char == "X" else int(char) for char in names[index]]
        if len(labels) == numbers.shape[0]:
            images_array.append(numbers)
            all_labels.append(numbers)
        else:
            logging.warning("Image {} will be skipped since detected numbers and labels doesn´t fit togther.")

    data = np.concatenate(images_array, axis=0)
    labels = to_categorical(np.array(names), num_classes=11)
    det.train_network(data, labels)



def train_reward_detector():
    pass

if __name__=="__main__":
    read_and_save_images("../../data/img/reward_catching/")