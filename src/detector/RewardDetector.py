import numpy as np
from scipy.misc import imread
import tensorflow as tf
from keras.layers import Dense
import keras
import os

from src.preprocessing.Preprocessor import RewardPreprocessor, Preprocessor


class RewardDetector:
    """
    this class uses a minimal neural network to catch the reward from the top left corner
    """
    def __init__(self, preprocessor: Preprocessor):
        assert isinstance(preprocessor, Preprocessor)
        self.preprocessor = preprocessor
        self._save_file = "../../data/weights/reward/weights_reward.h5"
        # TODO load saved model if existing
        with tf.variable_scope("rewards", reuse=False):
            network = keras.models.Sequential()
            network.add(Dense(64, activation="relu", input_shape=(25, 25)))
            network.add(Dense(11, activation="softmax"))
        self.network = network
        self.network.compile(optimizer="adam", loss="categorical_crossentropy",
                             metrics=["accuracy"])
        self.trained = False
        if os.path.isfile(self._save_file):
            self.network.load_weights(self._save_file)

    def get_reward(self, img: np.ndarray):
        preprocessed_img = self.preprocessor.preprocess(img)

        numbers = self._return_number(preprocessed_img)
        self._predict_numbers(numbers)
        reward = 0
        #TODO detect reward from preprocessed image
        return reward

    def _predict_numbers(self, numbers: np.ndarray):
        assert self.trained, "Network has to be trained on detecting reward first!"
        numbers_pred = self.network(numbers)
        return numbers_pred

    def _return_number(self, img: np.ndarray):
        min_brightness_cols = np.min(img[:int(0.5 * img.shape[0]), :, 0], 0)

        # init control variables
        symbol_found = False
        indices = []
        for i in range(min_brightness_cols.shape[0] - 2):
            if min_brightness_cols[i, ] - min_brightness_cols[i + 1, ] > 0.3 and not symbol_found:
                symbol_found = True
                index = {"from": i}
            if min_brightness_cols[i, ] - min_brightness_cols[i + 2, ] > 0.3 and not symbol_found:
                symbol_found = True
                index = {"from": i}
            if min_brightness_cols[i - 1, ] - min_brightness_cols[i, ] < -0.3 and symbol_found:
                symbol_found = False
                index["to"] = i + 1
                indices.append(index)
            if min_brightness_cols[i - 2, ] - min_brightness_cols[i, ] < -0.3 and symbol_found:
                symbol_found = False
                index["to"] = i + 2
                indices.append(index)

        numbers = []
        for index in indices:
            resized_img = self.preprocessor._resize_img(img[:, index["from"]:index["to"], 0], (25, 25)) / 255.
            numbers.append(resized_img)

        numbers = np.stack(numbers)
        numbers = numbers[1:, :, :]
        return numbers

    def _encode_target(self, targets: list):
        converted_targets = []
        # TODO check if one hot encodign from numpy is a thing

    def _train_network(self, numbers: np.ndarray, label: np.ndarray):
        self.network.fit(numbers, label, epochs=10, batch_size=16)
        self.network.save_weights(self._save_file)
        self.trained = True

if __name__=="__main__":
    img_path = "../../data/img/reward_catching/20190505150812_1.jpg"
    det = RewardDetector(RewardPreprocessor((50, 200)))
    img = imread(img_path)
    det.get_reward(img)