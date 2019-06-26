import numpy as np
from scipy.misc import imread
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import os
import matplotlib.pyplot as plt

from src.preprocessing.Preprocessor import RewardPreprocessor, Preprocessor


class RewardDetector:
    """
    this class uses a minimal neural network to catch the reward from the top left corner
    """
    def __init__(self, preprocessor: Preprocessor):
        assert isinstance(preprocessor, Preprocessor)
        self.preprocessor = preprocessor
        self._save_file = "../../data/weights/reward/weights_reward.h5"
        self.network = tf.keras.models.Sequential([
            Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(25, 25, 1)),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation="relu"),
            Dense(11, activation="softmax")])
        self.network.compile(optimizer="adam", loss="categorical_crossentropy",
                             metrics=["accuracy"])
        self.trained = False
        if os.path.isfile(self._save_file):
            self.network.load_weights(self._save_file)

    def get_reward(self, img: np.ndarray):
        preprocessed_img_right, preprocessed_img_left = self.preprocessor.preprocess(img)
        # TODO WE need to parse left rewards as well to see if reward is lost or gained
        # this is just a real quick exit if there is no 

        numbers_left = self._return_numbers(preprocessed_img_left)
        numbers_right = self._return_numbers(preprocessed_img_right)
        numbers_left = self._predict_numbers(numbers_left)
        numbers_right = self._predict_numbers(numbers_right)
        
        def _get_label(numbers: np.ndarray):
            indices = np.argmax(numbers, axis = 1)
            #TODO build number here
            
        reward = 0
        #TODO detect reward from preprocessed image
        return reward

    def _predict_numbers(self, numbers: np.ndarray):
        assert self.trained, "Network has to be trained on detecting reward first!"
        numbers_pred = self.network(numbers)
        return numbers_pred

    def _return_numbers(self, img: np.ndarray):
        min_brightness_cols = np.min(img[:int(0.5 * img.shape[0]), :, 0], 0)

        # init control variables
        symbol_found = False
        indices = []
        for i in range(min_brightness_cols.shape[0] - 2):
            # TODO define start and end of number by other criteria, not difference, since there may be nearly black
            # elements behind the reward. Define it by absolute value instead
            if min_brightness_cols[i, ] - min_brightness_cols[i + 1, ] > 0.3 and not symbol_found:
                symbol_found = True
                index = {"from": i}
                continue
            if min_brightness_cols[i, ] - min_brightness_cols[i + 2, ] > 0.3 and not symbol_found:
                symbol_found = True
                index = {"from": i}
                continue
            if min_brightness_cols[i - 1, ] - min_brightness_cols[i, ] < -0.3 and symbol_found:
                symbol_found = False
                index["to"] = i + 1
                indices.append(index)
                continue
            if min_brightness_cols[i - 2, ] - min_brightness_cols[i, ] < -0.3 and symbol_found:
                symbol_found = False
                index["to"] = i + 2
                indices.append(index)
                continue

        numbers = []
        for index in indices:
            resized_img = self.preprocessor._resize_img(img[:, index["from"]:index["to"], 0], (25, 25)) / 255.
            numbers.append(resized_img)
            plt.imshow(resized_img)

        numbers = np.stack(numbers)
        numbers = numbers[1:, :, :]
        return numbers

    def train_network(self, numbers: np.ndarray, label: np.ndarray):
        self.network.fit(numbers, label, epochs=100, batch_size=16)
        self.network.save_weights(self._save_file)
        self.trained = True


if __name__=="__main__":
    img_path = "../../data/img/reward_catching/00007_2800X2.jpg"
    det = RewardDetector(RewardPreprocessor((50, 200)))
    img = imread(img_path)
    det.get_reward(img)