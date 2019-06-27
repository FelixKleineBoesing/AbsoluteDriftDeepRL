import numpy as np
from scipy.misc import imread
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.utils.np_utils import to_categorical
import os
import matplotlib.pyplot as plt
import logging

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
            self.trained = True

    def get_reward(self, img: np.ndarray):
        preprocessed_img_right = self.preprocessor.preprocess(img)
        # TODO WE need to parse left rewards as well to see if reward is lost or gained
        # this is just a real quick exit if there is no

        #numbers_left = self._return_numbers(preprocessed_img_left)
        numbers_right = self._return_numbers(preprocessed_img_right)
        #numbers_left = self._predict_numbers(numbers_left)
        numbers_right = self._predict_numbers(numbers_right)
        
        def _get_label(numbers: np.ndarray):
            indices = np.argmax(numbers, axis=1)
            number_left, multi = "", ""
            multi_reached = False
            for number in indices.tolist():
                if number == 10:
                    multi_reached = True
                    continue
                if not multi_reached:
                    number_left += str(number)
                else:
                    multi += str(number)
            return number_left, multi

        number, multi = _get_label(numbers_right)
        print(number)
        print(multi)
        reward = float(number) * float(multi)
        return reward

    def _predict_numbers(self, numbers: np.ndarray):
        assert self.trained, "Network has to be trained on detecting reward first!"
        if len(numbers.shape) == 3:
            numbers = numbers.reshape(numbers.shape + (1, ))
        numbers_pred = self.network.predict(numbers)
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


def train_detector(image_path: str, det: RewardDetector):
    names = os.listdir(image_path)
    names = names[1:]
    images = [image_path + image for image in names]

    preprocessor = RewardPreprocessor((50, 200))
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
            all_labels += labels
        else:
            logging.warning("Image {} will be skipped since detected numbers and labels doesn´t fit togther.")

    data = np.concatenate(images_array, axis=0)
    data = data.reshape(data.shape + (1,))

    labels = to_categorical(np.array(all_labels), num_classes=11)

    det.train_network(data, labels)


if __name__=="__main__":
    img_path = "../../data/reward_catching/00007_2800X2.jpg"
    det = RewardDetector(RewardPreprocessor((50, 200)))
    train_detector("../../data/reward_catching/", det)

    img = imread(img_path)
    det.get_reward(img)
