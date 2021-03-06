import abc
import numpy as np
import cv2


class Preprocessor(abc.ABC):
    """
    takes image and preprocess this for the agent /learner/reward catcher
    """

    def __init__(self, img_size: tuple):
        self.img_size = img_size

    @abc.abstractmethod
    def preprocess(self, img: np.ndarray):
        pass

    @abc.abstractmethod
    def _crop_image(self, img: np.ndarray):
        pass

    def _normalize_img(self, img: np.ndarray):
        img = img.astype('float32') / 255.
        return img

    def _convert_to_grey(self, img: np.ndarray):
        img = img.mean(-1, keepdims=True)
        return img

    def _resize_img(self, img: np.ndarray, img_size: tuple):
        # resize img to specified range
        img = cv2.resize(img, img_size)
        return img


class RewardPreprocessor(Preprocessor):
    """
    preprocessor that preprocesses image to catch the reward
    """

    def __init__(self, img_size: tuple):
        super().__init__(img_size=img_size)

    def preprocess(self, img: np.ndarray):
        left_img, right_img = self._crop_image(img)
        left_img = self._convert_to_grey(left_img)
        right_img = self._convert_to_grey(right_img)
        left_img = self._normalize_img(left_img)
        right_img = self._normalize_img(right_img)
        left_img = self._cut_smaller(left_img)
        right_img = self._cut_smaller(right_img)

        return left_img, right_img

    def _crop_image(self, img: np.ndarray):
        # slice img here (crop top/bottom,  left/right edgeds which are not really neceessary
        left_img = img[18:68, 90:256, :]
        right_img = img[18:68, 300:440, :]

        return left_img, right_img

    def _cut_smaller(self, img: np.ndarray):
        min_brightness_rows = np.min(img[:, :, 0], 1)
        # get rows with black color
        bools = min_brightness_rows < 0.3
        index_min = np.min(np.arange(min_brightness_rows.shape[0])[bools]) - 1
        index_max = np.max(np.arange(min_brightness_rows.shape[0])[bools]) + 1
        img = img[index_min:index_max, :, :]
        return img


class AgentPreprocessor(Preprocessor):
    """
    preprocessor that preprocesses the image as a state for the agent
    """

    def __init__(self, img_size: tuple):
        super().__init__(img_size=img_size)

    def preprocess(self, img: np.ndarray):
        img = self._crop_image(img)
        img = self._resize_img(img)
        img = self._convert_to_grey(img)
        img = self._normalize_img(img)
        return img

    def _crop_image(self, img: np.ndarray):
        # slice img here (crop top/bottom,  left/right edgeds which are not really neceessary
        # TODO inspect image and crop unused  parts
        img = img[:, :, :]
        return img
