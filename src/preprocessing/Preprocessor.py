import abc
import numpy as np



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

    def _resize_img(self, img: np.ndarray):
        # resize img to specified range
        from scipy.misc import imresize
        img = imresize(img, self.img_size)
        return img


class RewardPreprocessor(Preprocessor):
    """
    preprocessor that preprocesses image to catch the reward
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
        # TODO inspect image and crop unused  parts (top left bottom is the only area that is useful
        # TODO since the reward is printed there
        img = img[18:68, 300:440, :]
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
