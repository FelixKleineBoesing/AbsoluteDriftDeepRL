import numpy as np
from src.preprocessing.Preprocessor import RewardPreprocessor, Preprocessor
from scipy.misc import imread
import matplotlib.pyplot as plt


class RewardDetector:
    """
    this class uses a minimal neural network to catch the reward from the top left corner
    """
    def __init__(self, preprocessor: Preprocessor):
        assert isinstance(preprocessor, Preprocessor)
        self.preprocessor = preprocessor

    def get_reward(self, img: np.ndarray):
        preprocessed_img = self.preprocessor.preprocess(img)
        # take only min value from top half since we don´t want to have that comma

        reward = 0
        #TODO detect reward from preprocessed image
        return reward

    def _return_number(self, img: np.ndarray):
        min_brightness_cols = np.min(img[:int(0.5 * img.shape[0]), :, 0], 0)

        # init control variables
        symbol_found = False
        indices = []
        for i in range(min_brightness_cols.shape[0] - 2):
            if min_brightness_cols[i, ] - min_brightness_cols[i + 1, ] > 0.3 and not symbol_found:
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

        # TODO cut here when value bigger than  0.8 to cut each nunber in an own np array
        for index in indices:
            plt.imshow(img[:, index["from"]: index["to"], 0], interpolation='none', cmap='gray')
            plt.show()


if __name__=="__main__":
    img_path = "../../data/img/reward_catching/20190505150812_1.jpg"
    det = RewardDetector(RewardPreprocessor((50, 200)))
    img = imread(img_path)
    det.get_reward(img)