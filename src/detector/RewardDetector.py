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
        plt.imshow(preprocessed_img[:, :, 0], interpolation='none', cmap='gray')
        plt.show()
        reward = 0
        #TODO detect reward from preprocessed image
        return reward


if __name__=="__main__":
    img_path = "../../data/img/reward_catching/20190505150812_1.jpg"
    det = RewardDetector(RewardPreprocessor((50, 200)))
    img = imread(img_path)
    det.get_reward(img)